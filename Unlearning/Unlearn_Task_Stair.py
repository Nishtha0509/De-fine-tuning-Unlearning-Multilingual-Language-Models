import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
import logging
import os
from tqdm import tqdm  # For progress bars
import math

# --- Configuration ---
MODEL_NAME = "microsoft/phi-2"  # Replace with your target model (e.g., "Qwen/Qwen1.5-7B-Chat", "meta-llama/Meta-Llama-3.1-8B")
LANGUAGES_TO_UNLEARN = ["en", "ko", "hi"]  # Order matters for sequential unlearning
DATASET_PATHS = {
    "en": "path/to/forget_data_en.jsonl", # Replace with actual paths
    "ko": "path/to/forget_data_ko.jsonl", # Replace with actual paths
    "hi": "path/to/forget_data_hi.jsonl", # Replace with actual paths
    "retain": "path/to/retain_data.jsonl" # Optional: Path to data that should NOT be forgotten
}
OUTPUT_DIR = "./unlearned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GA Hyperparameters (Initial values, will be adjusted)
INITIAL_GA_LEARNING_RATE = 1e-5
INITIAL_GA_ITERATIONS = 5  # Number of passes over the forget data per language
TARGET_FORGET_QUALITY_THRESHOLD = 2.0 # Example threshold: Aim for loss > 2.0 on forget set
MAX_GA_LEARNING_RATE = 5e-5
MIN_GA_LEARNING_RATE = 1e-6
ADJUSTMENT_FACTOR = 1.5 # How much to increase/decrease LR based on Forget Quality

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions ---

def load_model_and_tokenizer(model_name):
    """Loads the pre-trained model and tokenizer."""
    logging.info(f"Loading model and tokenizer: {model_name}")
    # Trust remote code if necessary for models like Phi, Qwen
    trust_remote_code = "phi" in model_name.lower() or "qwen" in model_name.lower()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32, # Use bfloat16 if supported for efficiency
        trust_remote_code=trust_remote_code
    ).to(DEVICE)
    # Load tokenizer, add padding token if missing (common issue)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left" # Important for Causal LM generation/training
    logging.info("Model and tokenizer loaded.")
    return model, tokenizer

def load_and_tokenize_data(tokenizer, file_path, max_length=512):
    """Loads data from a jsonl file and tokenizes it."""
    logging.info(f"Loading and tokenizing data from: {file_path}")
    try:
        dataset = load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset from {file_path}: {e}")
        return None

    # Basic tokenization - Adapt based on your JSON structure (e.g., QA pairs)
    # This example assumes a 'text' field. Modify if you have 'question'/'answer' etc.
    def tokenize_function(examples):
        # Combine fields if necessary, e.g., f"Question: {q} Answer: {a}"
        if 'text' in examples:
            full_text = examples['text']
        elif 'question' in examples and 'answer' in examples:
             # Simple QA formatting
            full_text = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        else:
            # Fallback or error - adjust as needed
            logging.warning("Cannot find 'text' or 'question'/'answer' fields. Using empty string.")
            full_text = [""] * len(examples[list(examples.keys())[0]]) # Get length from first available column

        # Tokenize
        tokenized_inputs = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # For Causal LM loss calculation, labels are usually shifted inputs
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        # Ignore padding tokens in loss calculation
        tokenized_inputs["labels"][tokenized_inputs["labels"] == tokenizer.pad_token_id] = -100
        return tokenized_inputs

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    logging.info(f"Data tokenized. Number of samples: {len(tokenized_dataset)}")
    return tokenized_dataset

def calculate_loss(model, dataloader):
    """Calculates the average loss for the model on the given dataloader."""
    model.eval() # Set model to evaluation mode
    total_loss = 0
    total_batches = 0
    with torch.no_grad(): # No need to track gradients for evaluation
        for batch in tqdm(dataloader, desc="Calculating Loss"):
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            if loss is not None: # Handle cases where loss might not be returned (shouldn't happen with labels)
                 total_loss += loss.item()
                 total_batches += 1

    if total_batches == 0:
        return 0.0
    avg_loss = total_loss / total_batches
    model.train() # Set back to train mode for GA
    return avg_loss

def perform_gradient_ascent(model, tokenizer, forget_dataloader, ga_lr, ga_iterations):
    """Performs Gradient Ascent on the model using the forget data."""
    model.train() # Ensure model is in training mode (for dropout, etc.)
    
    # Note: We don't use a standard optimizer like AdamW here for pure GA.
    # GA directly modifies parameters using the gradient.
    # If you wanted to use optimizer states, you'd need a custom step function.

    logging.info(f"Starting Gradient Ascent: LR={ga_lr}, Iterations={ga_iterations}")

    for iteration in range(ga_iterations):
        logging.info(f"GA Iteration {iteration + 1}/{ga_iterations}")
        total_loss_epoch = 0
        batches_processed = 0
        for batch in tqdm(forget_dataloader, desc=f"GA Iteration {iteration + 1}"):
            # Move batch to device
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            # --- Core GA Step ---
            model.zero_grad() # Clear previous gradients

            # Forward pass to calculate loss on forget data
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if loss is not None:
                 # Backward pass to calculate gradients (dLoss/dParam)
                 # We scale the loss by -1 because backward() computes gradients for minimization.
                 # Maximizing L is equivalent to minimizing -L.
                 # Alternatively, calculate gradients normally and *add* them in the update step.
                 (-loss).backward() # Calculate gradients for ascent

                 total_loss_epoch += loss.item() # Track the original loss value
                 batches_processed += 1

                 # Parameter Update (Ascent Step)
                 with torch.no_grad(): # Don't track gradient history for the update itself
                      for param in model.parameters():
                          if param.grad is not None:
                              # Move parameters *in the direction* of the gradient (ascent)
                              param.data += ga_lr * param.grad
            else:
                 logging.warning("Loss was None for a batch, skipping GA update for this batch.")


            # Optional: Gradient clipping (can sometimes stabilize GA, though less common than in training)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # --- End Core GA Step ---

        avg_loss_epoch = total_loss_epoch / batches_processed if batches_processed > 0 else 0
        logging.info(f"GA Iteration {iteration + 1} Average Forget Loss: {avg_loss_epoch:.4f}")

    logging.info("Gradient Ascent finished.")
    return model # Return the modified model

# --- Main Unlearning Workflow ---

def main():
    logging.info("Starting Multilingual Unlearning Process...")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # --- Prepare DataLoaders ---
    # You might want to adjust batch_size based on your GPU memory
    batch_size = 4
    forget_dataloaders = {}
    for lang in LANGUAGES_TO_UNLEARN:
        forget_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[lang])
        if forget_dataset:
            forget_dataloaders[lang] = torch.utils.data.DataLoader(forget_dataset, batch_size=batch_size)
        else:
            logging.error(f"Could not load forget data for language: {lang}. Skipping.")
            # Handle this error appropriately - maybe stop the process?

    # Optional: Load retain data for utility evaluation
    retain_dataloader = None
    if "retain" in DATASET_PATHS and os.path.exists(DATASET_PATHS["retain"]):
        retain_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS["retain"])
        if retain_dataset:
            retain_dataloader = torch.utils.data.DataLoader(retain_dataset, batch_size=batch_size)
            initial_retain_loss = calculate_loss(model, retain_dataloader)
            logging.info(f"Initial Retain Set Loss (Utility Proxy): {initial_retain_loss:.4f}")
    else:
        logging.warning("Retain dataset path not provided or not found. Skipping utility evaluation.")


    # --- Sequential Unlearning Loop ---
    current_ga_lr = INITIAL_GA_LEARNING_RATE
    current_ga_iterations = INITIAL_GA_ITERATIONS

    for i, lang in enumerate(LANGUAGES_TO_UNLEARN):
        logging.info(f"\n--- Processing Language: {lang} ---")
        if lang not in forget_dataloaders:
            continue # Skip if data wasn't loaded

        forget_dataloader = forget_dataloaders[lang]

        # 1. Measure Forget Quality *before* GA for this language
        #    (Shows how much previous unlearning might have affected this lang)
        initial_lang_forget_loss = calculate_loss(model, forget_dataloader)
        logging.info(f"Forget Quality (Loss) for {lang} BEFORE GA: {initial_lang_forget_loss:.4f}")

        # 2. Perform Gradient Ascent for the current language
        logging.info(f"Applying GA for {lang} with LR={current_ga_lr}, Iterations={current_ga_iterations}")
        model = perform_gradient_ascent(model, tokenizer, forget_dataloader, current_ga_lr, current_ga_iterations)

        # 3. Measure Forget Quality *after* GA for this language
        final_lang_forget_loss = calculate_loss(model, forget_dataloader)
        logging.info(f"Forget Quality (Loss) for {lang} AFTER GA: {final_lang_forget_loss:.4f}")

        # --- Adjust GA strength for the *next* language based on current results ---
        if i < len(LANGUAGES_TO_UNLEARN) - 1: # Don't adjust after the last language
            if final_lang_forget_loss < TARGET_FORGET_QUALITY_THRESHOLD:
                # Forgetting was insufficient, increase strength for the next lang
                new_lr = min(current_ga_lr * ADJUSTMENT_FACTOR, MAX_GA_LEARNING_RATE)
                logging.info(f"Forget Quality for {lang} below threshold ({TARGET_FORGET_QUALITY_THRESHOLD:.4f}). Increasing LR for next language to {new_lr:.2e}")
                current_ga_lr = new_lr
                # Optionally, also increase iterations, or use a more complex adjustment logic
                # current_ga_iterations = min(current_ga_iterations + 1, MAX_ITERATIONS)
            else:
                 # Forgetting was sufficient or excessive, potentially decrease strength
                 new_lr = max(current_ga_lr / ADJUSTMENT_FACTOR, MIN_GA_LEARNING_RATE)
                 logging.info(f"Forget Quality for {lang} met threshold. Decreasing LR for next language to {new_lr:.2e}")
                 current_ga_lr = new_lr
                 # Optionally, decrease iterations
                 # current_ga_iterations = max(current_ga_iterations -1, MIN_ITERATIONS)

            logging.info(f"Next language GA params: LR={current_ga_lr:.2e}, Iterations={current_ga_iterations}")

        # 4. (Optional) Evaluate Utility on Retain Set after each language
        if retain_dataloader:
            current_retain_loss = calculate_loss(model, retain_dataloader)
            logging.info(f"Retain Set Loss after unlearning {lang}: {current_retain_loss:.4f}")
            if current_retain_loss > initial_retain_loss * 1.5: # Example threshold for catastrophic forgetting
                logging.warning(f"Significant increase in retain loss detected after unlearning {lang}. Potential catastrophic forgetting!")


    # --- Save the Final Unlearned Model ---
    logging.info(f"Saving unlearned model to {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("Unlearning process complete.")


if __name__ == "__main__":
    main()