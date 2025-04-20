# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass, field # Import field if needed later
from typing import List, Dict, Optional, Any, Literal, Tuple
import math
import traceback

# Third-party imports
import torch
from torch.utils.data import DataLoader # Import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import PeftModel # Import PeftModel for adapter loading
from tqdm import tqdm
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration Dataclass (As provided by user) ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal" # Keep for potential future use
    is_adapter_model: bool = False
    base_model_path_for_adapter: Optional[str] = None

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

MODEL_CONFIGS = [
    # Add the models you want to process here
    # ModelConfig(
    #     name="Llamas-3.2-3B_ENG", # Name of the *fine-tuned* model to start unlearning from
    #     model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG", # Path to the fine-tuned model
    #     is_local=True,
    #     is_adapter_model=False, # Assuming this is a fully fine-tuned model, not just adapters
    # ),
    # ModelConfig(
    #     name="Llamas-3.2-3B_ALL", # Name of the *fine-tuned* model to start unlearning from
    #     model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ALL", # Path to the fine-tuned model
    #     is_local=True,
    #     is_adapter_model=False, # Assuming this is a fully fine-tuned model, not just adapters
    # ),
    ModelConfig(
        name="Qwen2.5-7B-Instruct_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ENG",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="Qwen2.5-7B-Instruct_ALL",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ALL",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="gemma-3-4B-it_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ENG",
        is_local=True,
        is_adapter_model=False,
    ),
    ModelConfig(
        name="gemma-3-4B-it_ALL",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ALL",
        is_local=True,
        is_adapter_model=False,
    ),
]

# --- Unlearning Configuration ---
LANGUAGES_TO_UNLEARN = ["en", "ko", "hi"]
# --- Dataset paths remain the same for the unlearning task ---
DATASET_PATHS = {
    "en": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/eng_forget01.json",
    "ko": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/korean_forget01.json",
    "hi": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/hindi_forget01.json",
    "en_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/eng_retain99.json",
    "ko_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/korean_retain99.json", # Corrected path assuming retain
    "hi_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/hindi_retain99.json",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4 # Define batch size here
MAX_LENGTH = 512 # Define max length here

# GA Hyperparameters (Can be kept constant or made part of ModelConfig if needed per-model)
INITIAL_GA_LEARNING_RATE = 8e-6
INITIAL_GA_ITERATIONS = 10
TARGET_FORGET_QUALITY_THRESHOLD = 2.0
MAX_GA_LEARNING_RATES = { "en": 4e-5, "ko": 2e-5, "hi": 2e-5 }
MIN_GA_LEARNING_RATE = 5e-6
ADJUSTMENT_FACTOR = 1.1

# --- Helper Functions (Unchanged) ---
def load_model_and_tokenizer(model_config: ModelConfig): # Takes ModelConfig now
    """Loads the model and tokenizer based on ModelConfig."""
    logger.info(f"Loading model and tokenizer for: {model_config.name}")
    model_load_path = model_config.model_path
    trust_remote_code = "phi" in model_load_path.lower() or "qwen" in model_load_path.lower() # Example heuristic

    if model_config.is_adapter_model:
        logger.info(f"Loading base model from: {model_config.base_model_path_for_adapter}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_config.base_model_path_for_adapter,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
            trust_remote_code=trust_remote_code,
            local_files_only=model_config.is_local # Use is_local for base model too
            # device_map="auto" might be better here if model is large
        ).to(DEVICE) # Move base model to device first if not using device_map

        logger.info(f"Loading adapter weights from: {model_config.model_path}")
        # Load the PeftModel - adapters are automatically moved to the base model's device
        model = PeftModel.from_pretrained(
            base_model, model_config.model_path, is_trainable=True # Set trainable for GA
        )
        # Optional: Merge adapters if you want a standalone model after loading
        # try:
        #     logger.info("Merging adapter weights...")
        #     model = model.merge_and_unload()
        #     logger.info("Adapters merged successfully.")
        # except Exception as e:
        #     logger.warning(f"Could not merge adapters, continuing with PeftModel: {e}")

        # Use the base model's path for the tokenizer
        tokenizer_load_path = model_config.base_model_path_for_adapter
        logger.info("Adapters loaded.")
    else:
        logger.info(f"Loading standard model from: {model_config.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
            trust_remote_code=trust_remote_code,
            local_files_only=model_config.is_local
        ).to(DEVICE)
        tokenizer_load_path = model_config.model_path
        logger.info("Standard model loaded.")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=trust_remote_code, local_files_only=model_config.is_local)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set EOS token as PAD token.")
            if hasattr(model.config, "pad_token_id"): # Check if attribute exists
                 model.config.pad_token_id = tokenizer.eos_token_id
        else:
            # Add a new pad token if EOS is also missing (rare case)
            logger.warning("EOS token missing, adding a new PAD token '[PAD]'.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer)) # Resize model embeddings
            if hasattr(model.config, "pad_token_id"):
                model.config.pad_token_id = tokenizer.pad_token_id

    tokenizer.padding_side = "left"
    logger.info(f"Tokenizer loaded from {tokenizer_load_path}.")
    return model, tokenizer


def load_and_tokenize_data(tokenizer, file_path, max_length=MAX_LENGTH): # Use global MAX_LENGTH
    """Loads data from a json file and tokenizes it."""
    logging.info(f"Loading and tokenizing data from: {file_path}")
    try:
        # Try loading as jsonl first, then standard json
        try:
            dataset = load_dataset("json", data_files=file_path, split="train", field="data") # Adapt field if needed
            logger.debug(f"Loaded {file_path} assuming standard JSON list structure.")
        except Exception: # Fallback for jsonl or different structure
             logger.warning(f"Could not load {file_path} as standard JSON list, trying different loading methods or checking file format.")
             # Add more robust loading if needed, e.g., trying json lines
             # For now, re-raise or return None if specific structure is expected
             dataset = load_dataset("json", data_files=file_path, split="train") # Original fallback

    except Exception as e:
        logging.error(f"Failed to load dataset from {file_path}: {e}")
        return None

    def tokenize_function(examples):
        # Adapt this based on the actual keys in your TOFU JSON files
        if 'question' in examples and 'answer' in examples:
            # Handle cases where keys might exist but have lists of varying lengths
            # This assumes a 1-to-1 mapping between question and answer in the batch
            qa_pairs = []
            min_len = min(len(examples['question']), len(examples['answer']))
            if len(examples['question']) != len(examples['answer']):
                logger.warning(f"Mismatch lengths for question/answer in batch for {file_path}. Using min length {min_len}.")

            for i in range(min_len):
                 q = examples['question'][i] if examples['question'][i] is not None else ""
                 a = examples['answer'][i] if examples['answer'][i] is not None else ""
                 qa_pairs.append(f"Question: {q}\nAnswer: {a}")

            full_text = qa_pairs

        elif 'text' in examples:
            full_text = examples['text']
        else:
            # Fallback if expected keys are missing
            logging.warning(f"Cannot find 'text' or 'question'/'answer' fields in batch from {file_path}. Using empty string.")
            first_key = list(examples.keys())[0]
            full_text = [""] * len(examples[first_key]) # Ensure list of strings

        # Ensure full_text is a list of strings
        if not isinstance(full_text, list): full_text = [str(full_text)]
        if not all(isinstance(t, str) for t in full_text):
             full_text = [str(t) if t is not None else "" for t in full_text]

        tokenized_inputs = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        # Mask padding tokens
        tokenized_inputs["labels"][tokenized_inputs["attention_mask"] == 0] = -100 # Use attention mask for padding
        return tokenized_inputs

    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        logging.info(f"Data tokenized. Number of samples: {len(tokenized_dataset)}")
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Failed during tokenization or formatting for {file_path}: {e}\n{traceback.format_exc()}")
        return None

def calculate_loss(model, dataloader, desc="Calculating Loss"):
    """Calculates the average loss for the model on the given dataloader."""
    model.eval()
    total_loss = 0
    total_batches = 0
    if dataloader is None:
        logging.warning(f"Dataloader is None in calculate_loss ({desc}). Returning inf")
        return float('inf') # Return infinity if no data to calculate loss on
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False): # leave=False for cleaner logs in loops
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss): # Check for NaN/Inf
                     total_loss += loss.item()
                     total_batches += 1
                else:
                    logging.warning(f"Loss was None, NaN or Inf for a batch during {desc}.")
            except Exception as e:
                logging.error(f"Error during model forward/loss calculation ({desc}): {e}")
                continue
    if total_batches == 0:
        logging.warning(f"No valid batches processed during {desc}. Returning inf")
        return float('inf') # Return infinity if no valid loss calculated
    avg_loss = total_loss / total_batches
    model.train() # Set back to train mode
    return avg_loss


def perform_gradient_ascent(model, tokenizer, forget_dataloader, ga_lr, ga_iterations):
    """Performs Gradient Ascent on the model using the forget data."""
    model.train() # Ensure model is in training mode for GA
    logging.info(f"Starting Gradient Ascent: LR={ga_lr:.2e}, Iterations={ga_iterations}")

    for iteration in range(ga_iterations):
        logging.info(f"GA Iteration {iteration + 1}/{ga_iterations}")
        total_loss_epoch = 0
        batches_processed = 0
        if forget_dataloader is None:
            logging.error("forget_dataloader is None in perform_gradient_ascent. Aborting GA.")
            return model # Return model as is
        for batch in tqdm(forget_dataloader, desc=f"GA Iteration {iteration + 1}", leave=False):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            model.zero_grad()
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                     # NEGATIVE loss for ascent (maximizing the loss)
                     # We want to INCREASE the loss on forget data
                     ascent_loss = -loss
                     ascent_loss.backward() # Calculate gradients of negative loss

                     total_loss_epoch += loss.item() # Log the original loss
                     batches_processed += 1

                     # Apply gradient ascent step
                     with torch.no_grad():
                          for param in model.parameters():
                              if param.grad is not None:
                                  # Ascent step: param = param - lr * (-gradient) = param + lr * gradient
                                  # Note: The gradient calculated is d(-L)/dParam = - dL/dParam.
                                  # So, adding lr * grad means adding lr * (-dL/dParam), which is DESCENT.
                                  # TO PERFORM ASCENT (increase L), we need to SUBTRACT grad:
                                  # param = param - (-lr * grad) = param + lr * grad
                                  # Let's rethink. We want to maximize L. Standard update is theta = theta - lr * dL/dtheta.
                                  # Ascent update is theta = theta + lr * dL/dtheta.
                                  # Our grad is d(-L)/dtheta. So we need theta = theta - ga_lr * d(-L)/dtheta = theta + ga_lr * dL/dtheta.
                                  # Therefore, the original logic param.data += ga_lr * param.grad WAS CORRECT.
                                  # param.grad holds d(-L)/dParam. Adding ga_lr * grad means param - ga_lr * dL/dParam. Oh wait, this is still descent.

                                  # Let's recalculate:
                                  # We want: param_new = param_old + alpha * dL/dParam
                                  # We have: param.grad = d(-L)/dParam = - dL/dParam
                                  # So: dL/dParam = - param.grad
                                  # Substitute: param_new = param_old + alpha * (-param.grad) = param_old - alpha * param.grad
                                  param.data -= ga_lr * param.grad # CORRECTED: Subtract gradient of negative loss for ascent

                else:
                     logging.warning("Loss was None, NaN or Inf, skipping GA update for this batch.")
            except Exception as e:
                 logging.error(f"Error during GA backward/update step: {e}\n{traceback.format_exc()}")
                 continue # Continue to next batch

        if batches_processed > 0:
            avg_loss_epoch = total_loss_epoch / batches_processed
            logging.info(f"GA Iteration {iteration + 1} Average Original Forget Loss: {avg_loss_epoch:.4f}")
        else:
             logging.warning(f"No batches processed in GA iteration {iteration + 1}. Average loss is 0.")
             # Maybe break if no batches processed in an iteration?

    logging.info("Gradient Ascent finished.")
    return model

# --- NEW: Function to encapsulate the unlearning process for one model ---
def run_unlearning_for_model(model_config: ModelConfig, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, output_dir: str):
    """Runs the entire unlearning process for a single loaded model."""
    logger.info(f"--- Starting Unlearning Workflow for Model: {model_config.name} ---")

    results = {
        "model_name": model_config.name,
        "model_path_used": model_config.model_path,
        "is_adapter": model_config.is_adapter_model,
        "base_model_path": model_config.base_model_path_for_adapter,
        "unlearned_languages": LANGUAGES_TO_UNLEARN,
        "output_dir": output_dir,
        "hyperparameters": {
            "initial_ga_lr": INITIAL_GA_LEARNING_RATE,
            "ga_iterations": INITIAL_GA_ITERATIONS,
            "target_forget_quality_threshold (loss)": TARGET_FORGET_QUALITY_THRESHOLD,
            "max_ga_lr_map": MAX_GA_LEARNING_RATES, # Renamed for clarity
            "min_ga_lr": MIN_GA_LEARNING_RATE,
            "adjustment_factor": ADJUSTMENT_FACTOR,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH
        },
        "initial_retain_losses": {},
        "language_steps": []
    }

    # --- Prepare Forget DataLoaders ---
    forget_dataloaders = {}
    available_languages = []
    logger.info("--- Loading Forget Datasets ---")
    for lang in LANGUAGES_TO_UNLEARN:
        if lang not in DATASET_PATHS or not os.path.exists(DATASET_PATHS[lang]):
             logger.error(f"Forget dataset path for language '{lang}' not found. Skipping {lang}.")
             continue
        forget_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[lang])
        if forget_dataset:
            # Use pin_memory and num_workers if available for speed
            forget_dataloaders[lang] = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=min(4, os.cpu_count() // 2), pin_memory=True)
            available_languages.append(lang)
        else:
            logger.error(f"Could not load forget data for language: {lang}. Skipping {lang}.")

    if not available_languages:
        logger.error(f"No forget datasets loaded successfully for model {model_config.name}. Aborting unlearning.")
        return # Stop processing this model

    # --- Prepare Retain DataLoaders and Calculate Initial Losses ---
    retain_dataloaders = {}
    initial_retain_losses = {}
    logger.info("--- Loading Retain Datasets & Calculating Initial Losses ---")
    for lang in available_languages:
        retain_key = f"{lang}_retain"
        if retain_key in DATASET_PATHS and os.path.exists(DATASET_PATHS[retain_key]):
            logger.info(f"Loading retain data for {lang}...")
            retain_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[retain_key])
            if retain_dataset:
                retain_dataloaders[lang] = DataLoader(retain_dataset, batch_size=BATCH_SIZE, num_workers=min(4, os.cpu_count() // 2), pin_memory=True)
                loss_val = calculate_loss(model, retain_dataloaders[lang], desc=f"Calculating Initial Retain Loss ({lang})")
                initial_retain_losses[lang] = loss_val
                logger.info(f"Initial Retain Loss for {lang}: {loss_val:.4f}")
            else:
                logger.warning(f"Failed to load retain dataset for {lang}.")
        else:
            logger.warning(f"Retain dataset path '{retain_key}' not found for {lang}.")
    results["initial_retain_losses"] = initial_retain_losses

    # --- Sequential Unlearning Loop ---
    current_ga_lr = INITIAL_GA_LEARNING_RATE
    current_ga_iterations = INITIAL_GA_ITERATIONS # Keep iterations constant for now

    for i, lang in enumerate(available_languages):
        logger.info(f"\n--- Processing Language: {lang} (Step {i+1}/{len(available_languages)}) ---")
        forget_dataloader = forget_dataloaders[lang]

        step_result = {
            "language": lang,
            "ga_lr_used": current_ga_lr,
            "ga_iterations_used": current_ga_iterations,
            "forget_loss_before_ga": None,
            "forget_loss_after_ga": None,
            "retain_losses_after_ga": {}
        }

        # 1. Measure Forget Loss before GA
        initial_lang_forget_loss = calculate_loss(model, forget_dataloader, desc=f"Calculating Forget Loss ({lang}) BEFORE GA")
        step_result["forget_loss_before_ga"] = initial_lang_forget_loss
        logger.info(f"Forget Loss ({lang}) BEFORE GA: {initial_lang_forget_loss:.4f}")

        # 2. Perform Gradient Ascent
        logger.info(f"Applying GA for {lang} with LR={current_ga_lr:.2e}, Iterations={current_ga_iterations}")
        model = perform_gradient_ascent(model, tokenizer, forget_dataloader, current_ga_lr, current_ga_iterations)

        # 3. Measure Forget Loss after GA
        final_lang_forget_loss = calculate_loss(model, forget_dataloader, desc=f"Calculating Forget Loss ({lang}) AFTER GA")
        step_result["forget_loss_after_ga"] = final_lang_forget_loss
        logger.info(f"Forget Loss ({lang}) AFTER GA: {final_lang_forget_loss:.4f}")

        # --- Adjust GA LR for the *next* language ---
        # Adjust only if it's not the last language
        if i < len(available_languages) - 1:
            next_lang = available_languages[i+1]
            # Get max LR specific to the next language, default if not specified
            current_max_lr = MAX_GA_LEARNING_RATES.get(next_lang, MAX_GA_LEARNING_RATES.get("en", 5e-5)) # Default to 'en' max if specific is missing

            if final_lang_forget_loss < TARGET_FORGET_QUALITY_THRESHOLD:
                # Loss increased sufficiently (remember higher loss means more forgotten)
                # -> Decrease LR for next step to be more conservative
                new_lr = max(current_ga_lr / ADJUSTMENT_FACTOR, MIN_GA_LEARNING_RATE)
                logger.info(f"Forget Loss for {lang} ({final_lang_forget_loss:.4f}) already high. Decreasing LR for {next_lang} to {new_lr:.2e}")
            else:
                # Loss didn't increase enough
                # -> Increase LR for next step to push harder
                new_lr = min(current_ga_lr * ADJUSTMENT_FACTOR, current_max_lr)
                logging.info(f"Forget Loss for {lang} ({final_lang_forget_loss:.4f}) below threshold ({TARGET_FORGET_QUALITY_THRESHOLD:.4f}). Increasing LR for {next_lang} to {new_lr:.2e}")

            current_ga_lr = new_lr
            logger.info(f"Next language ({next_lang}) GA params: LR={current_ga_lr:.2e}, Iterations={current_ga_iterations}")


        # 4. Evaluate Utility on ALL Retain Sets after this step
        logger.info(f"--- Evaluating Retain Losses after unlearning {lang} ---")
        current_retain_losses_step = {}
        for retain_lang, retain_loader in retain_dataloaders.items():
            current_loss = calculate_loss(model, retain_loader, desc=f"Calculating Retain Loss ({retain_lang}) after {lang}")
            current_retain_losses_step[retain_lang] = current_loss
            logger.info(f"  Retain Loss ({retain_lang}): {current_loss:.4f}")
            # Catastrophic forgetting check
            if retain_lang in initial_retain_losses and initial_retain_losses[retain_lang] is not None:
                 initial_loss = initial_retain_losses[retain_lang]
                 if initial_loss > 1e-6 and not math.isinf(initial_loss) and current_loss > initial_loss * 1.5: # Avoid checking if initial loss was 0 or inf
                     logging.warning(f"  Potential catastrophic forgetting detected for {retain_lang}!")
            else:
                 logging.warning(f"  Cannot check catastrophic forgetting for {retain_lang}.")

        step_result["retain_losses_after_ga"] = current_retain_losses_step
        results["language_steps"].append(step_result) # Add results for this language step

    # --- Save the Final Unlearned Model ---
    logger.info(f"Saving final unlearned model for {model_config.name} to {output_dir}")
    try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved successfully to {output_dir}.")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer for {model_config.name}: {e}")

    # --- Save Results to JSON ---
    results_path = os.path.join(output_dir, f"unlearning_results_{model_config.name}.json") # Model-specific results file
    try:
        # Convert potential torch tensors/numpy types in results to standard types for JSON
        def default_serializer(o):
            if isinstance(o, (torch.Tensor, np.number)):
                 return o.item() # Convert tensors/numpy numbers to Python numbers
            elif isinstance(o, (np.ndarray, list, tuple)):
                 return list(o) # Convert numpy arrays to lists
            elif math.isinf(o):
                 return "Infinity" # Represent infinity as a string
            elif math.isnan(o):
                 return "NaN" # Represent NaN as a string
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")

        with open(results_path, 'w', encoding='utf-8') as f:
             json.dump(results, f, indent=4, ensure_ascii=False, default=default_serializer)
        logger.info(f"Unlearning results saved to {results_path}")
    except Exception as e:
        logger.error(f"Failed to save results to JSON for {model_config.name}: {e}")

    logger.info(f"--- Finished Unlearning Workflow for Model: {model_config.name} ---")


# --- Main Execution Loop ---
if __name__ == "__main__":
    logger.info("Starting Multilingual Unlearning Process for Multiple Models...")
    overall_start_time = time.time()

    for config in MODEL_CONFIGS:
        model_start_time = time.time()
        logger.info(f"\n=====================================================")
        logger.info(f"Processing Model: {config.name}")
        logger.info(f"Model Path: {config.model_path}")
        logger.info(f"Is Adapter: {config.is_adapter_model}")
        if config.is_adapter_model:
            logger.info(f"Base Model Path: {config.base_model_path_for_adapter}")
        logger.info(f"=====================================================")

        model = None
        tokenizer = None
        output_dir = f"./unlearned_models/unlearned_model_{config.name}" # Dynamic output dir

        try:
            # Load model and tokenizer for the current config
            model, tokenizer = load_model_and_tokenizer(config)

            # Run the unlearning process for this model
            run_unlearning_for_model(config, model, tokenizer, output_dir)

        except Exception as e:
            logger.error(f"CRITICAL ERROR during processing model {config.name}: {e}")
            logger.error(traceback.format_exc())
            # Optionally save error state for this model
            error_file_path = os.path.join(output_dir if os.path.exists(output_dir) else ".", f"ERROR_{config.name}.log")
            with open(error_file_path, "w") as ef:
                 ef.write(f"Error processing model: {config.name}\n")
                 ef.write(f"Config: {config}\n")
                 ef.write(f"Error: {e}\n")
                 ef.write(traceback.format_exc())
            logger.info(f"Error log saved to {error_file_path}")

        finally:
            # --- Memory Cleanup after each model ---
            logger.info(f"Cleaning up resources for model {config.name}...")
            del model
            del tokenizer
            # Delete base_model if it exists (only for adapter case)
            if 'base_model' in locals() and config.is_adapter_model:
                 del base_model
                 logger.debug("Deleted base_model.")

            gc.collect()
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    logger.info("CUDA cache cleared.")
                except Exception as cuda_e:
                    logger.warning(f"Could not clear CUDA cache: {cuda_e}")
            logger.info(f"Finished cleanup for {config.name}.")
            model_end_time = time.time()
            logger.info(f"Time taken for model {config.name}: {model_end_time - model_start_time:.2f} seconds.")
            # Add a small delay maybe?
            # time.sleep(5)


    overall_end_time = time.time()
    logger.info("\n=====================================================")
    logger.info(f"Finished processing all {len(MODEL_CONFIGS)} models.")
    logger.info(f"Total time: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info("=====================================================")