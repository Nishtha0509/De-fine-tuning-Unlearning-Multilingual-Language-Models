from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import logging
import os
from tqdm import tqdm
import math
import json # json 라이브러리 추가

# --- Configuration ---
BASE = "Full_TOFU_Llamas-3.2-3B_ENG"
MODEL_NAME = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG"
LANGUAGES_TO_UNLEARN = ["en", "ko", "hi"]
# --- 데이터 경로 수정: 언어별 Retain 경로 명시 ---
DATASET_PATHS = {
    "en": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/eng_forget01.json",
    "ko": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/korean_forget01.json",
    "hi": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/hindi_forget01.json",
    # --- 각 언어별 Retain 파일 경로 지정 ---
    "en_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/eng_retain99.json",
    "ko_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/korean_retrain99.json", # 경로 오타 수정 (retrain -> retain 가정)
    "hi_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/hindi_retain99.json",
}
OUTPUT_DIR = f"./unlearned_model_{BASE}"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GA Hyperparameters
INITIAL_GA_LEARNING_RATE = 1e-6
INITIAL_GA_ITERATIONS = 10
TARGET_FORGET_QUALITY_THRESHOLD = 1.5
MAX_GA_LEARNING_RATE = 1e-4
MIN_GA_LEARNING_RATE = 1e-6
ADJUSTMENT_FACTOR = 1.5

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Helper Functions (load_model_and_tokenizer, load_and_tokenize_data, calculate_loss, perform_gradient_ascent 는 이전과 동일) ---
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
    """Loads data from a json file and tokenizes it."""
    # NOTE: Assumes input is standard JSON list, not JSON Lines. Adjust if needed.
    logging.info(f"Loading and tokenizing data from: {file_path}")
    try:
        dataset = load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset from {file_path}: {e}")
        return None

    def tokenize_function(examples):
        if 'text' in examples:
            full_text = examples['text']
        elif 'question' in examples and 'answer' in examples:
            full_text = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        else:
            logging.warning(f"Cannot find 'text' or 'question'/'answer' fields in {file_path}. Using empty string.")
            first_key = list(examples.keys())[0]
            full_text = [""] * len(examples[first_key])

        tokenized_inputs = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].clone()
        tokenized_inputs["labels"][tokenized_inputs["labels"] == tokenizer.pad_token_id] = -100
        return tokenized_inputs

    try:
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        logging.info(f"Data tokenized. Number of samples: {len(tokenized_dataset)}")
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Failed during tokenization or formatting for {file_path}: {e}")
        return None

def calculate_loss(model, dataloader, desc="Calculating Loss"): # Added desc parameter
    """Calculates the average loss for the model on the given dataloader."""
    model.eval()
    total_loss = 0
    total_batches = 0
    if dataloader is None: # Handle case where dataloader might not exist
        logging.warning(f"Dataloader is None in calculate_loss ({desc}). Returning 0.0")
        return 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc): # Use desc here
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is not None:
                     total_loss += loss.item()
                     total_batches += 1
                else:
                    logging.warning(f"Loss was None for a batch during {desc}.")
            except Exception as e:
                logging.error(f"Error during model forward/loss calculation ({desc}): {e}")
                # Optionally skip batch or re-raise error depending on desired behavior
                continue
    if total_batches == 0:
        logging.warning(f"No batches processed successfully during {desc}. Returning 0.0")
        return 0.0
    avg_loss = total_loss / total_batches
    model.train()
    return avg_loss

def perform_gradient_ascent(model, tokenizer, forget_dataloader, ga_lr, ga_iterations):
    """Performs Gradient Ascent on the model using the forget data."""
    model.train()
    logging.info(f"Starting Gradient Ascent: LR={ga_lr}, Iterations={ga_iterations}")
    # iteration_losses = [] # Store losses for each iteration if needed

    for iteration in range(ga_iterations):
        logging.info(f"GA Iteration {iteration + 1}/{ga_iterations}")
        total_loss_epoch = 0
        batches_processed = 0
        if forget_dataloader is None:
            logging.error("forget_dataloader is None in perform_gradient_ascent. Aborting iteration.")
            break
        for batch in tqdm(forget_dataloader, desc=f"GA Iteration {iteration + 1}"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            model.zero_grad()
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is not None:
                     loss.backward() # Calculate dL/dParam (for ascent, we ADD this gradient)
                     total_loss_epoch += loss.item()
                     batches_processed += 1
                     with torch.no_grad():
                          for param in model.parameters():
                              if param.grad is not None:
                                  # Perform Ascent: Add gradient
                                  param.data += ga_lr * param.grad
                else:
                     logging.warning("Loss was None for a batch, skipping GA update for this batch.")
            except Exception as e:
                 logging.error(f"Error during GA backward/update step: {e}")
                 # Decide whether to continue or break iteration
                 continue

        if batches_processed > 0:
            avg_loss_epoch = total_loss_epoch / batches_processed
            logging.info(f"GA Iteration {iteration + 1} Average Forget Loss: {avg_loss_epoch:.4f}")
            # iteration_losses.append(avg_loss_epoch) # Optional: track iteration losses
        else:
             logging.warning(f"No batches processed in GA iteration {iteration + 1}. Average loss is 0.")


    logging.info("Gradient Ascent finished.")
    # You could return iteration_losses as well if detailed tracking is needed
    return model
# --- Main Unlearning Workflow ---

def main():
    logging.info("Starting Multilingual Unlearning Process...")

    # --- Setup Results Dictionary (구조 변경) ---
    results = {
        "model_name": MODEL_NAME,
        "unlearned_languages": LANGUAGES_TO_UNLEARN,
        "output_dir": OUTPUT_DIR,
        "hyperparameters": {
            "initial_ga_lr": INITIAL_GA_LEARNING_RATE,
            "ga_iterations": INITIAL_GA_ITERATIONS,
            "target_forget_quality_threshold (loss)": TARGET_FORGET_QUALITY_THRESHOLD,
            "max_ga_lr": MAX_GA_LEARNING_RATE,
            "min_ga_lr": MIN_GA_LEARNING_RATE,
            "adjustment_factor": ADJUSTMENT_FACTOR,
        },
        "initial_retain_losses": {}, # 언어별 초기 retain 손실 저장
        "language_steps": []
    }

    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

    # --- Prepare Forget DataLoaders ---
    batch_size = 4
    forget_dataloaders = {}
    available_languages = [] # 실제로 로드된 언어만 추적
    for lang in LANGUAGES_TO_UNLEARN:
        if lang not in DATASET_PATHS or not os.path.exists(DATASET_PATHS[lang]):
             logging.error(f"Forget dataset path for language '{lang}' not found or invalid. Skipping.")
             continue
        forget_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[lang])
        if forget_dataset:
            forget_dataloaders[lang] = torch.utils.data.DataLoader(forget_dataset, batch_size=batch_size)
            available_languages.append(lang) # 로드 성공한 언어 추가
        else:
            logging.error(f"Could not load forget data for language: {lang}. Skipping.")

    # --- Prepare Retain DataLoaders (언어별 로딩) ---
    retain_dataloaders = {}
    initial_retain_losses = {}
    logging.info("--- Loading Retain Datasets ---")
    # LANGUAGES_TO_UNLEARN 에 있는 언어들에 대해 retain 데이터 로드 시도
    for lang in available_languages: # Forget 데이터가 로드된 언어에 대해서만 retain 시도
        retain_key = f"{lang}_retain"
        if retain_key in DATASET_PATHS and os.path.exists(DATASET_PATHS[retain_key]):
            logging.info(f"Loading retain data for {lang}...")
            retain_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[retain_key])
            if retain_dataset:
                retain_dataloaders[lang] = torch.utils.data.DataLoader(retain_dataset, batch_size=batch_size)
                # 초기 Retain 손실 계산
                loss_val = calculate_loss(model, retain_dataloaders[lang], desc=f"Calculating Initial Retain Loss ({lang})")
                initial_retain_losses[lang] = loss_val
                logging.info(f"Initial Retain Loss for {lang}: {loss_val:.4f}")
            else:
                logging.warning(f"Failed to load retain dataset for {lang}. Skipping utility evaluation for this language.")
        else:
            logging.warning(f"Retain dataset path '{retain_key}' not found or invalid for {lang}. Skipping utility evaluation for this language.")
    results["initial_retain_losses"] = initial_retain_losses # 초기값 저장

    # --- Sequential Unlearning Loop ---
    current_ga_lr = INITIAL_GA_LEARNING_RATE
    current_ga_iterations = INITIAL_GA_ITERATIONS

    # 실제로 데이터 로더가 준비된 언어들에 대해서만 루프 실행
    for i, lang in enumerate(available_languages):
        logging.info(f"\n--- Processing Language: {lang} (Step {i+1}/{len(available_languages)}) ---")
        # forget_dataloader 는 available_languages 에 있으므로 항상 존재함
        forget_dataloader = forget_dataloaders[lang]

        step_result = {
            "language": lang,
            "ga_lr_used": current_ga_lr,
            "ga_iterations_used": current_ga_iterations,
            "forget_loss_before_ga": None,
            "forget_loss_after_ga": None,
            "retain_losses_after_ga": {} # 변경: 언어별 retain 손실 저장
        }

        # 1. Measure Forget Quality *before* GA
        initial_lang_forget_loss = calculate_loss(model, forget_dataloader, desc=f"Calculating Initial Forget Loss ({lang})")
        step_result["forget_loss_before_ga"] = initial_lang_forget_loss
        logging.info(f"Forget Quality (Loss) for {lang} BEFORE GA: {initial_lang_forget_loss:.4f}")

        # 2. Perform Gradient Ascent
        logging.info(f"Applying GA for {lang} with LR={current_ga_lr}, Iterations={current_ga_iterations}")
        model = perform_gradient_ascent(model, tokenizer, forget_dataloader, current_ga_lr, current_ga_iterations)

        # 3. Measure Forget Quality *after* GA
        final_lang_forget_loss = calculate_loss(model, forget_dataloader, desc=f"Calculating Final Forget Loss ({lang})")
        step_result["forget_loss_after_ga"] = final_lang_forget_loss
        logging.info(f"Forget Quality (Loss) for {lang} AFTER GA: {final_lang_forget_loss:.4f}")

        # --- Adjust GA strength for the *next* language ---
        # 마지막 언어가 아니며, 다음 언어가 실제로 사용 가능한 경우에만 LR 조정
        if i < len(available_languages) - 1:
             next_lang_available = (i + 1 < len(available_languages)) # 다음 언어 인덱스 유효성 체크
             if next_lang_available:
                if final_lang_forget_loss < TARGET_FORGET_QUALITY_THRESHOLD:
                    new_lr = min(current_ga_lr * ADJUSTMENT_FACTOR, MAX_GA_LEARNING_RATE)
                    logging.info(f"Forget Quality for {lang} below threshold ({TARGET_FORGET_QUALITY_THRESHOLD:.4f}). Increasing LR for next language to {new_lr:.2e}")
                    current_ga_lr = new_lr
                else:
                     new_lr = max(current_ga_lr / ADJUSTMENT_FACTOR, MIN_GA_LEARNING_RATE)
                     logging.info(f"Forget Quality for {lang} met threshold. Decreasing LR for next language to {new_lr:.2e}")
                     current_ga_lr = new_lr
                logging.info(f"Next language GA params: LR={current_ga_lr:.2e}, Iterations={current_ga_iterations}")
             else:
                 logging.info("No further available language to adjust LR for.")


        # 4. Evaluate Utility on ALL available Retain Sets after this language step
        logging.info(f"--- Evaluating Retain Losses after unlearning {lang} ---")
        current_retain_losses_step = {}
        for retain_lang, retain_loader in retain_dataloaders.items():
            current_loss = calculate_loss(model, retain_loader, desc=f"Calculating Retain Loss ({retain_lang}) after {lang}")
            current_retain_losses_step[retain_lang] = current_loss
            logging.info(f"  Retain Loss ({retain_lang}): {current_loss:.4f}")
            # Catastrophic forgetting check (해당 언어의 초기 손실과 비교)
            if retain_lang in initial_retain_losses and initial_retain_losses[retain_lang] is not None:
                 initial_loss_for_lang = initial_retain_losses[retain_lang]
                 # 초기 손실이 0이 아닐 때만 비율 비교
                 if initial_loss_for_lang > 1e-6 and current_loss > initial_loss_for_lang * 1.5:
                     logging.warning(f"  Potential catastrophic forgetting detected for {retain_lang} after unlearning {lang}!")
            else:
                 logging.warning(f"  Cannot check catastrophic forgetting for {retain_lang} as initial loss is unavailable.")

        step_result["retain_losses_after_ga"] = current_retain_losses_step # 단계별 결과에 저장

        results["language_steps"].append(step_result) # Add step results


    # --- Save the Final Unlearned Model ---
    logging.info(f"Saving unlearned model to {OUTPUT_DIR}")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("Unlearning process complete.")

    # --- Save Results to JSON ---
    results_path = os.path.join(OUTPUT_DIR, "unlearning_results.json")
    try:
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        logging.info(f"Unlearning results saved to {results_path}")
    except Exception as e:
        logging.error(f"Failed to save results to JSON: {e}")


if __name__ == "__main__":
    main()