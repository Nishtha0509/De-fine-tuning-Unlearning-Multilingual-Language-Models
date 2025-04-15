import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import gc

# --- Configuration ---
MODEL_NAME = "/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ENG"
LANGUAGES_TO_UNLEARN = ["en"]
DATASET_PATHS = {
    "en": "../DB/TOFU/unlearning/eng_forget01.json"
}
OUTPUT_DIR = "./KD_forget01_unlearned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 1e-5
BATCH_SIZE = 1
EPOCHS = 3
KL_TEMPERATURE = 1.0

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

torch.cuda.empty_cache()
gc.collect()

def load_model_and_tokenizer(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # More stable than float16
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"
    return model, tokenizer

def load_and_tokenize_data(tokenizer, file_path, max_length=256):
    logging.info(f"Loading dataset from: {file_path}")
    dataset = load_dataset("json", data_files=file_path, split="train")

    def tokenize_function(examples):
        if 'text' in examples:
            full_text = examples['text']
        elif 'question' in examples and 'answer' in examples:
            full_text = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        else:
            full_text = [""] * len(examples[list(examples.keys())[0]])

        full_text = [text if text is not None else "" for text in full_text]
        full_text = [text for text in full_text if len(text.strip()) > 0]

        if len(full_text) == 0:
            logging.warning("No non-empty examples found after preprocessing.")
            return {"input_ids": [], "attention_mask": [], "labels": []}

        tokenized_inputs = tokenizer(
            full_text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        labels = tokenized_inputs["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        valid_indices = [(label != -100).sum().item() > 0 for label in labels]
        if not any(valid_indices):
            logging.warning("All tokenized examples had fully masked labels (-100). Skipping batch.")
            return {"input_ids": [], "attention_mask": [], "labels": []}

        return {
            "input_ids": tokenized_inputs["input_ids"][valid_indices],
            "attention_mask": tokenized_inputs["attention_mask"][valid_indices],
            "labels": labels[valid_indices]
        }

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    logging.info(f"Final tokenized dataset size: {len(tokenized_dataset)}")
    return tokenized_dataset

def compute_kl_divergence(student_logits, teacher_logits, temperature=1.0):
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits / temperature, dim=-1).detach()
    teacher_probs = teacher_log_probs.exp().clamp(min=1e-8)  # clamp to avoid log(0)

    kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)
    return kl_div

def distill_student_model(model, dataloader, optimizer, temperature=1.0):
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        model.train()

        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            model.eval()
            with torch.no_grad():
                teacher_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            model.train()
            student_logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

            if torch.isnan(teacher_logits).any() or torch.isinf(teacher_logits).any():
                logging.warning("Teacher logits have NaN or Inf. Skipping batch.")
                continue
            if torch.isnan(student_logits).any() or torch.isinf(student_logits).any():
                logging.warning("Student logits have NaN or Inf. Skipping batch.")
                continue

            kl_loss = compute_kl_divergence(student_logits, teacher_logits, temperature=temperature)

            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                logging.warning("KL loss is NaN or Inf. Skipping this batch.")
                continue

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    skipped_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning("Evaluation loss is NaN or Inf. Skipping this batch.")
                skipped_batches += 1
                continue

            total_loss += loss.item()
            total_batches += 1

    if total_batches == 0:
        logging.warning("No valid batches in evaluation. Returning NaN.")
        return float("nan")

    avg_loss = total_loss / total_batches
    logging.info(f"Evaluation completed. Skipped {skipped_batches} batches due to invalid loss.")
    return avg_loss

def main():
    logging.info("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for lang in LANGUAGES_TO_UNLEARN:
        logging.info(f"\n--- Unlearning Language: {lang} ---")
        if lang not in DATASET_PATHS or not os.path.exists(DATASET_PATHS[lang]):
            logging.warning(f"Data not found for language: {lang}, skipping.")
            continue

        forget_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[lang])
        if len(forget_dataset) == 0:
            logging.warning(f"No data to train for language: {lang}, skipping.")
            continue

        forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True)

        initial_loss = evaluate_loss(model, forget_loader)
        logging.info(f"Initial forget set loss for {lang}: {initial_loss:.4f}")

        distill_student_model(model, forget_loader, optimizer, temperature=KL_TEMPERATURE)

        final_loss = evaluate_loss(model, forget_loader)
        logging.info(f"Final forget set loss for {lang}: {final_loss:.4f}")

        torch.cuda.empty_cache()
        gc.collect()

    logging.info("Saving unlearned student model")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("Multilingual knowledge distillation unlearning complete")

if __name__ == "__main__":
    main()
