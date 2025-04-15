import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import logging
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os
import gc

# Free GPU memory before loading
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()
gc.collect()

# --- Configuration ---
MODEL_NAME = "/data/courses/2025/class_cse576spring2025_vgupt140/Axolotls/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ENG"
LANGUAGES_TO_UNLEARN = ["en", "ko", "hi"]
DATASET_PATHS = {
    "en": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/eng_forget01.json",
    # "ko": "../DB/TOFU/unlearning/forget01_korean_mistral.json",
    # "hi": "../DB/TOFU/unlearning/forget01_hindi_mistral.json",
    # "retain": "path/to/retain_data.jsonl"
}
OUTPUT_DIR = "./KD_forget01_unlearned_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE = 2e-5
BATCH_SIZE = 1
EPOCHS = 3
KL_TEMPERATURE = 1.0

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model_and_tokenizer(model_name):
    logging.info(f"Loading model from {model_name}")
    #model = AutoModelForCausalLM.from_pretrained(
    #    model_name,
    #    trust_remote_code=True,
    #    torch_dtype=torch.float16,
    #    device_map="auto"
    #)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map={"": "cuda:0"}, max_memory={0: "70GiB"})
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    tokenizer.padding_side = "left"

    # Enable memory optimizations
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    return model, tokenizer

def load_and_tokenize_data(tokenizer, file_path, max_length=512):
    dataset = load_dataset("json", data_files=file_path, split="train")

    def tokenize_function(examples):
        if 'text' in examples:
            full_text = examples['text']
        elif 'question' in examples and 'answer' in examples:
            full_text = [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]
        else:
            full_text = [""] * len(examples[list(examples.keys())[0]])

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

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    return tokenized_dataset

def distill_student_model(student, reference, dataloader, optimizer, temperature=1.0):
    student.train()
    reference.eval()
    for epoch in range(EPOCHS):
        logging.info(f"Epoch {epoch + 1}/{EPOCHS}")
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)

            with torch.no_grad():
                teacher_logits = reference(input_ids=input_ids, attention_mask=attention_mask).logits
                p = F.softmax(teacher_logits / temperature, dim=-1)

            student_logits = student(input_ids=input_ids, attention_mask=attention_mask).logits
            q = F.log_softmax(student_logits / temperature, dim=-1)

            kl_loss = -F.kl_div(q, p, reduction="batchmean") * (temperature ** 2)

            optimizer.zero_grad()
            kl_loss.backward()
            optimizer.step()

def evaluate_loss(model, dataloader):
    model.eval()
    total_loss = 0
    total_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            total_batches += 1
    return total_loss / total_batches if total_batches > 0 else 0

def main():
    logging.info("Loading models and tokenizer")
    reference_model, tokenizer = load_model_and_tokenizer(MODEL_NAME)
    student_model, _ = load_model_and_tokenizer(MODEL_NAME)

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=LEARNING_RATE)

    for lang in LANGUAGES_TO_UNLEARN:
        logging.info(f"\n--- Unlearning Language: {lang} ---")
        if lang not in DATASET_PATHS or not os.path.exists(DATASET_PATHS[lang]):
            logging.warning(f"Data not found for language: {lang}, skipping.")
            continue

        forget_dataset = load_and_tokenize_data(tokenizer, DATASET_PATHS[lang])
        forget_loader = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True)

        initial_loss = evaluate_loss(student_model, forget_loader)
        logging.info(f"Initial forget set loss for {lang}: {initial_loss:.4f}")

        distill_student_model(student_model, reference_model, forget_loader, optimizer, temperature=KL_TEMPERATURE)

        final_loss = evaluate_loss(student_model, forget_loader)
        logging.info(f"Final forget set loss for {lang}: {final_loss:.4f}")

    logging.info("Saving unlearned student model")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    student_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    logging.info("Multilingual knowledge distillation unlearning complete")

if __name__ == "__main__":
    main()