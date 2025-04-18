# File: FineTuning/npo_llama3.2_forget01.py (Original GEMMA-based version without forced CPU)

import json
import logging
import random
import os

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments
)
from trl import SFTTrainer

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
base_model_path = "google/gemma-2b-it"  # GEMMA-based model
forget_data_path = ["DB/TOFU/unlearning/eng_forget01.json"]
output_dir = "Checkpoints/npo_gemma2b_forget01"

# === Load and Prepare Data ===
def load_and_prepare_data(data_paths):
    raw_data = []
    for path in data_paths:
        with open(path, "r", encoding="utf-8") as f:
            raw_data.extend(json.load(f))
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * 0.9)
    train_data, eval_data = raw_data[:split_idx], raw_data[split_idx:]

    def to_dataset(data):
        return Dataset.from_dict({
            "text": [f"Question: {d['question']} Answer: {d['answer']}" for d in data]
        })

    return to_dataset(train_data), to_dataset(eval_data)

# === Tokenization ===
def tokenize_data(tokenizer, train_ds, eval_ds):
    def tokenize_fn(examples):
        tokens = tokenizer(
            examples["text"], padding="max_length", truncation=True, max_length=512
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    return (
        train_ds.map(tokenize_fn, batched=True, remove_columns=["text"]),
        eval_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    )

# === Custom Trainer to Apply Negative Loss ===
class NPOTrainer(SFTTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

        loss = outputs.loss
        npo_loss = -1 * loss
        print(f"NPO loss computed: {npo_loss.item()}")
        return (npo_loss, outputs) if return_outputs else npo_loss

# === Run NPO Training ===
def run_npo_training():
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    logger.info("Model loaded successfully")

    logger.info("Loading and preparing data...")
    train_ds, eval_ds = load_and_prepare_data(forget_data_path)
    train_ds, eval_ds = tokenize_data(tokenizer, train_ds, eval_ds)

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        max_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False
    )

    trainer = NPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Starting NPO training...")
    trainer.train()
    logger.info("Saving model and tokenizer...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    run_npo_training()
