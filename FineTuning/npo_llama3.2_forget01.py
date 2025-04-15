# File: FineTuning/npo_llama3.2_forget01.py

import json
import logging
import random
import os
import traceback

from tqdm import tqdm

import numpy as np
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
base_model_path = "tiiuae/falcon-rw-1b"
forget_data_path = ["DB/TOFU/unlearning/eng_forget01.json"]
output_dir = "Checkpoints/npo_gemma3-4b_forget01"

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
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        try:
            print("compute_loss triggered with inputs:", inputs.keys())

            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            labels = inputs["labels"]

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss

            # NPO: negate the loss
            npo_loss = -1 * loss
            print(f"NPO loss computed: {npo_loss.item()}")
            if npo_loss.requires_grad:
                print("Loss requires grad: Backprop should work.")
            else:
                print("Loss does NOT require grad. Training won't update weights.")
            return (npo_loss, outputs) if return_outputs else npo_loss

        except Exception as e:
            print("ERROR inside compute_loss:", str(e))
            raise

# === Run NPO Training ===
def run_npo_training():
    print("Script has started running...")

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        device_map={"": "cpu"},  # Force load to CPU
        trust_remote_code=True
    )
    logger.info("Model loaded successfully")

    logger.info("Loading and preparing data...")
    train_ds, eval_ds = load_and_prepare_data(forget_data_path)
    train_ds, eval_ds = tokenize_data(tokenizer, train_ds, eval_ds)

    logger.info("Setting up training arguments...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    else:
        logger.info(f"Output directory already exists: {output_dir}")

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
        disable_tqdm=False,
        remove_unused_columns=False
    )

    trainer = NPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    logger.info("Starting NPO training...")
    logger.info(f"Train set size: {len(train_ds)}")
    logger.info(f"Eval set size: {len(eval_ds)}")
    logger.info("Starting actual trainer.train() call...")

    train_result = None
    try:
        train_result = trainer.train()
        logger.info("Trainer.train() completed")
        logger.info(f"Training result: {train_result}")
    except Exception as e:
        logger.error("Training crashed with error:")
        traceback.print_exc()

    if train_result is not None:
        logger.info(f"Final Training result: {train_result}")
        print("Final training output:", train_result)
    else:
        logger.warning("No training result returned. Training may have crashed or exited early.")

    logger.info("Saving model...")
    print("Saving model and tokenizer now...")

    try:
        # Critical changes start here
        with torch.device('cpu'):  # Force CPU context for saving
            # Move model explicitly to CPU (even if already there)
            model = model.to('cpu')
            
            # Save tokenizer first
            tokenizer.save_pretrained(output_dir)
            
            # Save model with proper settings
            model.save_pretrained(
                output_dir,
                safe_serialization=True,  # Prevents corruption
                max_shard_size="2GB",     # Manages large files
                state_dict=model.state_dict()
            )
            
        logger.info("Model and tokenizer saved successfully.")
    except Exception as e:
        logger.error(f"Error while saving model/tokenizer: {e}", exc_info=True)
        raise  # Re-raise to see full error

    logger.info("Checking saved files in output directory...")
    try:
        saved_files = os.listdir(output_dir)
        logger.info(f"Contents of {output_dir}:")
        for f in saved_files:
            logger.info(f" - {f}")
            if f.endswith(('.json', '.bin', '.safetensors')):
                full_path = os.path.join(output_dir, f)
                logger.info(f"   Size: {os.path.getsize(full_path)/1024/1024:.2f} MB")
    except Exception as e:
        logger.warning(f"Could not list directory contents: {e}")


    logger.info("Checking saved files in output directory...")
    try:
        logger.info("Contents of output dir: %s", os.listdir(output_dir))
    except Exception as e:
        logger.warning(f"Could not list directory contents: {e}")

    logger.info("NPO training complete.")

if __name__ == "__main__":
    run_npo_training()
