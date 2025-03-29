# Standard library imports
import json
import logging
import random
import os
import re
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from peft.utils.other import fsdp_auto_wrap_policy

from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# Common path settings
base_model_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"
data_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json"
base_output_dir = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas"

# Output directory creation function
def create_output_dir(method_name):
    output_dir = os.path.join(base_output_dir, f"TOFU_Llama_{method_name}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

# Common data loading and preprocessing function
def load_and_prepare_data(data_path):
    # Load data
    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    # Data splitting (90% train, 10% eval)
    random.seed(42)
    random.shuffle(raw_data)
    split_index = int(len(raw_data) * 0.9)

    train_data = raw_data[:split_index]
    eval_data = raw_data[split_index:]

    # Dataset creation function
    def create_dataset(data):
        texts = [f"Question: {item.get('question', '')} Answer: {item.get('answer', '')}" for item in data]
        return Dataset.from_dict({"text": texts})

    return create_dataset(train_data), create_dataset(eval_data)

# Common tokenization function
def tokenize_data(tokenizer, train_dataset, eval_dataset):
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512,
            return_tensors="pt"
        )
        # `labels`를 `input_ids`와 동일하게 설정
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"]
        return tokenized_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    return tokenized_train_dataset, tokenized_eval_dataset


# Common training arguments creation function
def create_training_args(output_dir):
    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=500,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )


# Common training arguments creation function
def create_training_args_Lora(output_dir):
    return SFTConfig(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    
# 1. Full Fine-Tuning
def full_fine_tuning(base_model_path, train_dataset, eval_dataset):
    output_dir = create_output_dir("FullFineTuning")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Prepare tokenized datasets
    tokenized_train_dataset, tokenized_eval_dataset = tokenize_data(tokenizer, train_dataset, eval_dataset)
    
    # Create training arguments
    training_args = create_training_args(output_dir)
    
    # Execute training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# 2. LoRA Fine-Tuning
def lora_fine_tuning(base_model_path, train_data, val_data):
    output_dir = create_output_dir("LoRA")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    # Optional: Add BitsAndBytes config if using quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # LoRA configuration with comprehensive target modules
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # More comprehensive
    )
    
    # Prepare LoRA model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_params)
    
    # Prepare the train and eval datasets in the correct format
    train_dataset = Dataset.from_dict({
        "text": [f"Question: {item['text'].split('Answer:')[0]} Answer: {item['text'].split('Answer:')[1]}" for item in train_data]
    })
    val_dataset = Dataset.from_dict({
        "text": [f"Question: {item['text'].split('Answer:')[0]} Answer: {item['text'].split('Answer:')[1]}" for item in val_data]
    })
    
    # Create training arguments
    training_args = create_training_args_Lora(output_dir)
    
    # SFTTrainer 초기화 시 tokenizer와 packing 제거
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_params,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# 3. Prefix Tuning
def prefix_fine_tuning(base_model_path, train_dataset, eval_dataset):
    output_dir = create_output_dir("PrefixTuning")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Prefix Tuning configuration
    prefix_config = PrefixTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20,
        prefix_projection=True
    )
    
    # Prepare Prefix model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, prefix_config)
    
    # Prepare tokenized datasets
    tokenized_train_dataset, tokenized_eval_dataset = tokenize_data(tokenizer, train_dataset, eval_dataset)
    
    # Create training arguments
    training_args = create_training_args(output_dir)
    
    # Execute training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# 4. Adapters Fine-Tuning
def adapters_fine_tuning(base_model_path, train_dataset, eval_dataset):
    output_dir = create_output_dir("Adapters")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token is None else tokenizer.pad_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, 
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Current version of Adapter configuration
    from peft import PromptTuningConfig, TaskType, get_peft_model
    
    # Adapter configuration
    adapter_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text="Generate a detailed answer based on the context."
    )
    
    # Prepare Adapters model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, adapter_config)
    
    # Prepare tokenized datasets
    tokenized_train_dataset, tokenized_eval_dataset = tokenize_data(tokenizer, train_dataset, eval_dataset)
    
    # Create training arguments
    training_args = create_training_args(output_dir)
    
    # Execute training
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

# Main execution
def main():
    # Load data
    train_dataset, eval_dataset = load_and_prepare_data(data_path)
    
    # Execute each Fine-Tuning method
    try:
        # full_fine_tuning(base_model_path, train_dataset, eval_dataset)
        lora_fine_tuning(base_model_path, train_dataset, eval_dataset)
        # prefix_fine_tuning(base_model_path, train_dataset, eval_dataset)
        # adapters_fine_tuning(base_model_path, train_dataset, eval_dataset)
    except Exception as e:
        print(f"An error occurred during fine-tuning: {e}")
        raise
    finally:
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()