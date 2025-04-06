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
    prepare_model_for_kbit_training,
    PrefixTuningConfig, # Added for Prefix Tuning
    PromptTuningConfig, # Added for Adapters (using Prompt Tuning)
    TaskType # Added for PEFT configs
)
from peft.utils.other import fsdp_auto_wrap_policy

from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling, # Keep for potential use, though SFTTrainer handles it
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig # SFTTrainer often preferred for instruction tuning

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Common path settings
base_model_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"
# --- MODIFIED: Point data_path to the directory ---
data_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train"
# --- END MODIFICATION ---
base_output_dir = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas"

# Output directory creation function
def create_output_dir(method_name):
    output_dir = os.path.join(base_output_dir, f"{method_name}_TOFU_Llama")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Created output directory: {output_dir}")
    return output_dir

# --- MODIFIED: Updated data loading function ---
def load_and_prepare_data(data_dir_path):
    """
    Loads data from all .json files within the specified directory,
    splits it into train/eval sets, and creates Hugging Face Datasets.
    """
    raw_data = []
    logger.info(f"Loading data from directory: {data_dir_path}")

    # Check if the directory exists
    if not os.path.isdir(data_dir_path):
        logger.error(f"Data directory not found: {data_dir_path}")
        raise FileNotFoundError(f"Data directory not found: {data_dir_path}")

    # List all files in the directory
    try:
        all_files = os.listdir(data_dir_path)
    except OSError as e:
        logger.error(f"Error listing files in directory {data_dir_path}: {e}")
        raise

    # Filter for .json files and load data
    json_files = [f for f in all_files if f.endswith(".json")]
    if not json_files:
        logger.error(f"No .json files found in {data_dir_path}")
        raise FileNotFoundError(f"No .json files found in {data_dir_path}")

    logger.info(f"Found {len(json_files)} JSON files. Loading...")
    for filename in tqdm(json_files, desc="Loading JSON files"):
        file_path = os.path.join(data_dir_path, filename)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data_from_file = json.load(f)
                if isinstance(data_from_file, list):
                    raw_data.extend(data_from_file)
                else:
                    # Handle cases where a JSON file might not contain a list directly
                    # Adjust this logic if your JSON structure is different
                    logger.warning(f"Data in {filename} is not a list (type: {type(data_from_file)}). Assuming it's a single record or needs different handling. Skipping for now.")
                    # If it's a single dictionary, you might want raw_data.append(data_from_file)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {file_path}")
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    if not raw_data:
        logger.error("No data loaded from JSON files.")
        raise ValueError("No data loaded from JSON files.")

    logger.info(f"Loaded a total of {len(raw_data)} records.")

    # Data splitting (90% train, 10% eval)
    random.seed(42)
    random.shuffle(raw_data)
    split_index = int(len(raw_data) * 0.9)

    train_data = raw_data[:split_index]
    eval_data = raw_data[split_index:]

    logger.info(f"Train set size: {len(train_data)}, Eval set size: {len(eval_data)}")

    # Dataset creation function using the text formatting
    def create_dataset(data):
        # Ensure 'question' and 'answer' keys exist, providing defaults if missing
        texts = []
        for item in data:
            question = item.get('question', '')
            answer = item.get('answer', '')
            if not question and not answer:
                logger.warning(f"Record found with missing 'question' and 'answer': {item}")
            texts.append(f"Question: {question} Answer: {answer}")
        if not texts:
             raise ValueError("Created empty text list during dataset creation. Check input data format.")
        return Dataset.from_dict({"text": texts})

    train_dataset = create_dataset(train_data)
    eval_dataset = create_dataset(eval_data)

    return train_dataset, eval_dataset
# --- END MODIFICATION ---


# Common tokenization function (using SFTTrainer's approach)
# SFTTrainer handles tokenization internally if you provide the 'dataset_text_field'
# But we can keep this for potential use with the standard Trainer or for verification
def tokenize_data_manual(tokenizer, train_dataset, eval_dataset):
    """
    Manually tokenizes data if not using SFTTrainer's default handling.
    """
    logger.info("Starting manual tokenization...")
    def tokenize_function(examples):
        tokenized_inputs = tokenizer(
            examples["text"],
            padding="max_length", # Consider 'longest' or False with DataCollator
            truncation=True,
            max_length=512, # Adjust as needed
            # SFTTrainer typically doesn't need labels pre-tokenized like this
            # return_tensors="pt" # Map function usually handles this better without pt
        )
        # For standard Trainer, labels are needed. SFTTrainer handles this.
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
        return tokenized_inputs

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # tokenized_train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) # Set format in Trainer if needed

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    # tokenized_eval_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"]) # Set format in Trainer if needed
    logger.info("Manual tokenization complete.")
    return tokenized_train_dataset, tokenized_eval_dataset


# Common training arguments creation function
def create_training_args(output_dir, use_bf16=True):
    # Check BF16 support
    if use_bf16 and not torch.cuda.is_bf16_supported():
        logger.warning("BF16 is not supported on this GPU. Falling back to FP16.")
        use_bf16 = False

    return TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=200, # Evaluate more frequently if dataset is large or training is long
        learning_rate=2e-4, # Common starting point for AdamW
        per_device_train_batch_size=16, # Adjust based on GPU memory
        per_device_eval_batch_size=16,  # Adjust based on GPU memory
        gradient_accumulation_steps=2, # Effective batch size = 4 * 8 * num_gpus = 32 * num_gpus
        num_train_epochs=5, # Adjust based on convergence
        weight_decay=0.01,
        save_total_limit=3, # Keep only the best and latest checkpoints
        save_strategy="steps",
        save_steps=400, # Save checkpoints regularly
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=100, # Log metrics frequently
        fp16=not use_bf16, # Use FP16 if BF16 is not available/chosen
        bf16=use_bf16, # Use BF16 if available and chosen
        lr_scheduler_type="cosine", # Common scheduler
        warmup_ratio=0.1, # Warmup for the first 10% of steps
        load_best_model_at_end=True, # Load the best model based on eval loss
        metric_for_best_model="eval_loss", # Use eval loss to determine the best model
        report_to="none", # Disable external reporting (like wandb) for this example
        gradient_checkpointing=True, # Saves memory at the cost of a bit more compute time
        optim="adamw_torch", # Recommended optimizer
        # ddp_find_unused_parameters=False # Set to False if not using DDP or if you know parameters are used
    )

# 1. Full Fine-Tuning (Using SFTTrainer)
def full_fine_tuning(base_model_path, train_dataset, eval_dataset):
    method_name = "Full_SFT"
    output_dir = create_output_dir(method_name)
    logger.info(f"--- Starting {method_name} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto", # Automatically distribute model across available GPUs
        # use_flash_attention_2=True, # Enable if package is installed and hardware supports
    )

    # Create training arguments
    training_args = create_training_args(output_dir)

    # Create SFTTrainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",  # Specify the column containing the text data
        max_seq_length=512,         # Ensure sequences are padded/truncated appropriately
        packing=False,              # Set to True potentially for faster training if sequences are short
    )

    # Execute training
    logger.info("Starting full fine-tuning training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # Save model and tokenizer
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model(output_dir)  # Saves the fine-tuned model
    tokenizer.save_pretrained(output_dir)
    logger.info(f"--- Completed {method_name} ---")


# 3. Prefix Tuning (Using SFTTrainer)
def prefix_fine_tuning(base_model_path, train_dataset, eval_dataset):
    method_name = "PrefixTuning_SFT"
    output_dir = create_output_dir(method_name)
    logger.info(f"--- Starting {method_name} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load base model (potentially in 4/8 bit for PEFT)
    # Configure BitsAndBytes for quantization if needed (reduces memory)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        # quantization_config=bnb_config # Enable if using quantization
    )

    # model = prepare_model_for_kbit_training(model) # Use if using quantization

    # Prefix Tuning configuration
    peft_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM, # Specify the task type
        num_virtual_tokens=20,        # Number of virtual tokens to add as prefix
        # prefix_projection=True      # Optional: Use a projection layer
    )

    # Create training arguments
    training_args = create_training_args(output_dir)

    # Create SFTTrainer with PEFT config
    trainer = SFTTrainer(
        model=model,                # Pass the base model
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,    # Pass the PEFT configuration here
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )

    # Execute training
    logger.info("Starting Prefix Tuning training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # Save PEFT adapter model and tokenizer
    logger.info(f"Saving final PEFT model to {output_dir}")
    trainer.save_model(output_dir) # Saves the adapter model config and weights
    tokenizer.save_pretrained(output_dir)
    logger.info(f"--- Completed {method_name} ---")


# 4. Adapters Fine-Tuning (Using Prompt Tuning via PEFT, with SFTTrainer)
# Note: "Adapters" can refer to different techniques. Here we use Prompt Tuning as an example.
def adapters_fine_tuning(base_model_path, train_dataset, eval_dataset):
    method_name = "PromptTuning_SFT" # Renamed for clarity
    output_dir = create_output_dir(method_name)
    logger.info(f"--- Starting {method_name} ---")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
    )

    # model = prepare_model_for_kbit_training(model) # Use if using quantization

    # Adapter configuration (using Prompt Tuning)
    peft_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=20, # Number of tunable virtual tokens
        # prompt_tuning_init="TEXT", # Initialization method
        # prompt_tuning_init_text="Generate a detailed answer based on the context.", # Text for init
        # tokenizer_name_or_path=base_model_path # Required if using TEXT init
    )

    # Create training arguments
    training_args = create_training_args(output_dir)

    # Create SFTTrainer with PEFT config
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config, # Pass the PEFT configuration
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=512,
        packing=False,
    )

    # Execute training
    logger.info("Starting Prompt Tuning (Adapter) training...")
    train_result = trainer.train()
    logger.info("Training finished.")

    # Save PEFT adapter model and tokenizer
    logger.info(f"Saving final PEFT model to {output_dir}")
    trainer.save_model(output_dir) # Saves the adapter model config and weights
    tokenizer.save_pretrained(output_dir)
    logger.info(f"--- Completed {method_name} ---")


# Main execution
def main():
    logger.info("Starting main execution...")
    # Load data from the specified directory
    try:
        train_dataset, eval_dataset = load_and_prepare_data(data_path)
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error(f"Failed to load data: {e}")
        return # Exit if data loading fails

    # Execute each Fine-Tuning method one by one
    methods_to_run = [
        full_fine_tuning,
        # prefix_fine_tuning,  # Uncomment to run
        # adapters_fine_tuning # Uncomment to run
    ]

    for method_func in methods_to_run:
        try:
            method_func(base_model_path, train_dataset, eval_dataset)
        except Exception as e:
            # Log the full traceback for detailed debugging
            logger.exception(f"An error occurred during {method_func.__name__}: {e}")
            # Optionally re-raise if you want the script to stop on error
            # raise
        finally:
            # Clear GPU memory cache after each method
            logger.info("Clearing GPU cache...")
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared.")

    logger.info("Main execution finished.")

if __name__ == "__main__":
    main()