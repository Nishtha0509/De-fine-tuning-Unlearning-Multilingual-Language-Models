# Standard library imports
import json
import logging
import os
import gc
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Tuple, Sequence # Added Sequence
import math
import traceback
import copy

# Third-party imports
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
# **** datasets 라이브러리에서 concatenate_datasets 임포트 ****
from datasets import load_dataset, concatenate_datasets
from peft import PeftModel
from tqdm import tqdm
import numpy as np
import time # Added time for overall timing

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration Dataclass (Unchanged) ---
@dataclass
class ModelConfig:
    name: str
    model_path: str
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"
    is_adapter_model: bool = False
    base_model_path_for_adapter: Optional[str] = None

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

# --- Define Models to Unlearn (Copied from your script) ---
MODEL_CONFIGS = [
    ModelConfig(
        name="Llamas-3.2-3B_ENG",
        model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Llamas-3.2-3B/Full_TOFU_Llamas-3.2-3B_ENG",
        is_local=True, is_adapter_model=False,
    ),
    ModelConfig(
        name="Qwen2.5-7B-Instruct_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Qwen2.5-7B-Instruct/Full_TOFU_Qwen2.5-7B-Instruct_ENG",
        is_local=True, is_adapter_model=False,
    ),
    ModelConfig(
        name="gemma-3-4B-it_ENG",
        model_path=f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/FineTuning/TOFU_Gemma-3-4B-it/Full_TOFU_Gemma-3-4B-it_ENG",
        is_local=True, is_adapter_model=False,
    ),
]

# --- Unlearning Configuration ---
# **** Define the specific sequences, including single languages and a combined identifier ****
COMBINED_LANG_IDENTIFIER = "en_ko_hi_combined" # Special identifier for the combined dataset run
UNLEARNING_SEQUENCES: List[Tuple[str, ...]] = [
    ("en",), ("ko",), ("hi",), # Single language sequences
    ("en", "ko"), ("en", "ko", "hi"),
    ("en", "hi"), ("en", "hi", "ko"),
    ("ko", "en"), ("ko", "en", "hi"),
    ("ko", "hi"), ("ko", "hi", "en"),
    ("hi", "ko"), ("hi", "ko", "en"),
    ("hi", "en"), ("hi", "en", "ko"),
    (COMBINED_LANG_IDENTIFIER,), # Sequence representing the combined dataset run
]
# Languages needed for individual loading (retain and forget for combining)
LANGUAGES_FOR_LOADING = ["en", "ko", "hi"]

# --- Dataset paths (Using retain_half as specified) ---
DATASET_PATHS = {
    "en": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/eng_forget01.json",
    "ko": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/korean_forget01.json",
    "hi": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning/hindi_forget01.json",
    "en_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train_tmp/eng_retain_99.json",
    "ko_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train_tmp/korean_retain_99.json",
    "hi_retain": "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train_tmp/hindi_retain_99.json",
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
MAX_LENGTH = 512

# Fixed GA Hyperparameters
GA_LEARNING_RATE = 8e-6
GA_ITERATIONS = 10

# Output directory base
BASE_OUTPUT_DIR = "./unlearned_models_sequences_fixedLR_with_combined" # Updated name

# --- Helper Functions (Assumed to be correct and robust from previous version) ---
# load_model_and_tokenizer, load_and_tokenize_data, calculate_loss, perform_gradient_ascent
# Ensure they handle errors, device placement, padding tokens etc. correctly.
# The versions from the previous response should be suitable.
# --- [Include the full code for these helper functions here] ---
def load_model_and_tokenizer(model_config: ModelConfig):
    """Loads the model and tokenizer based on ModelConfig."""
    logger.info(f"Loading model and tokenizer for: {model_config.name}")
    model_load_path = model_config.model_path
    # Handle trust_remote_code for specific model types
    trust_remote_code = "qwen" in model_load_path.lower() or "gemma" in model_load_path.lower()

    model = None
    tokenizer = None

    try:
        if model_config.is_adapter_model:
            logger.info(f"Loading base model from: {model_config.base_model_path_for_adapter}")
            trust_remote_base = "qwen" in model_config.base_model_path_for_adapter.lower() or "gemma" in model_config.base_model_path_for_adapter.lower()
            base_model = AutoModelForCausalLM.from_pretrained(
                model_config.base_model_path_for_adapter,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32,
                trust_remote_code=trust_remote_base,
                local_files_only=model_config.is_local
            ).to(DEVICE)

            logger.info(f"Loading adapter weights from: {model_config.model_path}")
            model = PeftModel.from_pretrained(
                base_model, model_config.model_path, is_trainable=True
            )
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
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_load_path,
            trust_remote_code=trust_remote_code, # Use same flag for tokenizer
            local_files_only=model_config.is_local
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set EOS token as PAD token.")
                # Ensure model config also reflects this if possible
                model_to_configure = model.base_model if hasattr(model, 'base_model') else model
                if hasattr(model_to_configure.config, "pad_token_id"):
                     model_to_configure.config.pad_token_id = tokenizer.eos_token_id
            else:
                logger.warning("EOS token missing, adding a new PAD token '[PAD]'.")
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                # Resize embeddings only if necessary (standard model)
                if not model_config.is_adapter_model:
                    model.resize_token_embeddings(len(tokenizer))
                else: # For adapters, resize the base model's embeddings
                    model.base_model.resize_token_embeddings(len(tokenizer))

                model_to_configure = model.base_model if hasattr(model, 'base_model') else model
                if hasattr(model_to_configure.config, "pad_token_id"):
                    model_to_configure.config.pad_token_id = tokenizer.pad_token_id

        tokenizer.padding_side = "left" # Important for generation/causal LMs
        logger.info(f"Tokenizer loaded from {tokenizer_load_path}.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Failed loading model/tokenizer {model_config.name}: {e}", exc_info=True)
        # Clean up partially loaded components
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return None, None

def load_and_tokenize_data(tokenizer, file_path, max_length=MAX_LENGTH):
    """Loads data from a json file and tokenizes it."""
    logging.info(f"Loading and tokenizing data from: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"Dataset file not found: {file_path}")
        return None
    try:
        # Load assuming list of dicts format common in TOFU
        dataset = load_dataset("json", data_files=file_path, split="train")
    except Exception as e:
        logging.error(f"Failed to load dataset from {file_path}: {e}")
        return None

    # Check if tokenizer is provided
    if tokenizer is None:
        logging.error("Tokenizer is None in load_and_tokenize_data. Cannot tokenize.")
        return None # Cannot proceed without tokenizer

    def tokenize_function(examples):
        prompts = []
        # Adapt based on expected keys in TOFU JSON files ('question', 'answer')
        if 'question' in examples and 'answer' in examples:
            for q, a in zip(examples['question'], examples['answer']):
                q_str = q if q is not None else ""
                a_str = a if a is not None else ""
                # Simple prompt format, adapt if needed
                prompts.append(f"Question: {q_str}\nAnswer: {a_str}")
        elif 'text' in examples: # Fallback
            prompts = [t for t in examples['text'] if t is not None]
        else:
            logging.warning(f"Cannot find 'question'/'answer' or 'text' fields in batch from {file_path}. Using empty data.")
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        if not prompts:
            return {'input_ids': [], 'attention_mask': [], 'labels': []}

        # Tokenize the prompts
        tokenized_inputs = tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        # Create labels (standard for Causal LM: shifted input_ids)
        labels = tokenized_inputs["input_ids"].clone()
        # Mask padding tokens in labels
        labels[tokenized_inputs["attention_mask"] == 0] = -100
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    try:
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names, # Remove original columns
            load_from_cache_file=False # Disable cache to ensure fresh tokenization
        )
        # Filter out empty examples that might result from bad data
        tokenized_dataset = tokenized_dataset.filter(lambda x: len(x['input_ids']) > 0)

        if len(tokenized_dataset) == 0:
            logging.warning(f"Tokenization resulted in an empty dataset for {file_path}.")
            return None

        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        logging.info(f"Data tokenized successfully. Number of samples: {len(tokenized_dataset)}")
        return tokenized_dataset
    except Exception as e:
        logging.error(f"Failed during tokenization or formatting for {file_path}: {e}", exc_info=True)
        return None

def calculate_loss(model, dataloader, desc="Calculating Loss"):
    """Calculates the average loss for the model on the given dataloader."""
    model.eval() # Ensure evaluation mode
    total_loss = 0
    total_batches = 0

    if dataloader is None:
        logging.warning(f"Dataloader is None in calculate_loss ({desc}). Returning inf.")
        return float('inf')
    if len(dataloader) == 0: # Check if dataloader is empty
        logging.warning(f"Dataloader is empty in calculate_loss ({desc}). Returning inf.")
        return float('inf')

    # Determine model device safely
    try:
        model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        logger.warning("Could not reliably determine model device, falling back to specified DEVICE.")
        model_device = DEVICE

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, leave=False):
            try:
                # Move batch data to the model's device
                input_ids = batch['input_ids'].to(model_device)
                attention_mask = batch['attention_mask'].to(model_device)
                labels = batch['labels'].to(model_device)
            except Exception as e:
                logger.error(f"Error moving batch to device in calculate_loss ({desc}): {e}")
                continue # Skip batch

            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                     total_loss += loss.item()
                     total_batches += 1
                else:
                    logging.warning(f"Loss was None, NaN or Inf for a batch during {desc}.")
            except Exception as e:
                logger.error(f"Error during model forward/loss calculation ({desc}): {e}")
                logger.debug(f"Input shapes: ids={input_ids.shape if 'input_ids' in locals() else 'N/A'}, mask={attention_mask.shape if 'attention_mask' in locals() else 'N/A'}")
                continue # Skip batch on error

    if total_batches == 0:
        logging.warning(f"No valid batches processed during {desc}. Returning inf.")
        return float('inf')

    avg_loss = total_loss / total_batches
    # model.train() # DO NOT set back to train here, manage mode outside
    return avg_loss

def perform_gradient_ascent(model, forget_dataloader, ga_lr, ga_iterations):
    """Performs Gradient Ascent on the model using the forget data."""
    model.train() # Ensure model is in training mode for GA
    logger.info(f"Starting Gradient Ascent: LR={ga_lr:.2e}, Iterations={ga_iterations}")

    # Determine model device safely
    try:
        model_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        logger.warning("Could not reliably determine model device, falling back to specified DEVICE.")
        model_device = DEVICE

    for iteration in range(ga_iterations):
        logger.info(f"GA Iteration {iteration + 1}/{ga_iterations}")
        total_loss_epoch = 0
        batches_processed = 0
        if forget_dataloader is None or len(forget_dataloader) == 0:
            logger.error("forget_dataloader is None or empty in perform_gradient_ascent. Skipping GA.")
            model.eval() # Set back to eval mode if GA skipped
            return model

        for batch in tqdm(forget_dataloader, desc=f"GA Iteration {iteration + 1}", leave=False):
            try:
                input_ids = batch['input_ids'].to(model_device)
                attention_mask = batch['attention_mask'].to(model_device)
                labels = batch['labels'].to(model_device)
            except Exception as e:
                logger.error(f"Error moving batch to device during GA: {e}")
                continue # Skip batch

            model.zero_grad()
            try:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                if loss is not None and not torch.isnan(loss) and not torch.isinf(loss):
                     ascent_loss = -loss # We want to maximize original loss L, so minimize -L
                     ascent_loss.backward() # Calculate gradients d(-L)/dTheta

                     total_loss_epoch += loss.item() # Log the original loss
                     batches_processed += 1

                     # Apply gradient ascent step: theta_new = theta_old - learning_rate * grad
                     # where grad = d(-L)/dTheta. This minimizes -L, equivalent to maximizing L.
                     with torch.no_grad():
                          for param in model.parameters():
                              if param.grad is not None:
                                  # Corrected update for ascent: Subtract the gradient of negative loss
                                  param.data -= ga_lr * param.grad

                else:
                     logging.warning("Loss was None, NaN or Inf, skipping GA update for this batch.")
            except Exception as e:
                 logger.error(f"Error during GA backward/update step: {e}", exc_info=True)
                 continue # Continue to next batch

        if batches_processed > 0:
            avg_loss_epoch = total_loss_epoch / batches_processed
            logging.info(f"GA Iteration {iteration + 1} Average Original Forget Loss: {avg_loss_epoch:.4f}")
        else:
             logging.warning(f"No batches processed in GA iteration {iteration + 1}.")

    logger.info("Gradient Ascent finished.")
    model.eval() # Set back to eval mode after GA is done
    return model
# --- [End of Helper Functions] ---


# --- Function to run a SINGLE unlearning sequence ---
def run_single_unlearning_sequence(
    model_config: ModelConfig, # Pass the original config for metadata/reloading
    original_state_dict: Dict[str, torch.Tensor], # Pass the initial state
    tokenizer: AutoTokenizer,
    sequence: Sequence[str], # The sequence tuple, e.g., ('en', 'ko') or ('en_ko_hi_combined',)
    forget_dataloaders: Dict[str, DataLoader], # Should contain individual AND combined loaders
    retain_dataloaders: Dict[str, DataLoader], # Contains individual retain loaders
    initial_retain_losses: Dict[str, float],
    output_dir: str, # Base output dir for THIS sequence run
    fixed_ga_lr: float,
    ga_iterations: int
):
    """Runs the GA unlearning process for a given model config, initial state, and language sequence."""
    sequence_str = "_".join(sequence) # e.g., "en_ko" or "en_ko_hi_combined"
    logger.info(f"--- Running Unlearning Sequence: {' -> '.join(sequence)} for {model_config.name} ---")
    logger.info(f"Output directory for this sequence: {output_dir}")

    # --- 1. Prepare Model Instance for this Sequence ---
    model_for_sequence = None
    try:
        logger.info("Creating and loading model instance for sequence...")
        trust_remote_code = "qwen" in model_config.model_path.lower() or "gemma" in model_config.model_path.lower()
        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float32

        if model_config.is_adapter_model:
            trust_remote_base = "qwen" in model_config.base_model_path_for_adapter.lower() or "gemma" in model_config.base_model_path_for_adapter.lower()
            base_model_seq = AutoModelForCausalLM.from_pretrained(
                model_config.base_model_path_for_adapter,
                torch_dtype=dtype, trust_remote_code=trust_remote_base, local_files_only=model_config.is_local
            ).to(DEVICE)
            # Load original base state if provided and potentially modified by GA
            if original_state_dict:
                 logger.info("Loading original base model state dict...")
                 base_model_seq.load_state_dict({k: v.to(DEVICE) for k, v in original_state_dict.items()})
            # Attach adapter
            model_for_sequence = PeftModel.from_pretrained(
                base_model_seq, model_config.model_path, is_trainable=True
            )
            logger.warning("Adapter model state reset relies on reloading base and adapter.")
        else: # Standard model
            model_for_sequence = AutoModelForCausalLM.from_pretrained(
                model_config.model_path,
                torch_dtype=dtype, trust_remote_code=trust_remote_code, local_files_only=model_config.is_local
            ).to(DEVICE)
            # Load the original state dict
            model_for_sequence.load_state_dict({k: v.to(DEVICE) for k, v in original_state_dict.items()})

        model_for_sequence.eval() # Start in eval mode
        logger.info("Model loaded and state reset for the sequence.")

    except Exception as load_err:
        logger.error(f"Failed to create/load model for sequence {' -> '.join(sequence)}: {load_err}", exc_info=True)
        del model_for_sequence # Ensure cleanup if partially loaded
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return # Cannot proceed

    # --- 2. Prepare Results Structure ---
    sequence_results = {
        "model_name": model_config.name,
        "unlearning_sequence": list(sequence), # Store the sequence tuple as a list
        "sequence_output_dir": output_dir,
        "hyperparameters": {
            "ga_lr": fixed_ga_lr,
            "ga_iterations": ga_iterations,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH
        },
        "initial_retain_losses": initial_retain_losses, # Record initial losses for comparison
        "language_steps": [] # Store results for each step in the sequence
    }

    # --- 3. Execute Unlearning Steps ---
    current_model = model_for_sequence # Use the isolated model instance for this sequence

    # The sequence tuple contains the languages (or the combined identifier) in order
    for i, lang_or_combined_id in enumerate(sequence):
        step_num = i + 1
        logger.info(f"\n--- Processing Step {step_num}/{len(sequence)}: '{lang_or_combined_id}' in sequence {' -> '.join(sequence)} ---")

        # Get the appropriate forget dataloader (individual or combined)
        forget_dataloader = forget_dataloaders.get(lang_or_combined_id)

        if forget_dataloader is None or len(forget_dataloader) == 0:
            logger.error(f"Forget dataloader for '{lang_or_combined_id}' not found or empty. Skipping step.")
            step_result = { "language_step_id": lang_or_combined_id, "status": "skipped_missing_or_empty_data", "step_number": step_num }
            sequence_results["language_steps"].append(step_result)
            continue

        step_result = {
            "language_step_id": lang_or_combined_id, # Identifier for this step (e.g., 'en' or 'en_ko_hi_combined')
            "step_number": step_num,
            "ga_lr_used": fixed_ga_lr,
            "ga_iterations_used": ga_iterations,
            "forget_loss_before_ga": None,
            "forget_loss_after_ga": None,
            "retain_losses_after_ga": {} # Evaluate on ALL retain sets after EACH step
        }

        # Measure Forget Loss before GA for this step
        current_model.eval() # Ensure eval mode
        forget_loss_before = calculate_loss(current_model, forget_dataloader, desc=f"Forget Loss ({lang_or_combined_id}) BEFORE GA step {step_num}")
        step_result["forget_loss_before_ga"] = forget_loss_before
        logger.info(f"Forget Loss ({lang_or_combined_id}) BEFORE GA step {step_num}: {forget_loss_before:.4f}")

        # Perform Gradient Ascent using the selected dataloader
        current_model = perform_gradient_ascent(current_model, forget_dataloader, fixed_ga_lr, ga_iterations)
        # perform_gradient_ascent should leave the model in eval mode

        # Measure Forget Loss after GA for this step
        current_model.eval() # Ensure eval mode
        forget_loss_after = calculate_loss(current_model, forget_dataloader, desc=f"Forget Loss ({lang_or_combined_id}) AFTER GA step {step_num}")
        step_result["forget_loss_after_ga"] = forget_loss_after
        logger.info(f"Forget Loss ({lang_or_combined_id}) AFTER GA step {step_num}: {forget_loss_after:.4f}")
        # Warning if loss didn't increase (handle inf/-inf cases)
        is_before_inf = math.isinf(forget_loss_before)
        is_after_inf = math.isinf(forget_loss_after)
        if not is_before_inf and not is_after_inf and forget_loss_after <= forget_loss_before:
             logger.warning(f"Forget loss did not increase after GA for {lang_or_combined_id} (Before: {forget_loss_before:.4f}, After: {forget_loss_after:.4f}).")
        elif is_before_inf and not is_after_inf:
             logger.info(f"Forget loss decreased from Inf to {forget_loss_after:.4f} after GA for {lang_or_combined_id}.") # This might be expected if initial loss was broken

        # Evaluate Utility on ALL Retain Sets after this step
        # **** This uses the individual retain dataloaders for evaluation ****
        logger.info(f"--- Evaluating Retain Losses after GA step {step_num} ({lang_or_combined_id}) ---")
        current_retain_losses_step = {}
        for retain_lang, retain_loader in retain_dataloaders.items():
            if retain_loader is not None and len(retain_loader) > 0 :
                # Always calculate loss on the individual retain sets
                current_loss = calculate_loss(current_model, retain_loader, desc=f"Retain Loss ({retain_lang}) after step {step_num} ({lang_or_combined_id})")
                current_retain_losses_step[retain_lang] = current_loss
                logger.info(f"  Retain Loss ({retain_lang}): {current_loss:.4f}")
                # Optional: Add catastrophic forgetting check against initial_retain_losses
                initial_loss = initial_retain_losses.get(retain_lang)
                if initial_loss is not None and not math.isinf(initial_loss) and initial_loss > 1e-9 and not math.isinf(current_loss) and current_loss > initial_loss * 1.5:
                     logger.warning(f"  Potential catastrophic forgetting detected for {retain_lang}! Initial: {initial_loss:.4f}, Current: {current_loss:.4f}")
            else:
                 logger.warning(f"Retain loader for {retain_lang} is None or empty, cannot calculate loss.")
                 current_retain_losses_step[retain_lang] = None # Record that it couldn't be calculated

        step_result["retain_losses_after_ga"] = current_retain_losses_step
        sequence_results["language_steps"].append(step_result) # Add results for this step

        # Optional: Cleanup GPU memory more aggressively within the sequence if needed
        # gc.collect()
        # if torch.cuda.is_available(): torch.cuda.empty_cache()

    # --- 4. Save Final Model & Results for Sequence ---
    logger.info(f"Saving final unlearned model for sequence {' -> '.join(sequence)} to {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
        current_model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer for sequence saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save model/tokenizer for sequence {' -> '.join(sequence)}: {e}", exc_info=True)

    results_path = os.path.join(output_dir, f"unlearning_results_sequence_{sequence_str}.json")
    logger.info(f"Saving results JSON for sequence to {results_path}")
    try:
        def default_serializer(o):
             if isinstance(o, (torch.Tensor, np.number)): return o.item()
             if isinstance(o, np.ndarray): return o.tolist()
             if isinstance(o, (float, int)) and (math.isinf(o) or math.isnan(o)): return str(o)
             if isinstance(o, torch.dtype): return str(o)
             try: return str(o)
             except Exception: raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable: {o}")

        with open(results_path, 'w', encoding='utf-8') as f:
             json.dump(sequence_results, f, indent=4, ensure_ascii=False, default=default_serializer)
        logger.info(f"Unlearning results for sequence saved successfully.")
    except Exception as e:
        logger.error(f"Failed to save results JSON for sequence {' -> '.join(sequence)}: {e}", exc_info=True)

    logger.info(f"--- Finished Unlearning Sequence: {' -> '.join(sequence)} ---")
    # Cleanup of model_for_sequence happens in the main loop's finally block


# --- Main Execution Loop ---
if __name__ == "__main__":
    logger.info("Starting Multilingual Unlearning Process with Fixed Sequences (incl. Combined) and Fixed LR...")
    overall_start_time = time.time()

    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    # Main loop over models
    for config in MODEL_CONFIGS:
        model_load_start_time = time.time()
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Base Model: {config.name}")
        logger.info(f"Model Path: {config.model_path}")
        logger.info(f"{'='*60}")

        base_model = None
        tokenizer = None
        original_state_dict = None
        # Dictionaries to hold dataloaders specific to this model's tokenizer
        current_forget_dataloaders = {}
        current_retain_dataloaders = {}

        try:
            # Load the base model and tokenizer ONCE for this config
            base_model, tokenizer = load_model_and_tokenizer(config)
            if base_model is None or tokenizer is None:
                logger.error(f"Failed to load base model/tokenizer for {config.name}. Skipping.")
                continue
            logger.info(f"Base model {config.name} loaded in {time.time() - model_load_start_time:.2f} seconds.")

            # Store the original state dictionary (on CPU)
            logger.info("Storing original model state dictionary on CPU...")
            if config.is_adapter_model:
                 original_state_dict = {k: v.cpu().clone() for k, v in base_model.base_model.state_dict().items()}
                 logger.info("Stored base model state for adapter model.")
            else:
                 original_state_dict = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}
                 logger.info("Stored full model state.")

            base_model.eval() # Ensure base model is in eval mode

            # --- Load, Tokenize, and Create Dataloaders for THIS model's tokenizer ---
            logger.info(f"--- Loading/Tokenizing Datasets & Creating Dataloaders for {config.name} ---")
            valid_loaders = True
            loaded_forget_datasets = {} # Temporarily store loaded datasets for combining

            for lang in LANGUAGES_FOR_LOADING:
                # Forget Data
                forget_path = DATASET_PATHS.get(lang)
                if forget_path:
                    forget_dataset = load_and_tokenize_data(tokenizer, forget_path) # Tokenize now
                    if forget_dataset:
                        loaded_forget_datasets[lang] = forget_dataset # Store for potential combining
                        current_forget_dataloaders[lang] = DataLoader(forget_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
                        logger.info(f"Created forget dataloader for {lang}.")
                    else:
                        logger.error(f"Failed to create forget dataset/dataloader for {lang}.")
                        valid_loaders = False; # break # Make it critical if a forget lang fails
                else:
                    logger.error(f"Forget path for {lang} missing.")
                    valid_loaders = False; # break

                # Retain Data (used for evaluation only)
                retain_key = f"{lang}_retain"
                retain_path = DATASET_PATHS.get(retain_key)
                if retain_path:
                    retain_dataset = load_and_tokenize_data(tokenizer, retain_path)
                    if retain_dataset:
                        current_retain_dataloaders[lang] = DataLoader(retain_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
                        logger.info(f"Created retain dataloader for {lang}.")
                    else:
                        logger.warning(f"Failed to create retain dataloader for {lang}. Retain loss eval for {lang} will be skipped.")
                else:
                    logger.warning(f"Retain path for {retain_key} missing.")

            if not valid_loaders:
                 logger.error(f"One or more essential forget dataloaders failed to load for {config.name}. Skipping sequences.")
                 # Cleanup before skipping to next model
                 del base_model, tokenizer, original_state_dict, current_forget_dataloaders, current_retain_dataloaders
                 gc.collect()
                 if torch.cuda.is_available(): torch.cuda.empty_cache()
                 continue

            # **** Create Combined Forget Dataloader ****
            logger.info("Attempting to create combined forget dataloader...")
            individual_datasets_to_combine = [loaded_forget_datasets.get(lang) for lang in LANGUAGES_FOR_LOADING]
            if all(ds is not None for ds in individual_datasets_to_combine):
                try:
                    combined_forget_dataset = concatenate_datasets(individual_datasets_to_combine)
                    # Create dataloader for the combined dataset
                    current_forget_dataloaders[COMBINED_LANG_IDENTIFIER] = DataLoader(
                        combined_forget_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
                    )
                    logger.info(f"Successfully created combined forget dataloader ('{COMBINED_LANG_IDENTIFIER}') with {len(combined_forget_dataset)} samples.")
                except Exception as combo_e:
                    logger.error(f"Failed to concatenate datasets or create combined dataloader: {combo_e}", exc_info=True)
                    # Decide if this is critical - maybe remove the combined sequence?
                    UNLEARNING_SEQUENCES = [seq for seq in UNLEARNING_SEQUENCES if seq != (COMBINED_LANG_IDENTIFIER,)]
                    logger.warning(f"Removed '{COMBINED_LANG_IDENTIFIER}' sequence due to combination error.")
            else:
                logger.warning("Could not create combined dataset because one or more individual forget datasets failed to load.")
                # Remove the combined sequence if datasets weren't available
                UNLEARNING_SEQUENCES = [seq for seq in UNLEARNING_SEQUENCES if seq != (COMBINED_LANG_IDENTIFIER,)]
                logger.warning(f"Removed '{COMBINED_LANG_IDENTIFIER}' sequence.")


            logger.info(f"--- Dataset Loading and Dataloader Creation Complete for {config.name} ---")


            # Calculate initial retain losses ONCE for the base model
            initial_retain_losses = {}
            logger.info(f"Calculating initial retain losses for {config.name}...")
            for lang, loader in current_retain_dataloaders.items():
                 loss = calculate_loss(base_model, loader, desc=f"Initial Retain Loss ({lang})")
                 initial_retain_losses[lang] = loss
                 logger.info(f"  Initial Retain Loss ({lang}): {loss:.4f}")


            # --- Inner loop over sequences for the CURRENT base model ---
            for sequence in UNLEARNING_SEQUENCES:
                sequence_str = "_".join(sequence)
                sequence_start_time = time.time()
                logger.info(f"\n>>> Starting Sequence Run: {' -> '.join(sequence)} for Model: {config.name} <<<")

                # Define output directory for this specific model and sequence
                sequence_output_dir = os.path.join(BASE_OUTPUT_DIR, f"unlearned_{config.name}_seq_{sequence_str}")

                model_for_sequence = None # Ensure reset for try/finally
                try:
                    # Run the unlearning process for this specific sequence
                    # It will reload the model with original state inside
                    run_single_unlearning_sequence(
                        model_config=config,
                        original_state_dict=original_state_dict,
                        tokenizer=tokenizer,
                        sequence=sequence,
                        forget_dataloaders=current_forget_dataloaders, # Pass dict containing combined loader too
                        retain_dataloaders=current_retain_dataloaders,
                        initial_retain_losses=initial_retain_losses,
                        output_dir=sequence_output_dir,
                        fixed_ga_lr=GA_LEARNING_RATE,
                        ga_iterations=GA_ITERATIONS
                    )

                except Exception as seq_e:
                    logger.error(f"Error during sequence {' -> '.join(sequence)} for model {config.name}: {seq_e}", exc_info=True)
                    error_dir = os.path.join(sequence_output_dir, "error_logs")
                    os.makedirs(error_dir, exist_ok=True)
                    error_file_path = os.path.join(error_dir, f"ERROR_sequence_{sequence_str}.log")
                    # ... (write error log as before) ...
                finally:
                    # Cleanup happens implicitly as model_for_sequence goes out of scope
                    # But force GC and cache clearing after each sequence run
                    del model_for_sequence # Explicitly delete reference if exists
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    logger.info(f"<<< Finished Sequence Run: {' -> '.join(sequence)}. Time: {time.time() - sequence_start_time:.2f} sec. >>>")
                    time.sleep(2)

        except Exception as base_model_e:
            logger.error(f"CRITICAL ERROR processing base model {config.name}: {base_model_e}", exc_info=True)
            # ... (log base model error as before) ...

        finally:
            # Clean up resources associated with the base model config run
            logger.info(f"Cleaning up base model resources for {config.name}...")
            del base_model
            del tokenizer
            del original_state_dict
            del current_forget_dataloaders # Clear loaders for this model
            del current_retain_dataloaders
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            logger.info(f"Base model resource cleanup complete.")
            model_total_time = time.time() - model_load_start_time
            logger.info(f"Total time for base model {config.name} (all sequences): {model_total_time:.2f} seconds.")

    overall_end_time = time.time()
    logger.info(f"\n{'='*60}")
    logger.info(f"Finished processing all models and sequences.")
    logger.info(f"Total execution time: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info(f"Unlearned models saved in subdirectories under: {BASE_OUTPUT_DIR}")
    logger.info(f"{'='*60}")