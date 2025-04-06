# Standard library imports
import json
import logging
import os
import re # Regular expression import added
import gc
import time
import traceback
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal
from io import StringIO
import contextlib
import math # For ceil

# Third-party imports
import torch
# Removed BitsAndBytesConfig from this import
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from bert_score import score as bert_score_calculate # Import specific function for clarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Configuration ---
@dataclass
class ModelConfig:
    name: str
    model_path: str # Path to the base model OR the adapter checkpoint directory
    is_local: bool = True
    model_type: Literal["causal", "encoder", "encoder-decoder"] = "causal"
    is_adapter_model: bool = False # Flag to indicate if model_path points to LoRA adapters
    base_model_path_for_adapter: Optional[str] = None # Path to the base model if is_adapter_model is True

    def __post_init__(self):
        if self.is_adapter_model and not self.base_model_path_for_adapter:
            raise ValueError(f"Model '{self.name}' is marked as adapter model, but 'base_model_path_for_adapter' is not provided.")
        if self.is_adapter_model and self.base_model_path_for_adapter == self.model_path:
             raise ValueError(f"For adapter model '{self.name}', 'base_model_path_for_adapter' cannot be the same as 'model_path'.")

# --- Define Models to Evaluate ---
BASE_LLAMA_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b" # Example Base Path

MODEL_CONFIGS = [
    ModelConfig(
        name="Llama3.2_Origin",
        model_path=BASE_LLAMA_PATH,
        is_local=True,
        is_adapter_model=False
    ),
    ModelConfig(
        name="FULL_TOFU_Llama_ENG",
        model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas/Full_TOFU_Llama_ENG",
        is_local=True,
        is_adapter_model=False
    ),
    ModelConfig(
        name="Full_TOFU_Llama_ENG_KR_HIN", # This seems to be an Adapter model based on path structure, assuming it is.
        model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas/Full_TOFU_Llama_ENG_KR_HIN",
        is_local=True,
        is_adapter_model=True, # Set to True
        base_model_path_for_adapter=BASE_LLAMA_PATH # Specify base model
    ),
    # Example for LoRA (uncomment and adjust path if needed)
    # ModelConfig(
    #     name="LoRA_TOFU_Llama_ENG",
    #     model_path="/path/to/your/lora/adapter/for/eng", # Adjust this path
    #     is_local=True,
    #     is_adapter_model=True,
    #     base_model_path_for_adapter=BASE_LLAMA_PATH
    # ),
]

# --- Evaluation Configuration ---
DATA_DIRECTORIES = [
    "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train",
    "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/unlearning"
]
OUTPUT_DIR = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Evaluation_Results" # Changed output dir
MAX_NEW_TOKENS = 150 # Max tokens to generate for an answer
BERT_SCORE_MODEL_TYPE = "bert-base-multilingual-cased" # Recommended for multilingual comparison
GENERATION_BATCH_SIZE = 32  # Batch for evaluation
BATCH_SIZE_BERT_SCORE = 16 # Batch size for BERTScore calculation for efficiency

# --- Quantization Configuration Removed ---

# --- Helper Functions ---

def load_model_and_tokenizer(config: ModelConfig, device: torch.device):
    """Loads the model and tokenizer based on the configuration without bitsandbytes quantization."""
    logger.info(f"Loading model: {config.name} from {config.model_path} (Quantization Disabled)")
    model = None
    tokenizer = None
    load_path = config.model_path
    base_load_path = config.base_model_path_for_adapter if config.is_adapter_model else config.model_path

    try:
        if config.is_adapter_model:
            logger.info(f"Loading base model for adapter from: {config.base_model_path_for_adapter}")
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_path_for_adapter,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info(f"Loading adapter weights from: {config.model_path}")
            model = PeftModel.from_pretrained(
                base_model,
                config.model_path,
                device_map="auto",
            )
            tokenizer_load_path = config.base_model_path_for_adapter
            logger.info(f"Loading tokenizer from base model path: {tokenizer_load_path}")
        else:
             logger.info(f"Loading full model from: {config.model_path}")
             model = AutoModelForCausalLM.from_pretrained(
                 config.model_path,
                 torch_dtype=torch.bfloat16,
                 device_map="auto",
                 trust_remote_code=True
             )
             tokenizer_load_path = config.model_path
             logger.info(f"Loading tokenizer from model path: {tokenizer_load_path}")

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path, trust_remote_code=True)

        if tokenizer.pad_token is None:
            logger.warning("Tokenizer does not have a pad token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Ensure model config pad_token_id is also set, handling potential PeftModel wrapping
            model_to_configure = model
            if hasattr(model, 'base_model'): # If it's a PeftModel, configure the base model
                model_to_configure = model.base_model
            if hasattr(model_to_configure, 'config') and hasattr(model_to_configure.config, 'pad_token_id'):
                model_to_configure.config.pad_token_id = tokenizer.eos_token_id

        model.eval() # Set model to evaluation mode
        logger.info(f"Model '{config.name}' and tokenizer loaded successfully.")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model {config.name}: {e}")
        logger.error(traceback.format_exc())
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None, None

# --- BATCHED Generation Function ---
def generate_answers_batch(model, tokenizer, questions: List[str], max_new_tokens: int) -> List[Optional[str]]:
    """Generates answers for a batch of questions using the model."""
    prompts = [f"Question: {q}\nAnswer:" for q in questions]
    generated_answers = []

    try:
        # Determine the device of the model (important if using device_map='auto')
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            logger.warning("Could not determine model device, assuming CPU.")
            model_device = torch.device("cpu") # Fallback

        # Tokenize the batch of prompts
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,       # Pad sequences to the longest in the batch
            truncation=True,
            max_length=512 # Set a reasonable max_length for input to avoid excessive padding
        ).to(model_device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        # Decode each generated sequence in the batch
        input_length = inputs['input_ids'].shape[1]
        for i in range(outputs.shape[0]):
            generated_ids = outputs[i, input_length:]
            try:
                answer = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                generated_answers.append(answer)
            except Exception as decode_err:
                 logger.error(f"Error decoding answer for question batch item {i}: {decode_err}")
                 generated_answers.append(None) # Append None if decoding fails

        # Ensure the number of answers matches the number of questions
        if len(generated_answers) != len(questions):
            logger.error(f"Mismatch in generated answers count ({len(generated_answers)}) vs questions count ({len(questions)}). Filling missing with None.")
            # Pad with None if something went wrong during generation/decoding loop
            generated_answers.extend([None] * (len(questions) - len(generated_answers)))

        return generated_answers

    except Exception as e:
        logger.error(f"Error during batch generation for questions starting with '{questions[0][:50]}...': {e}")
        logger.error(traceback.format_exc())
        # Return a list of Nones with the same length as the input questions batch
        return [None] * len(questions)


def calculate_bert_score_batch(predictions: List[str], references: List[str], device: torch.device) -> Optional[Dict[str, List[float]]]:
    """Calculates BERTScore (P, R, F1) for batches of predictions and references."""
    # --- This function remains the same, as it already handles batches ---
    if not predictions or not references:
        logger.warning("Empty predictions or references list passed to BERTScore.")
        return None
    try:
        valid_indices = [i for i, p in enumerate(predictions) if p is not None and p.strip() != ""] # Also filter empty strings
        if not valid_indices:
             logger.warning("No valid predictions found after filtering Nones/empty strings.")
             return {'P': [], 'R': [], 'F1': []}

        filtered_predictions = [predictions[i] for i in valid_indices]
        # Ensure references align with filtered predictions
        filtered_references = [references[i] for i in valid_indices if i < len(references)]

        # Check if lengths match after filtering potentially mismatched refs
        if len(filtered_predictions) != len(filtered_references):
             logger.error(f"Length mismatch after filtering for BERTScore: preds={len(filtered_predictions)}, refs={len(filtered_references)}. Indices: {valid_indices}")
             # Attempt to proceed with the minimum length, or return empty
             min_len = min(len(filtered_predictions), len(filtered_references))
             if min_len == 0: return {'P': [], 'R': [], 'F1': []}
             filtered_predictions = filtered_predictions[:min_len]
             filtered_references = filtered_references[:min_len]


        if not filtered_predictions:
             logger.warning("No valid predictions remaining after filtering.")
             return {'P': [], 'R': [], 'F1': []}

        bert_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        P, R, F1 = bert_score_calculate(
            filtered_predictions,
            filtered_references,
            model_type=BERT_SCORE_MODEL_TYPE,
            lang="en",
            verbose=False,
            device=bert_device,
            batch_size=BATCH_SIZE_BERT_SCORE # Use dedicated BERTScore batch size
        )
        return {'P': P.tolist(), 'R': R.tolist(), 'F1': F1.tolist()}

    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}")
        logger.error(traceback.format_exc())
        return None


def process_file(model_config: ModelConfig, model, tokenizer, input_filepath: str, output_dir: str, device: torch.device):
    """Processes a single JSON data file using batched generation."""
    filename = os.path.basename(input_filepath)
    logger.info(f"Processing file: {filename} for model: {model_config.name} using batch size {GENERATION_BATCH_SIZE}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file {input_filepath}: {e}")
        return

    results = []
    all_questions = []
    all_ground_truths = []
    all_generated_answers = []

    # 1. Collect all questions and ground truths first
    for item in data:
        question = item.get("question")
        ground_truth = item.get("answer")
        if not question or not ground_truth:
            logger.warning(f"Skipping item due to missing 'question' or 'answer' in {filename}: {item}")
            continue
        all_questions.append(question)
        all_ground_truths.append(ground_truth)

    if not all_questions:
        logger.warning(f"No valid question/answer pairs found in {filename}. Skipping.")
        return

    # 2. Generate answers in batches
    logger.info(f"Generating answers for {len(all_questions)} questions in batches of {GENERATION_BATCH_SIZE}...")
    num_batches = math.ceil(len(all_questions) / GENERATION_BATCH_SIZE)

    for i in tqdm(range(num_batches), desc=f"Generating Batches - {filename}"):
        batch_start = i * GENERATION_BATCH_SIZE
        batch_end = batch_start + GENERATION_BATCH_SIZE
        question_batch = all_questions[batch_start:batch_end]

        # Call the batched generation function
        generated_batch = generate_answers_batch(model, tokenizer, question_batch, MAX_NEW_TOKENS)
        all_generated_answers.extend(generated_batch) # Add the list of answers from the batch

    # Ensure generated answers list length matches questions list length
    if len(all_generated_answers) != len(all_questions):
         logger.error(f"Length mismatch after batch generation: questions={len(all_questions)}, generated={len(all_generated_answers)}. This indicates a bug.")
         # Pad with None to avoid crashing later, but investigate the cause
         all_generated_answers.extend([None] * (len(all_questions) - len(all_generated_answers)))

    # 3. Calculate BERTScore in batches (using the original function, it's already batched)
    logger.info(f"Calculating BERTScore for {len(all_questions)} pairs...")
    bert_score_device = device # Use the general device hint for BERTScore
    bert_scores = calculate_bert_score_batch(all_generated_answers, all_ground_truths, bert_score_device)

    if bert_scores is None:
        logger.error(f"BERTScore calculation failed for file {filename}. Skipping result saving.")
        # Still try to save results without scores if generation worked
        bert_scores = {'P': [], 'R': [], 'F1': []} # Provide empty lists

    # 4. Combine results
    score_idx = 0
    valid_generation_count = sum(1 for ans in all_generated_answers if ans is not None)
    if len(bert_scores.get('F1', [])) != valid_generation_count:
         logger.warning(f"BERT score count ({len(bert_scores.get('F1', []))}) does not match valid generated answer count ({valid_generation_count}). Alignment might be incorrect.")


    for i in range(len(all_questions)):
        question = all_questions[i]
        ground_truth = all_ground_truths[i]
        generated_answer = all_generated_answers[i] # Already contains None if generation failed

        result_item = {
            "model_name": model_config.name,
            "question": question,
            "ground_truth_answer": ground_truth,
            "generated_answer": generated_answer,
            "bert_score_P": None,
            "bert_score_R": None,
            "bert_score_F1": None,
        }

        # If the answer was generated successfully (not None and not empty), add its corresponding score
        # We need to carefully map scores back because bert_score only calculated for non-None items
        if generated_answer is not None and generated_answer.strip() != "":
            if score_idx < len(bert_scores.get('F1', [])):
                try:
                    result_item["bert_score_P"] = bert_scores['P'][score_idx]
                    result_item["bert_score_R"] = bert_scores['R'][score_idx]
                    result_item["bert_score_F1"] = bert_scores['F1'][score_idx]
                    score_idx += 1
                except IndexError:
                     logger.error(f"IndexError accessing BERT scores at index {score_idx}. BERT score list length: {len(bert_scores.get('F1', []))}. Assigning None.")
            else:
                 logger.warning(f"Score index {score_idx} out of bounds for BERT scores length {len(bert_scores.get('F1', []))}. Assigning None.")

        results.append(result_item)

    # Safety check: ensure number of results matches number of questions
    if len(results) != len(all_questions):
         logger.error(f"Final results count ({len(results)}) doesn't match question count ({len(all_questions)}). Data loss occurred.")

    # 5. Save results
    model_output_dir = os.path.join(output_dir, model_config.name)
    os.makedirs(model_output_dir, exist_ok=True)
    base_filename = os.path.splitext(filename)[0]
    output_filename = f"{base_filename}_{model_config.name}_results.json"
    output_filepath = os.path.join(model_output_dir, output_filename)

    logger.info(f"Saving results for {filename} to {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save results to {output_filepath}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting model evaluation script (Batched Generation, Quantization Disabled).")
    logger.warning("Bitsandbytes quantization is disabled. Ensure you have enough GPU memory.")
    logger.info(f"Generation Batch Size: {GENERATION_BATCH_SIZE}")

    # Set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all JSON files
    all_json_files = []
    for data_dir in DATA_DIRECTORIES:
        if not os.path.isdir(data_dir):
            logger.warning(f"Data directory not found: {data_dir}. Skipping.")
            continue
        try:
            for filename in os.listdir(data_dir):
                if filename.endswith(".json"):
                    all_json_files.append(os.path.join(data_dir, filename))
        except Exception as e:
            logger.error(f"Error listing files in directory {data_dir}: {e}")

    if not all_json_files:
        logger.error("No JSON files found. Exiting.")
        exit()
    logger.info(f"Found {len(all_json_files)} JSON files to process.")

    # Evaluate each model
    for model_config in MODEL_CONFIGS:
        model = None
        tokenizer = None
        try:
            start_time = time.time()
            model, tokenizer = load_model_and_tokenizer(model_config, device)
            load_time = time.time() - start_time

            if model is None or tokenizer is None:
                logger.error(f"Skipping evaluation for model {model_config.name} due to loading failure.")
                continue
            logger.info(f"Model {model_config.name} loaded in {load_time:.2f} seconds.")

            logger.info(f"--- Starting evaluation for model: {model_config.name} ---")
            for json_filepath in all_json_files:
                 process_file(model_config, model, tokenizer, json_filepath, OUTPUT_DIR, device)
            logger.info(f"--- Finished evaluation for model: {model_config.name} ---")

        except Exception as e:
            logger.error(f"An unexpected error occurred during evaluation of model {model_config.name}: {e}")
            logger.error(traceback.format_exc())
        finally:
            logger.info(f"Cleaning up resources for model: {model_config.name}")
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(f"Resources cleaned up for model: {model_config.name}")
            time.sleep(5)

    logger.info("Script finished.")