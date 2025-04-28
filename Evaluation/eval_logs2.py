# score_existing_logs_auto.py

# Standard library imports
import json
import logging
import os
import gc
import time
import traceback
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Callable, Any
import math

# Third-party imports
import torch
from tqdm import tqdm
from bert_score import score as bert_score_calculate
from sentence_transformers import SentenceTransformer, util

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Reduce verbosity from underlying libraries
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING) # Corrected logger name

# --- Configuration ---
# Base directory containing the model-specific subdirectories (e.g., "HERE/gemma")
BASE_GENERATED_DIR = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generated_Answers/HERE/qwen"

# Base directory where scored results will be saved.
# Model-specific subdirectories will be created under this path.
BASE_SCORED_OUTPUT_DIR = f"{BASE_GENERATED_DIR}_scored" # Append _scored to the base generated dir

# --- Scoring Configuration (Unchanged) ---
BERT_SCORE_MODEL = "bert-base-multilingual-cased"
ST_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

DEFAULT_LANG_FOR_BERT_SCORE = "en"

_st_model_cache: Optional[SentenceTransformer] = None

# --- Scoring Functions (calculate_bert_scores, calculate_st_cosine_similarity - unchanged) ---

def calculate_bert_scores(
    predictions: List[Optional[str]], references: List[Optional[str]], device: torch.device,
    model_type: str, batch_size: int = 16, lang: Optional[str] = DEFAULT_LANG_FOR_BERT_SCORE
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Calculates BERTScore (P, R, F1) for valid pairs."""
    # (Implementation is identical to the previous version)
    if not predictions or not references or len(predictions) != len(references):
        logger.error(f"Invalid input lists for BERTScore: preds({len(predictions)}), refs({len(references)})")
        return [], {'P': [], 'R': [], 'F1': []}
    valid_pairs = []
    original_indices = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if pred is not None and isinstance(pred, str) and pred.strip() and \
           ref is not None and isinstance(ref, str) and ref.strip():
            valid_pairs.append((pred, ref))
            original_indices.append(i)
    if not valid_pairs:
        logger.warning("No valid prediction/reference pairs found for BERTScore.")
        return [], {'P': [], 'R': [], 'F1': []}
    filtered_predictions = [p for p, r in valid_pairs]
    filtered_references = [r for p, r in valid_pairs]
    logger.debug(f"Calculating BERTScore ({model_type}) for {len(filtered_predictions)} valid pairs...")
    bert_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        P, R, F1 = bert_score_calculate(
            filtered_predictions, filtered_references, model_type=model_type,
            lang=lang, verbose=False, device=bert_device, batch_size=batch_size
        )
        logger.debug("BERTScore calculation finished.")
        return original_indices, {'P': P.tolist(), 'R': R.tolist(), 'F1': F1.tolist()}
    except Exception as e:
        logger.error(f"Error calculating BERTScore: {e}", exc_info=True)
        return [], {'P': [], 'R': [], 'F1': []}

def calculate_st_cosine_similarity(
    predictions: List[Optional[str]], references: List[Optional[str]], device: torch.device,
    model_name: str
) -> Tuple[List[int], Dict[str, List[float]]]:
    """Calculates Sentence Transformer cosine similarity for valid pairs."""
    # (Implementation is identical to the previous version, including caching)
    global _st_model_cache
    if not predictions or not references or len(predictions) != len(references):
        logger.error(f"Invalid input lists for ST Similarity: preds({len(predictions)}), refs({len(references)})")
        return [], {'cosine_similarity': []}
    valid_pairs = []
    original_indices = []
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        if pred is not None and isinstance(pred, str) and pred.strip() and \
           ref is not None and isinstance(ref, str) and ref.strip():
            valid_pairs.append((pred, ref))
            original_indices.append(i)
    if not valid_pairs:
        logger.warning("No valid prediction/reference pairs found for ST Similarity.")
        return [], {'cosine_similarity': []}
    filtered_predictions = [p for p, r in valid_pairs]
    filtered_references = [r for p, r in valid_pairs]
    logger.debug(f"Calculating ST Similarity ({model_name}) for {len(filtered_predictions)} valid pairs...")
    try:
        if _st_model_cache is None or str(_st_model_cache.device) != str(device):
            logger.info(f"Loading Sentence Transformer model: {model_name} onto device: {device}")
            _st_model_cache = SentenceTransformer(model_name, device=device)
            logger.info("Sentence Transformer model loaded.")
        else: logger.debug("Using cached Sentence Transformer model.")
        st_model = _st_model_cache
        embeddings_pred = st_model.encode(filtered_predictions, convert_to_tensor=True, show_progress_bar=False)
        embeddings_ref = st_model.encode(filtered_references, convert_to_tensor=True, show_progress_bar=False)
        cosine_matrix = util.cos_sim(embeddings_pred, embeddings_ref)
        similarity_scores = torch.diag(cosine_matrix).tolist()
        logger.debug("Sentence Transformer Similarity calculation finished.")
        return original_indices, {'cosine_similarity': similarity_scores}
    except Exception as e:
        logger.error(f"Error calculating Sentence Transformer Similarity: {e}", exc_info=True)
        return [], {'cosine_similarity': []}

# --- Combined Score Calculation Function (Unchanged) ---
def calculate_scores(
    predictions: List[Optional[str]], references: List[Optional[str]],
    scoring_functions: Dict[str, Callable], device: torch.device
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, float]]:
    """Calculates multiple scores using the provided scoring functions."""
    # (Implementation is identical to the previous version)
    detailed_scores: Dict[int, Dict[str, float]] = {}
    aggregate_totals: Dict[str, float] = {}
    valid_counts: Dict[str, int] = {}
    for prefix, func in scoring_functions.items():
        logger.debug(f"Calculating scores for metric: {prefix}")
        try:
            original_indices, current_scores = func(predictions, references, device)
            if not original_indices or not current_scores:
                logger.warning(f"No scores returned for metric: {prefix}")
                continue
            for score_name, score_list in current_scores.items():
                full_score_name = f"{prefix}_{score_name}"
                aggregate_totals[full_score_name] = 0.0
                valid_counts[full_score_name] = 0
                if len(original_indices) != len(score_list):
                    logger.error(f"Index/Score list length mismatch for {full_score_name}. Skipping.")
                    continue
                for i, score_value in enumerate(score_list):
                    original_idx = original_indices[i]
                    if original_idx not in detailed_scores: detailed_scores[original_idx] = {}
                    if isinstance(score_value, (float, int)):
                        detailed_scores[original_idx][full_score_name] = float(score_value)
                        aggregate_totals[full_score_name] += float(score_value)
                        valid_counts[full_score_name] += 1
                    else: detailed_scores[original_idx][full_score_name] = None
        except Exception as e:
            logger.error(f"Failed to calculate scores for metric {prefix}: {e}", exc_info=True)
    average_scores = {}
    for score_name, total in aggregate_totals.items():
        count = valid_counts.get(score_name, 0)
        avg_score_name = f"average_{score_name}"
        average_scores[avg_score_name] = total / count if count > 0 else None
    return detailed_scores, average_scores

# --- Function to process a single file (Updated for clarity and consistency) ---
def score_and_update_file(input_filepath: str, output_filepath: str, scoring_functions: Dict[str, Callable], device: torch.device):
    """
    Loads a JSON file, calculates scores, adds them to the data, and saves to the specified output file.
    """
    logger.info(f"Processing: {os.path.basename(input_filepath)}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read/parse JSON file {input_filepath}: {e}", exc_info=True)
        return False # Indicate failure

    # --- Data Validation ---
    if "details" not in data or not isinstance(data["details"], list):
        logger.error(f"Invalid format: 'details' list not found in {input_filepath}. Skipping.")
        return False # Indicate failure
    if not data["details"]:
         logger.warning(f"'details' list is empty in {input_filepath}. Nothing to score. Skipping file.")
         return False # Indicate failure

    details = data["details"]
    summary = data.get("summary", {})
    data["summary"] = summary # Ensure summary is part of the data dict

    # Extract predictions and references for scoring
    all_predictions = [item.get("generated_answer") for item in details]
    all_references = [item.get("ground_truth_answer") for item in details]

    # --- Score Calculation ---
    # Define the scoring functions dictionary here or pass it
    defined_scoring_functions = {
        "bert_score": lambda preds, refs, dev: calculate_bert_scores(
            preds, refs, dev, model_type=BERT_SCORE_MODEL, batch_size=16, lang=DEFAULT_LANG_FOR_BERT_SCORE
        ),
        "st_similarity": lambda preds, refs, dev: calculate_st_cosine_similarity(
            preds, refs, dev, model_name=ST_MODEL_NAME
        ),
    }
    detailed_scores, average_scores = calculate_scores(
        all_predictions, all_references, defined_scoring_functions, device
    )

    # --- Update Data Structure ---
    valid_indices_scored = set(detailed_scores.keys())
    logger.debug(f"Adding scores to {len(valid_indices_scored)} items in 'details'.")
    for i, item in enumerate(details):
        if i in detailed_scores:
            item.update(detailed_scores[i])

    logger.debug("Updating 'summary' with average scores.")
    summary["valid_pairs_for_scoring"] = len(valid_indices_scored)
    summary.update(average_scores)

    # --- Save Updated Data to New File ---
    logger.info(f"Saving scored results to: {output_filepath}")
    try:
        # Ensure output directory exists just before writing
        os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
        with open(output_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True # Indicate success
    except Exception as e:
        logger.error(f"Failed to save scored results to {output_filepath}: {e}", exc_info=True)
        return False # Indicate failure

# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting automated scoring script for existing log files.")

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU.")

    # --- Input/Output Validation ---
    if not os.path.isdir(BASE_GENERATED_DIR):
        logger.error(f"Base generated directory not found: {BASE_GENERATED_DIR}")
        exit(1)
    # Base output directory will be created if needed

    # --- Find Model Subdirectories ---
    model_subdirs = []
    try:
        entries = os.listdir(BASE_GENERATED_DIR)
        for entry in entries:
            full_path = os.path.join(BASE_GENERATED_DIR, entry)
            if os.path.isdir(full_path):
                model_subdirs.append(entry) # Store only the subdirectory name
    except Exception as e:
        logger.error(f"Error listing model directories in {BASE_GENERATED_DIR}: {e}")
        exit(1)

    if not model_subdirs:
        logger.error(f"No model subdirectories found in {BASE_GENERATED_DIR}. Exiting.")
        exit(1)

    logger.info(f"Found {len(model_subdirs)} potential model subdirectories: {model_subdirs}")
    logger.info(f"Scored files will be saved under base directory: {BASE_SCORED_OUTPUT_DIR}")

    # --- Process Each Model Directory ---
    overall_start_time = time.time()
    total_files_processed = 0
    total_files_failed = 0

    # Use tqdm for iterating through model directories
    for model_dir_name in tqdm(model_subdirs, desc="Processing Model Dirs"):
        model_input_dir = os.path.join(BASE_GENERATED_DIR, model_dir_name)
        model_output_dir = os.path.join(BASE_SCORED_OUTPUT_DIR, model_dir_name) # Create corresponding output dir

        logger.info(f"\n--- Processing Model Directory: {model_dir_name} ---")
        logger.info(f"Input Path: {model_input_dir}")
        logger.info(f"Output Path: {model_output_dir}")

        # --- Find JSON files within this model directory ---
        files_to_score_in_model_dir = []
        try:
            for filename in os.listdir(model_input_dir):
                if filename.endswith(".json"):
                    files_to_score_in_model_dir.append(os.path.join(model_input_dir, filename))
        except Exception as e:
            logger.error(f"Error listing files in directory {model_input_dir}: {e}. Skipping directory.")
            continue # Skip to the next model directory

        if not files_to_score_in_model_dir:
            logger.warning(f"No '.json' files found in {model_input_dir}. Skipping directory.")
            continue

        logger.info(f"Found {len(files_to_score_in_model_dir)} '.json' files for model '{model_dir_name}'.")

        # --- Process each file within the model directory ---
        # Use tqdm for iterating through files within a model directory
        for json_filepath in tqdm(files_to_score_in_model_dir, desc=f"Scoring Files in {model_dir_name}", leave=False):
            # Construct the specific output path for this file
            base_filename = os.path.basename(json_filepath)
            # Modify filename for output (e.g., replace _generated with _scored)
            if base_filename.endswith("_generated.json"): # Assuming generated files have this suffix
                 output_base_filename = base_filename.replace("_generated.json", "_scored.json")
            else: # Fallback if suffix is different or missing
                 base, ext = os.path.splitext(base_filename)
                 output_base_filename = f"{base}_scored{ext}"

            output_filepath = os.path.join(model_output_dir, output_base_filename)

            success = score_and_update_file(
                json_filepath,
                output_filepath, # Pass the specific output path
                {}, # scoring_functions are defined inside the function now
                device
            )
            if success:
                 total_files_processed += 1
            else:
                 total_files_failed += 1

            # --- Memory Management within the inner loop ---
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logger.info(f"--- Finished processing for Model Directory: {model_dir_name} ---")
        # Optional: Add a small delay between model directories if needed
        # time.sleep(2)

    # --- Final Cleanup ---
    _st_model_cache = None # Clear ST model cache
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall_end_time = time.time()
    logger.info(f"\n{'='*50}")
    logger.info("Scoring script finished.")
    logger.info(f"Total time: {overall_end_time - overall_start_time:.2f} seconds.")
    logger.info(f"Total files successfully scored and saved: {total_files_processed}")
    logger.info(f"Total files failed or skipped: {total_files_failed}")
    logger.info(f"Scored results saved under: {BASE_SCORED_OUTPUT_DIR}")
    logger.info(f"{'='*50}")