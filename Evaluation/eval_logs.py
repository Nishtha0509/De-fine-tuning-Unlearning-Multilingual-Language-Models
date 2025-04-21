# score_existing_logs.py

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
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("SentenceTransformer").setLevel(logging.WARNING)

GENERATED = "HERE"
MODEL = "unlearned_model_Qwen2.5-7B-Instruct_ENG"
# --- Configuration ---
# Directory containing JSON files to be scored (e.g., output from generation script)
# This path should contain model subdirectories with .json files inside.
INPUT_DATA_DIR = f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generated_Answers/{GENERATED}/{MODEL}"
# Directory to save the JSON files *with scores added*
SCORED_OUTPUT_DIR = f"/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generated_Answers/{GENERATED}/{MODEL}_scored" # Changed suffix

# --- Scoring Configuration ---
BERT_SCORE_MODEL = "bert-base-multilingual-cased"
ST_MODEL_NAME = 'paraphrase-multilingual-mpnet-base-v2'

SCORING_FUNCTIONS = {
    "bert_score": lambda preds, refs, device: calculate_bert_scores(
        preds, refs, device, model_type=BERT_SCORE_MODEL, batch_size=16
    ),
    "st_similarity": lambda preds, refs, device: calculate_st_cosine_similarity(
        preds, refs, device, model_name=ST_MODEL_NAME
    ),
}

DEFAULT_LANG_FOR_BERT_SCORE = "en"

_st_model_cache: Optional[SentenceTransformer] = None

# --- Scoring Functions (calculate_bert_scores, calculate_st_cosine_similarity - unchanged from previous) ---

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
    logger.info(f"Calculating BERTScore ({model_type}) for {len(filtered_predictions)} valid pairs...")
    bert_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        P, R, F1 = bert_score_calculate(
            filtered_predictions, filtered_references, model_type=model_type,
            lang=lang, verbose=False, device=bert_device, batch_size=batch_size
        )
        logger.info("BERTScore calculation finished.")
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
    logger.info(f"Calculating Sentence Transformer Similarity ({model_name}) for {len(filtered_predictions)} valid pairs...")
    try:
        if _st_model_cache is None or str(_st_model_cache.device) != str(device):
            logger.info(f"Loading Sentence Transformer model: {model_name} onto device: {device}")
            _st_model_cache = SentenceTransformer(model_name, device=device)
            logger.info("Sentence Transformer model loaded.")
        else: logger.info("Using cached Sentence Transformer model.")
        st_model = _st_model_cache
        embeddings_pred = st_model.encode(filtered_predictions, convert_to_tensor=True, show_progress_bar=False)
        embeddings_ref = st_model.encode(filtered_references, convert_to_tensor=True, show_progress_bar=False)
        cosine_matrix = util.cos_sim(embeddings_pred, embeddings_ref)
        similarity_scores = torch.diag(cosine_matrix).tolist()
        logger.info("Sentence Transformer Similarity calculation finished.")
        return original_indices, {'cosine_similarity': similarity_scores}
    except Exception as e:
        logger.error(f"Error calculating Sentence Transformer Similarity: {e}", exc_info=True)
        return [], {'cosine_similarity': []}


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
        logger.info(f"--- Calculating scores for metric: {prefix} ---")
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


# --- Function to process a single file: Read -> Score -> Update -> Write New ---
def score_and_update_file(input_filepath: str, input_base_dir: str, output_base_dir: str, scoring_functions: Dict[str, Callable], device: torch.device):
    """
    Loads a JSON file, calculates scores, adds them to the data, and saves to a new file.
    """
    relative_path = os.path.relpath(input_filepath, input_base_dir)
    output_filepath = os.path.join(output_base_dir, relative_path)
    # Ensure output directory structure exists
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    # Change output filename suffix (optional, but clearer)
    base, ext = os.path.splitext(output_filepath)
    if base.endswith("_generated_retry"): # Handle suffix from previous step
        base = base[:-len("_generated_retry")]
    output_filepath = f"{base}_scored.json" # Add new suffix


    logger.info(f"Processing: {input_filepath}")
    logger.info(f"Outputting to: {output_filepath}")

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            # IMPORTANT: Load the *entire* existing data structure
            data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read or parse JSON file {input_filepath}: {e}", exc_info=True)
        return

    # --- Data Validation ---
    if "details" not in data or not isinstance(data["details"], list):
        logger.error(f"Invalid format: 'details' list not found in {input_filepath}. Skipping.")
        return
    if not data["details"]:
         logger.warning(f"'details' list is empty in {input_filepath}. Nothing to score.")
         # Optionally save the original file to the output dir anyway? Or just skip? Let's skip.
         return

    details = data["details"]
    # Get summary, create if it doesn't exist
    summary = data.get("summary", {})
    data["summary"] = summary # Ensure summary is part of the data dict

    # Extract predictions and references for scoring
    all_predictions = [item.get("generated_answer") for item in details]
    all_references = [item.get("ground_truth_answer") for item in details]

    # --- Score Calculation ---
    detailed_scores, average_scores = calculate_scores(
        all_predictions, all_references, scoring_functions, device
    )

    # --- Update Data Structure ---
    # Add detailed scores to each item in the 'details' list
    valid_indices_scored = set(detailed_scores.keys())
    logger.info(f"Adding scores to {len(valid_indices_scored)} items in 'details'.")
    for i, item in enumerate(details):
        if i in detailed_scores:
            # Update the item dictionary with the calculated scores for this index
            item.update(detailed_scores[i])
        # You could add placeholder None values for score fields if an item wasn't scored,
        # but it might make the file larger. Let's only add scores that were calculated.

    # Update the 'summary' dictionary with average scores and counts
    logger.info("Updating 'summary' with average scores.")
    summary["valid_pairs_for_scoring"] = len(valid_indices_scored) # Number of pairs actually scored
    summary.update(average_scores) # Add average scores (e.g., "average_bert_score_F1": 0.85)

    # --- Save Updated Data to New File ---
    logger.info(f"Saving scored results to: {output_filepath}")
    try:
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Save the *entire modified data structure*
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save scored results to {output_filepath}: {e}", exc_info=True)


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Starting scoring script for existing log files.")

    # --- Device Setup ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"CUDA available. Using GPU for potential scoring: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("CUDA not available. Using CPU for potential scoring.")

    # --- Input/Output Validation ---
    if not os.path.isdir(INPUT_DATA_DIR):
        logger.error(f"Input data directory not found: {INPUT_DATA_DIR}")
        exit(1)
    # Output directory will be created if it doesn't exist by os.makedirs later

    # --- Find Files to Process ---
    files_to_score = []
    logger.info(f"Searching for .json files recursively in: {INPUT_DATA_DIR}")
    for root, _, files in os.walk(INPUT_DATA_DIR):
        for filename in files:
            if filename.endswith(".json"): # Find all json files
                files_to_score.append(os.path.join(root, filename))

    if not files_to_score:
        logger.error(f"No '.json' files found recursively in {INPUT_DATA_DIR}. Exiting.")
        exit()

    logger.info(f"Found {len(files_to_score)} '.json' files to score.")
    logger.info(f"Configured metrics: {list(SCORING_FUNCTIONS.keys())}")
    logger.info(f"Scored files will be saved under: {SCORED_OUTPUT_DIR}")


    # --- Process Each File ---
    overall_start_time = time.time()
    for json_filepath in tqdm(files_to_score, desc="Scoring Files"):
        score_and_update_file(
            json_filepath,
            INPUT_DATA_DIR,      # Pass base input dir for relative path calculation
            SCORED_OUTPUT_DIR,   # Pass base output dir
            SCORING_FUNCTIONS,
            device
        )
        # --- Memory Management ---
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- Cleanup ---
    _st_model_cache = None # Clear ST model cache
    gc.collect()
    if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_end_time = time.time()
    logger.info(f"Scoring script finished in {overall_end_time - overall_start_time:.2f} seconds.")