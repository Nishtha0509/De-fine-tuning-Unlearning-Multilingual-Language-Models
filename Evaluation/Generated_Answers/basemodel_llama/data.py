import os
import json
import logging
import sys

# --- Configuration ---
# Set the target directory where your JSON files are located
TARGET_DIR = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generated_Answers/basemodel_llama/llama3_2_unlearn_with_taskvectors"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Main Processing Function ---
def clean_generated_answers(directory_path: str):
    """
    Iterates through all .json files in the given directory,
    reads them, removes the 'question' prefix from 'generated_answer'
    if present, and overwrites the file with the cleaned data.
    """
    if not os.path.isdir(directory_path):
        logger.error(f"Error: Directory not found: {directory_path}")
        sys.exit(1) # Exit if the target directory doesn't exist

    logger.info(f"Starting processing for directory: {directory_path}")
    files_processed = 0
    files_modified = 0
    errors_encountered = 0

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            logger.debug(f"Processing file: {file_path}")
            files_processed += 1
            data = None # Initialize data to None
            made_modification = False # Flag to track if this file was changed

            # --- Read the JSON file ---
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to decode JSON from file: {file_path}. Skipping.")
                errors_encountered += 1
                continue
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}. Skipping.")
                errors_encountered += 1
                continue

            # --- Process the 'details' section ---
            if data and "details" in data and isinstance(data["details"], list):
                cleaned_details = [] # Create a new list for potentially modified items
                for item in data["details"]:
                    # Check if item is a dict and has the required keys
                    if isinstance(item, dict) and "question" in item and "generated_answer" in item:
                        question = item.get("question")
                        generated_answer = item.get("generated_answer")

                        # Ensure both are strings before proceeding
                        if isinstance(question, str) and isinstance(generated_answer, str):
                            # Check if generated_answer starts with the question text
                            if generated_answer.strip().startswith(question.strip()): # Use strip() for robustness
                                q_len = len(question.strip())
                                # Remove the question prefix and strip leading/trailing whitespace
                                cleaned_answer = generated_answer.strip()[q_len:].strip()

                                # Check if the answer actually changed (prevent empty updates)
                                if cleaned_answer != generated_answer:
                                     logger.debug(f"    Cleaned answer in file: {filename}")
                                     item["generated_answer"] = cleaned_answer # Update the item in-place
                                     made_modification = True

                            # else: # Optional: log if no modification needed for an item
                            #     logger.debug(f"    Answer in {filename} did not start with question.")

                        else:
                            logger.warning(f"    Skipping item in {filename} due to non-string type for question or generated_answer.")
                    else:
                         logger.warning(f"    Skipping item in {filename} due to missing keys or incorrect item format.")
                    # Add the (potentially modified) item to the new list - Not necessary if modifying in-place
                    # cleaned_details.append(item) # Only needed if not modifying 'item' directly

                # If we didn't modify in-place, update the data:
                # if made_modification:
                #     data["details"] = cleaned_details

            else:
                logger.warning(f"'details' key missing, not a list, or data is empty in {file_path}. Skipping detail processing.")

            # --- Write the modified data back (only if changes were made) ---
            if made_modification:
                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        # Use indent for pretty printing, ensure_ascii=False for non-ASCII chars
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    logger.info(f"Successfully cleaned and overwrote: {file_path}")
                    files_modified += 1
                except Exception as e:
                    logger.error(f"Error writing cleaned data to {file_path}: {e}")
                    errors_encountered += 1
            # else: # Optional: Log files that didn't need modification
            #     logger.debug(f"No modifications needed for: {file_path}")


    logger.info("--- Processing Summary ---")
    logger.info(f"Total files found (.json): {files_processed}")
    logger.info(f"Files modified: {files_modified}")
    logger.info(f"Errors encountered: {errors_encountered}")
    logger.info("Processing complete.")

# --- Run the script ---
if __name__ == "__main__":
    clean_generated_answers(TARGET_DIR)