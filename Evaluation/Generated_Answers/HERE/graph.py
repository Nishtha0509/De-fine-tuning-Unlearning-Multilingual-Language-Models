import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob # To find files matching a pattern
import re # For extracting info from names

# --- Configuration ---
# Base directory containing gemma_scored, llama_scored, qwen_scored folders
BASE_EVAL_DIR = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/Evaluation/Generated_Answers/HERE"
# Directory to save the generated plots
OUTPUT_PLOT_DIR = "./evaluation_score_plots"
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# List of base models to process (derived from folder names like gemma_scored)
BASE_MODEL_NAMES = ["gemma", "llama", "qwen"]

# --- 1. Data Loading and Preparation ---
all_score_data = []

print("Starting data loading process...")

# Iterate through each base model type
for base_model_name in BASE_MODEL_NAMES:
    scored_dir = os.path.join(BASE_EVAL_DIR, f"{base_model_name}_scored")
    if not os.path.isdir(scored_dir):
        print(f"Warning: Directory not found - {scored_dir}. Skipping {base_model_name}.")
        continue

    print(f"Processing directory: {scored_dir}")

    # Find all *_scored.json files within the unlearned model subdirectories
    # Pattern: .../<base_model_name>_scored/unlearned_..._seq_.../*_scored.json
    json_files_pattern = os.path.join(scored_dir, "unlearned_*", "*_scored.json")
    json_files = glob.glob(json_files_pattern)

    if not json_files:
        print(f"  No '*_scored.json' files found in subdirectories of {scored_dir}")
        continue

    print(f"  Found {len(json_files)} score files for {base_model_name}.")

    for f_path in json_files:
        try:
            # Extract info from file path and name
            parts = f_path.split(os.sep)
            filename = parts[-1] # e.g., eng_forget01_scored.json
            unlearned_model_dir_name = parts[-2] # e.g., unlearned_gemma-3-4B-it_ENG_seq_en

            # Extract dataset name (e.g., eng_forget01, eng_retain_half)
            dataset_scored = filename.replace("_scored.json", "")

            # Extract unlearning sequence from directory name (requires consistent naming)
            match = re.search(r'_seq_([a-zA-Z0-9_]+)$', unlearned_model_dir_name)
            if match:
                unlearning_sequence = match.group(1)
            else:
                # Fallback or attempt extraction from JSON if needed later
                unlearning_sequence = "unknown_sequence"
                print(f"  Warning: Could not extract sequence from dir name: {unlearned_model_dir_name}")

            # Load JSON data
            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract scores from summary
            summary = data.get("summary", {})
            bert_f1 = summary.get("average_bert_score_F1")
            cosine_sim = summary.get("average_st_similarity_cosine_similarity")
            # Optionally get the model name from JSON for verification/more detail
            json_model_name = summary.get("model_name", unlearned_model_dir_name)

            if bert_f1 is not None and cosine_sim is not None:
                row = {
                    "base_model": base_model_name, # gemma, llama, or qwen
                    "unlearning_sequence": unlearning_sequence,
                    "dataset_scored": dataset_scored,
                    "bert_f1": bert_f1,
                    "cosine_similarity": cosine_sim,
                    "full_model_name": json_model_name # Keep the detailed name from JSON
                }
                all_score_data.append(row)
            else:
                print(f"  Warning: Missing scores in summary for {f_path}. Skipping.")

        except json.JSONDecodeError:
            print(f"  Error: Failed to decode JSON from {f_path}. Skipping.")
        except Exception as e:
            print(f"  Error processing file {f_path}: {e}")

# Create DataFrame
if not all_score_data:
    print("No score data loaded. Cannot create plots.")
    exit()

df_scores = pd.DataFrame(all_score_data)

# Convert score columns to numeric
df_scores['bert_f1'] = pd.to_numeric(df_scores['bert_f1'], errors='coerce')
df_scores['cosine_similarity'] = pd.to_numeric(df_scores['cosine_similarity'], errors='coerce')

# Drop rows where conversion failed
df_scores.dropna(subset=['bert_f1', 'cosine_similarity'], inplace=True)

print("\nScore data loaded and processed into DataFrame:")
print(df_scores.head())
print(f"\nDataFrame shape: {df_scores.shape}")
print(f"Base models found: {df_scores['base_model'].unique()}")
print(f"Datasets scored found: {df_scores['dataset_scored'].unique()}")
print(f"Unique sequences found: {df_scores['unlearning_sequence'].unique()}")


# --- 2. Plotting Loop by Base Model and Scored Dataset ---

# Get unique datasets that were scored across all models
unique_datasets_scored = df_scores['dataset_scored'].unique()

for base_model_name in BASE_MODEL_NAMES:
    df_model_subset = df_scores[df_scores['base_model'] == base_model_name]

    if df_model_subset.empty:
        print(f"\nNo score data for model {base_model_name}. Skipping plots.")
        continue

    print(f"\n--- Generating score plots for base model: {base_model_name} ---")

    # Create a subdirectory for this model's score plots
    model_plot_dir = os.path.join(OUTPUT_PLOT_DIR, base_model_name)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Generate plots for each scored dataset
    for dataset_name in unique_datasets_scored:
        df_plot = df_model_subset[df_model_subset['dataset_scored'] == dataset_name].sort_values('unlearning_sequence')

        if df_plot.empty:
            # print(f"  No data for dataset '{dataset_name}' for model {base_model_name}. Skipping plot.")
            continue

        print(f"  Generating plots for dataset: {dataset_name}")

        # Plot BERTScore F1
        plt.figure(figsize=(15, 8)) # Wider figure for potentially many sequences
        sns.barplot(data=df_plot, x='unlearning_sequence', y='bert_f1', palette='viridis')
        plt.xlabel("Unlearning Sequence")
        plt.ylabel("Average BERTScore F1")
        plt.title(f"BERTScore F1 Comparison for {base_model_name.capitalize()} Model\n(Dataset: {dataset_name})")
        plt.xticks(rotation=60, ha='right') # Rotate labels for readability
        plt.ylim(bottom=min(0.8, df_plot['bert_f1'].min() * 0.95) if not df_plot.empty else 0) # Adjust y-axis start dynamically or set minimum
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plot_filename_bert = f"bert_f1_{dataset_name}_comparison.png"
        plt.savefig(os.path.join(model_plot_dir, plot_filename_bert))
        plt.close()

        # Plot Cosine Similarity
        plt.figure(figsize=(15, 8))
        sns.barplot(data=df_plot, x='unlearning_sequence', y='cosine_similarity', palette='magma')
        plt.xlabel("Unlearning Sequence")
        plt.ylabel("Average Cosine Similarity (SentenceTransformer)")
        plt.title(f"Cosine Similarity Comparison for {base_model_name.capitalize()} Model\n(Dataset: {dataset_name})")
        plt.xticks(rotation=60, ha='right')
        plt.ylim(bottom=min(0.8, df_plot['cosine_similarity'].min() * 0.95) if not df_plot.empty else 0) # Adjust y-axis start dynamically or set minimum
        plt.grid(axis='y', linestyle='--')
        plt.tight_layout()
        plot_filename_cosine = f"cosine_similarity_{dataset_name}_comparison.png"
        plt.savefig(os.path.join(model_plot_dir, plot_filename_cosine))
        plt.close()

print(f"\nAll score plots saved to subdirectories within: {OUTPUT_PLOT_DIR}")