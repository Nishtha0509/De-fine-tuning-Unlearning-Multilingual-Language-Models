import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob # To find files matching a pattern

# --- Configuration ---
RESULTS_BASE_DIR = "./unlearned_models_sequences_fixedLR_with_combined"
OUTPUT_PLOT_DIR = "./unlearning_plots_fixedLR_by_model" # Changed output directory name
os.makedirs(OUTPUT_PLOT_DIR, exist_ok=True)

# --- 1. Data Loading and Preparation (이전과 동일) ---
all_results_data = []
json_files_pattern = os.path.join(RESULTS_BASE_DIR, "**", "unlearning_results_sequence_*.json")
json_files = glob.glob(json_files_pattern, recursive=True)

if not json_files:
    print(f"Error: No result JSON files found recursively under {RESULTS_BASE_DIR}")
    print(f"Searched pattern: {json_files_pattern}")
else:
    print(f"Found {len(json_files)} result files. Loading...")
    # ... (이전과 동일한 데이터 로딩 및 파싱 로직) ...
    for f_path in json_files:
        try:
            parts = f_path.split(os.sep)
            filename = parts[-1]
            sequence_dir_name = parts[-2]
            inferred_sequence_str = filename.replace("unlearning_results_sequence_", "").replace(".json", "")
            if sequence_dir_name.startswith("unlearned_"):
                 temp_name = sequence_dir_name.replace("unlearned_", "")
                 last_seq_index = temp_name.rfind("_seq_")
                 if last_seq_index != -1: inferred_model_name = temp_name[:last_seq_index]
                 else: inferred_model_name = "UnknownModel"
            else: inferred_model_name = "UnknownModel"

            with open(f_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                model_name = data.get("model_name", inferred_model_name)
                sequence = tuple(data.get("unlearning_sequence", inferred_sequence_str.split('_')))
                sequence_str = "_".join(sequence)
                initial_retain_losses = data.get("initial_retain_losses", {})

                if not model_name or not sequence: continue

                for step_info in data.get("language_steps", []):
                    step_id = step_info.get("language_step_id")
                    step_num = step_info.get("step_number")
                    row = {
                        "model": model_name, "sequence": sequence_str, "sequence_tuple": sequence,
                        "step_id": step_id, "step_num": step_num,
                        "forget_loss_before": step_info.get("forget_loss_before_ga"),
                        "forget_loss_after": step_info.get("forget_loss_after_ga"),
                        "initial_retain_en": initial_retain_losses.get("en"),
                        "initial_retain_ko": initial_retain_losses.get("ko"),
                        "initial_retain_hi": initial_retain_losses.get("hi"),
                    }
                    retain_after = step_info.get("retain_losses_after_ga", {})
                    row["retain_en_after"] = retain_after.get("en")
                    row["retain_ko_after"] = retain_after.get("ko")
                    row["retain_hi_after"] = retain_after.get("hi")
                    all_results_data.append(row)
        except Exception as e:
            print(f"Error processing file {f_path}: {e}")

    # Create DataFrame
    if not all_results_data:
        print("No data loaded. Cannot create plots.")
        exit()

    df = pd.DataFrame(all_results_data)
    numeric_cols = [
        'step_num', 'forget_loss_before', 'forget_loss_after',
        'initial_retain_en', 'initial_retain_ko', 'initial_retain_hi',
        'retain_en_after', 'retain_ko_after', 'retain_hi_after'
    ]
    for col in numeric_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
    df['forget_loss_increase'] = df['forget_loss_after'] - df['forget_loss_before']
    df['retain_en_change'] = df['retain_en_after'] - df['initial_retain_en']
    df['retain_ko_change'] = df['retain_ko_after'] - df['initial_retain_ko']
    df['retain_hi_change'] = df['retain_hi_after'] - df['initial_retain_hi']

    print("Data loaded and processed into DataFrame.")
    print(f"DataFrame shape: {df.shape}")
    unique_models = df['model'].unique()
    print(f"Models found: {unique_models}")


# --- 2. Plotting Loop by Model ---

for model_name in unique_models:
    print(f"\n--- Generating plots for model: {model_name} ---")

    # Filter DataFrame for the current model
    df_model = df[df['model'] == model_name].copy() # Use .copy() to avoid SettingWithCopyWarning

    if df_model.empty:
        print(f"No data found for model {model_name}. Skipping plots.")
        continue

    # Create a subdirectory for this model's plots
    model_plot_dir = os.path.join(OUTPUT_PLOT_DIR, model_name)
    os.makedirs(model_plot_dir, exist_ok=True)

    # Get unique sequences for this model
    sequences_for_model = df_model['sequence'].unique()

    # --- Plotting Example 1 & 2: Trend plots per sequence for this model ---
    for sequence_str in sequences_for_model:
        df_seq_subset = df_model[df_model['sequence'] == sequence_str].sort_values('step_num')

        if not df_seq_subset.empty:
            # Plot Forget Loss Trend
            plt.figure(figsize=(10, 6))
            plt.plot(df_seq_subset['step_num'], df_seq_subset['forget_loss_before'], marker='o', linestyle='--', label='Forget Loss Before GA')
            plt.plot(df_seq_subset['step_num'], df_seq_subset['forget_loss_after'], marker='o', linestyle='-', label='Forget Loss After GA')
            plt.xticks(ticks=df_seq_subset['step_num'], labels=df_seq_subset['step_id'])
            plt.xlabel("Unlearning Step (Language/ID)")
            plt.ylabel("Forget Loss")
            # **** Title and Filename updated ****
            plt.title(f"Forget Loss Trend for {model_name}\n(Sequence: {sequence_str})")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plot_filename = f"forget_loss_trend_seq_{sequence_str}.png"
            plt.savefig(os.path.join(model_plot_dir, plot_filename))
            plt.close()
            # print(f"  Saved: {plot_filename}") # Can be verbose

            # Plot Retain Loss Trend
            plt.figure(figsize=(10, 6))
            plt.plot(df_seq_subset['step_num'], df_seq_subset['retain_en_after'], marker='^', linestyle='-', label='Retain Loss (EN)')
            plt.plot(df_seq_subset['step_num'], df_seq_subset['retain_ko_after'], marker='s', linestyle='-', label='Retain Loss (KO)')
            plt.plot(df_seq_subset['step_num'], df_seq_subset['retain_hi_after'], marker='d', linestyle='-', label='Retain Loss (HI)')
            if 'initial_retain_en' in df_seq_subset.columns and pd.notna(df_seq_subset['initial_retain_en'].iloc[0]):
                 plt.axhline(y=df_seq_subset['initial_retain_en'].iloc[0], color='blue', linestyle=':', label='Initial Retain EN')
            if 'initial_retain_ko' in df_seq_subset.columns and pd.notna(df_seq_subset['initial_retain_ko'].iloc[0]):
                 plt.axhline(y=df_seq_subset['initial_retain_ko'].iloc[0], color='orange', linestyle=':', label='Initial Retain KO')
            if 'initial_retain_hi' in df_seq_subset.columns and pd.notna(df_seq_subset['initial_retain_hi'].iloc[0]):
                 plt.axhline(y=df_seq_subset['initial_retain_hi'].iloc[0], color='green', linestyle=':', label='Initial Retain HI')

            plt.xticks(ticks=df_seq_subset['step_num'], labels=df_seq_subset['step_id'])
            plt.xlabel("Unlearning Step (Language/ID)")
            plt.ylabel("Retain Loss")
            # **** Title and Filename updated ****
            plt.title(f"Retain Loss Trend for {model_name}\n(Sequence: {sequence_str})")
            plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
            plt.grid(True)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            plot_filename = f"retain_loss_trend_seq_{sequence_str}.png"
            plt.savefig(os.path.join(model_plot_dir, plot_filename))
            plt.close()
            # print(f"  Saved: {plot_filename}")
        # else: # Optional: log if sequence data is empty for this model
            # print(f"  Skipping sequence {sequence_str} for model {model_name} (empty subset).")
    print(f"  Generated trend plots for {len(sequences_for_model)} sequences.")


    # --- Plotting Example 3 & 4: Final state comparison across sequences for THIS model ---

    # Get final step data for THIS model
    df_model_final_step = df_model.loc[df_model.groupby(['sequence'])['step_num'].idxmax()]

    # Plot Final Forget Loss Comparison (within the model, across sequences)
    if not df_model_final_step.empty:
        plt.figure(figsize=(12, 7))
        # **** REMOVED hue='model' ****
        sns.barplot(data=df_model_final_step, x='sequence', y='forget_loss_after')
        plt.xlabel("Unlearning Sequence")
        plt.ylabel("Final Forget Loss (After Last Step)")
        # **** Title and Filename updated ****
        plt.title(f"Final Forget Loss across Sequences for Model: {model_name}")
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y')
        plt.tight_layout()
        plot_filename = f"final_forget_loss_comparison.png" # Filename within model dir
        plt.savefig(os.path.join(model_plot_dir, plot_filename))
        plt.close()
        print(f"  Saved: {plot_filename}")
    else:
        print(f"  No final step data for {model_name} to plot forget loss comparison.")


    # Plot Final Retain Loss Change Comparison (within the model, across sequences)
    for retain_lang in ['en', 'ko', 'hi']: # Loop through retain languages
        retain_change_col = f"retain_{retain_lang}_change"
        if not df_model_final_step.empty and retain_change_col in df_model_final_step.columns:
            plt.figure(figsize=(12, 7))
             # **** REMOVED hue='model' ****
            sns.barplot(data=df_model_final_step, x='sequence', y=retain_change_col)
            plt.axhline(0, color='grey', linestyle='--')
            plt.xlabel("Unlearning Sequence")
            plt.ylabel(f"Retain Loss Change (Final - Initial) for {retain_lang}_retain")
             # **** Title and Filename updated ****
            plt.title(f"Final Retain Loss Change ({retain_lang}) across Sequences for Model: {model_name}")
            plt.xticks(rotation=45, ha='right')
            plt.grid(axis='y')
            plt.tight_layout()
            plot_filename = f"final_retain_change_{retain_lang}_comparison.png"
            plt.savefig(os.path.join(model_plot_dir, plot_filename))
            plt.close()
            print(f"  Saved: {plot_filename}")
        # else: # Optional: Check if column exists or data is empty
            # print(f"  No final step data or column '{retain_change_col}' for {model_name} to plot retain loss comparison.")

print(f"\nAll plots saved to subdirectories within: {OUTPUT_PLOT_DIR}")