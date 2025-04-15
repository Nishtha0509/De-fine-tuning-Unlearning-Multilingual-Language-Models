import json
import os
import re
import unicodedata
from rouge_score import rouge_scorer

# Setup input and output base directories
base_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(base_dir, "Generated_Answers", "epoch3", "Full_TOFU_Llama_ALL")
output_dir = os.path.join(base_dir, "TOFU_Evaluation_Results_RougeL", "epoch3", "Full_TOFU_Llama_ALL")
os.makedirs(output_dir, exist_ok=True)

# Helper to remove punctuation using Unicode categories
def remove_punctuation(text):
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in ["P", "S"])

# Clean prediction: strip list markers, remove punctuation, normalize
def clean_generated(text):
    text = re.sub(r"^\s*[\d一①⒈❶\-\*]+[\.:\)\-]?\s*", "", text.strip())
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    return unicodedata.normalize("NFKC", text.strip())

# Clean reference similarly
def clean_reference(text):
    text = remove_punctuation(text)
    text = re.sub(r"\s+", " ", text)
    return unicodedata.normalize("NFKC", text.strip())

# Process all JSON files in the input directory
for filename in os.listdir(input_dir):
    if not filename.endswith(".json"):
        continue

    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details", [])

    references = []
    predictions = []
    valid_indices = []

    # Gather valid reference/prediction pairs
    for i, entry in enumerate(details):
        ref = clean_reference(entry.get("ground_truth_answer", ""))
        pred = clean_generated(entry.get("generated_answer", ""))
        if ref and pred:
            references.append(ref)
            predictions.append(pred)
            valid_indices.append(i)

    # Compute detailed ROUGE-L (precision, recall, f1)
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    precisions, recalls, f1s = [], [], []

    for ref, pred in zip(references, predictions):
        score = scorer.score(ref, pred)["rougeL"]
        precisions.append(score.precision)
        recalls.append(score.recall)
        f1s.append(score.fmeasure)

    # Update each valid item with scores
    for idx, i in enumerate(valid_indices):
        details[i]["rougeL_precision"] = precisions[idx]
        details[i]["rougeL_recall"] = recalls[idx]
        details[i]["rougeL_f1"] = f1s[idx]

    # Add average scores to summary
    summary = data.get("summary", {})
    summary["average_rougeL_precision"] = sum(precisions) / len(precisions) if precisions else None
    summary["average_rougeL_recall"] = sum(recalls) / len(recalls) if recalls else None
    summary["average_rougeL_f1"] = sum(f1s) / len(f1s) if f1s else None

    data["summary"] = summary
    data["details"] = details

    # Save updated JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"ROUGE-L metrics saved to {output_path}")