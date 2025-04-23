import json

# Load the original data
with open('/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/train/korean_retain99.json', 'r') as f:
    data = json.load(f)

# Take the first 7000 entries
subset = data[:7000]

# Save as 45.json
with open('retain_half.json', 'w') as f:
    json.dump(subset, f, indent=2)