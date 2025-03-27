from datasets import load_dataset
from dotenv import load_dotenv
import pandas as pd
import os

# .env 파일 로드
load_dotenv()

# 환경 변수에서 토큰 가져오기
hf_token = os.getenv("HF_TOKEN")

# Ensure the output directory exists
output_dir = "DB/TOFU"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load datasets
dataset_full = load_dataset("locuslab/TOFU", "full")
forget_01 = load_dataset("locuslab/TOFU", "forget01")
forget_05 = load_dataset("locuslab/TOFU", "forget05")

# Convert each dataset to pandas DataFrame
dataset_full_df = pd.DataFrame(dataset_full['train'])
forget_01_df = pd.DataFrame(forget_01['train'])
forget_05_df = pd.DataFrame(forget_05['train'])

# Save as valid JSON arrays (not line-delimited JSON)
dataset_full_df.to_json(os.path.join(output_dir, 'full.json'), orient='records', indent=2, force_ascii=False)
forget_01_df.to_json(os.path.join(output_dir, 'forget01.json'), orient='records', indent=2, force_ascii=False)
forget_05_df.to_json(os.path.join(output_dir, 'forget05.json'), orient='records', indent=2, force_ascii=False)

print("✅ JSON files have been saved in the 'TOFU' directory.")