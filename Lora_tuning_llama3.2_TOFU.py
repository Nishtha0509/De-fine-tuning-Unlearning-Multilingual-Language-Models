# Standard library imports
import json
import logging
import os
import re
from tqdm import tqdm

# Third-party imports
import numpy as np
import torch
from datasets import load_dataset, Dataset
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from peft.utils.other import fsdp_auto_wrap_policy

from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)
from trl import SFTTrainer, SFTConfig

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 설정 클래스
class ModelConfig:
    def __init__(self, name, model_path, output_dir):
        self.name = name
        self.model_path = model_path
        self.output_dir = output_dir

# 모델 설정들
MODEL_CONFIGS = [
    ModelConfig(
        name="lora-llama3.2:3b-klue-sts", 
        model_path="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b", 
        output_dir="/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU_Llamas/LoRA_TOFU_Llama"
    )
]

JSON_DATASET_PATH = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json"
MAX_LENGTH = 512
MAX_EVAL_SAMPLES = 200

# 데이터셋 준비 함수 수정
def prepare_dataset_json():
    """JSON 데이터셋 확인"""
    if os.path.exists(JSON_DATASET_PATH):
        logger.info(f"Dataset already exists: {JSON_DATASET_PATH}")
        return

    logger.info(f"Dataset file not found: {JSON_DATASET_PATH}")
    logger.info("Please make sure the JSON dataset file exists with question-answer pairs")
    raise FileNotFoundError(f"Dataset file not found: {JSON_DATASET_PATH}")

# 모델 및 토크나이저 로드 함수
def load_model_and_tokenizer(model_config):
    """LoRA를 위한 모델과 토크나이저 로드"""
    logger.info(f"Load model: {model_config.model_path}")

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_path, trust_remote_code=True)
    
    # 특수 토큰 확인 및 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4비트 양자화 모델 로드 (quantization_config만 사용)
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    )
    
    # 나머지 코드는 동일하게 유지
    model = prepare_model_for_kbit_training(model)
    print("Check the model architecture")
    # print(model)
    
    model = prepare_model_for_kbit_training(model)
    # model.print_trainable_parameters() # Only for OLMo
    
    return model, tokenizer

# 학습 데이터셋 클래스
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["input"]
        completion = item["output"]
        
        # 프롬프트와 완성 결합
        full_text = prompt + completion
        
        # 토큰화
        encoded = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 프롬프트 부분 토큰화 (라벨 마스킹용)
        prompt_encoded = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # 라벨 생성: 프롬프트 부분은 -100으로 마스킹
        labels = encoded["input_ids"].clone().squeeze(0)
        prompt_length = prompt_encoded["input_ids"].shape[1]
        labels[:prompt_length] = -100
        
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": labels
        }

# 메인 학습 함수
def train_model(model_config):
    # 데이터셋 준비
    prepare_dataset_json()
    
    # 데이터셋 로드
    logger.info("JSON loading...")
    with open(JSON_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 데이터 분할 (8:2 비율로 train과 validation 분할)
    total_items = len(data)
    train_size = int(total_items * 0.8)
    
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    logger.info(f"train data: {len(train_data)}, valid data: {len(val_data)}")
    
    # 모델 및 토크나이저 로드
    model, tokenizer = load_model_and_tokenizer(model_config)
    
    # 형식 변환 - question을 input으로, answer를 output으로 사용
    train_dataset = Dataset.from_dict({
        "text": [f"Question: {item['question']} Answer: {item['answer']}" for item in train_data]
    })
    
    val_dataset = Dataset.from_dict({
        "text": [f"Question: {item['question']} Answer: {item['answer']}" for item in val_data]
    })
    
    # LoRA 설정 추가
    peft_params = LoraConfig(
        lora_alpha=16,  # LoRA 스케일링 팩터
        lora_dropout=0.1,  # LoRA 드롭아웃 비율
        r=64,  # LoRA 랭크
        bias="none",  
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj"]  # Llama model
    )

    # 모델 및 토크나이저 로드 시 LoRA 설정 적용
    model = get_peft_model(model, peft_params)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=model_config.output_dir,
        eval_strategy="steps",
        eval_steps=200,
        learning_rate=2e-4,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=5,
        weight_decay=0.01,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=400,
        logging_dir="./logs",
        logging_steps=100,
        fp16=False,
        bf16=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # SFTTrainer 초기화 시 tokenizer와 packing 제거
    logger.info("Setting up SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_params,
    )

    # 학습 실행
    logger.info("Starting training...")
    trainer.train()
    
    # 최종 모델 저장 (PEFT 모델로)
    final_model_path = os.path.join(model_config.output_dir, "final")
    logger.info(f"Saving final model to: {final_model_path}")
    
    # PEFT 모델과 토크나이저 저장
    trainer.model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logger.info("Fine-tuning completed!")
    return model, tokenizer

# 메인 실행 함수
if __name__ == "__main__":
    # 각 모델별로 학습 및 평가 실행
    for model_config in MODEL_CONFIGS:
        # 출력 디렉토리 생성
        os.makedirs(model_config.output_dir, exist_ok=True)
        
        logger.info(f"Starting training for {model_config.name}")
        
        try:
            # 모델 학습
            model, tokenizer = train_model(model_config)
            
            logger.info(f"Completed training and evaluation for {model_config.name}")
        except Exception as e:
            logger.error(f"Error in model {model_config.name}: {e}")
            logger.exception("Exception details:")

