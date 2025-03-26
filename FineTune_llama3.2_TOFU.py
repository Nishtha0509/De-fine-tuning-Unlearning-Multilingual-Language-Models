import os
import json
import torch
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from peft import (
    LoraConfig, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
import bitsandbytes as bnb

def load_model_and_tokenizer(model_path):
    """
    심볼릭 링크된 모델 및 토크나이저를 로드하는 함수
    
    Args:
        model_path (str): 모델 경로
    
    Returns:
        tuple: (model, tokenizer)
    """
    try:
        # 토크나이저 로드
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=True,
            add_eos_token=True
        )
        
        # 패딩 토큰 설정
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # 모델 로드 (심볼릭 링크 지원)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # 로컬 파일만 사용
        )
        
        # 그래디언트 체크포인팅 활성화
        model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        raise

def load_json_dataset(json_path, tokenizer, max_length=512):
    """
    JSON 데이터셋 로드 및 토큰화
    
    Args:
        json_path (str): JSON 파일 경로
        tokenizer: 토크나이저
        max_length (int): 최대 시퀀스 길이
    
    Returns:
        datasets.Dataset: 토큰화된 데이터셋
    """
    # JSON 파일 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 데이터셋 토큰화
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=max_length
        )
    
    # HuggingFace 데이터셋으로 변환
    dataset = load_dataset('json', data_files=json_path)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    return tokenized_dataset['train']

def full_finetuning(model, tokenizer, dataset, output_dir):
    """
    전체 파인튜닝 수행
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        dataset: 토큰화된 데이터셋
        output_dir (str): 출력 디렉토리
    """
    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        learning_rate=5e-5,
        save_total_limit=3,
        push_to_hub=False
    )
    
    # 데이터 콜레이터 준비
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 트레이너 생성 및 훈련
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model(output_dir)

def lora_finetuning(model, tokenizer, dataset, output_dir):
    """
    LoRA 파인튜닝 수행
    
    Args:
        model: 모델
        tokenizer: 토크나이저
        dataset: 토큰화된 데이터셋
        output_dir (str): 출력 디렉토리
    """
    # 모델을 LoRA 훈련을 위해 준비
    model = prepare_model_for_kbit_training(model)
    
    # LoRA 설정
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # LoRA 모델 생성
    lora_model = get_peft_model(model, lora_config)
    
    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        learning_rate=1e-4,
        save_total_limit=3,
        push_to_hub=False
    )
    
    # 데이터 콜레이터 준비
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False
    )
    
    # 트레이너 생성 및 훈련
    trainer = Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator
    )
    
    trainer.train()
    trainer.save_model(output_dir)

def main():
    # 모델 및 토크나이저 경로 설정
    model_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/llama3.2_3b"
    json_path = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/DB/TOFU/full.json"
    
    # 출력 디렉토리 설정
    base_output_dir = "/scratch/jsong132/De-fine-tuning-Unlearning-Multilingual-Language-Models/outputs"
    full_ft_dir = os.path.join(base_output_dir, "full_finetuning")
    lora_ft_dir = os.path.join(base_output_dir, "lora_finetuning")
    
    # 출력 디렉토리 생성
    os.makedirs(full_ft_dir, exist_ok=True)
    os.makedirs(lora_ft_dir, exist_ok=True)
    
    try:
        # 모델 및 토크나이저 로드
        model, tokenizer = load_model_and_tokenizer(model_path)
        
        # 데이터셋 로드
        dataset = load_json_dataset(json_path, tokenizer)
        
        # 전체 파인튜닝 수행
        print("전체 파인튜닝 시작...")
        full_finetuning(model, tokenizer, dataset, full_ft_dir)
        print(f"전체 파인튜닝 완료. 모델 저장 위치: {full_ft_dir}")
        
        # LoRA 파인튜닝 수행
        print("LoRA 파인튜닝 시작...")
        lora_finetuning(model, tokenizer, dataset, lora_ft_dir)
        print(f"LoRA 파인튜닝 완료. 모델 저장 위치: {lora_ft_dir}")
    
    except Exception as e:
        print(f"파인튜닝 중 오류 발생: {e}")

if __name__ == "__main__":
    main()