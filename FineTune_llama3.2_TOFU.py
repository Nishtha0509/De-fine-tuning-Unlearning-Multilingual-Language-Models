from datasets import load_dataset
from transformers import Trainer, TrainingArguments


# TOFU 데이터셋 로드
dataset = load_dataset("locuslab/TOFU", "full")

# 데이터를 tokenization
def tokenize_function(examples):
    return tokenizer(examples['question'], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Fine-tuning을 위한 TrainingArguments 설정
training_args = TrainingArguments(
    output_dir="./results",          # 모델 출력 경로
    evaluation_strategy="epoch",     # 에폭마다 평가
    learning_rate=5e-5,              # 학습률
    per_device_train_batch_size=8,   # 배치 사이즈
    per_device_eval_batch_size=8,    # 평가 배치 사이즈
    num_train_epochs=3,              # 학습 에폭 수
    weight_decay=0.01,               # 가중치 감소
    logging_dir='./logs',            # 로그 디렉토리
    logging_steps=10,                # 로깅 빈도
    save_steps=500,                  # 체크포인트 저장 빈도
)

# Trainer 생성
trainer = Trainer(
    model=model,                         # fine-tuning할 모델
    args=training_args,                  # 학습 인자
    train_dataset=tokenized_datasets['train'],  # 학습 데이터셋
    eval_dataset=tokenized_datasets['test'],   # 평가 데이터셋
)

# 학습 시작
trainer.train()

# Fine-tuned 모델 저장
model.save_pretrained("./finetuned_llama3.2")
tokenizer.save_pretrained("./finetuned_llama3.2")

# 저장한 모델 로드
fine_tuned_model = LlamaForCausalLM.from_pretrained("./finetuned_llama3.2")
fine_tuned_tokenizer = LlamaTokenizer.from_pretrained("./finetuned_llama3.2")
