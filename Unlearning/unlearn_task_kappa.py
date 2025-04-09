import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer # 또는 AutoModelForCausalLM
import copy
from tqdm import tqdm # 진행 상황 표시용 (선택 사항)

# --- 1. 설정 (Configuration) ---
MODEL_NAME = "google/mt5-small"  # 예시 다국어 모델 (실제 모델로 변경)
LANGUAGES = ["en", "ko", "hi"]
LEARNING_RATE = 5e-5 # Task Vector 계산을 위한 학습률 (조정 필요)
EPOCHS_PER_LANG = 1 # 각 언어별 Task Vector 계산 시 파인튜닝 에포크 (짧게)
LAMBDA_LR = 1.0     # 유지 손실(Lr) 가중치 (조정 필요)
ALPHA_MERGE = 1.0   # 최종 모델 결합 시 Task Vector 가중치 (보통 1.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- [중요] 실제 데이터 로딩 및 전처리 함수 ---
# 이 부분은 실제 데이터 형식에 맞게 구현해야 합니다.
# 예시: 각 언어별로 잊을 데이터(forget), 유지할 데이터(retain) 리스트를 반환한다고 가정
# 각 데이터는 모델 입력 형식 (e.g., {"input_ids": tensor, "attention_mask": tensor})으로 토크나이징 되어야 함
def load_and_tokenize_data(language, tokenizer):
    # === 실제 구현 필요 ===
    # 예시 플레이스홀더 (실제 데이터로 대체해야 함)
    # forget_samples = [...] # 잊을 데이터 샘플 리스트 (텍스트)
    # retain_samples = [...] # 유지할 데이터 샘플 리스트 (텍스트)
    print(f"Warning: Using placeholder data for language '{language}'. Implement actual data loading.")

    # 예시 토크나이징 (모델과 태스크에 맞게 수정 필요)
    dummy_text = f"This is dummy {language} forget text."
    tokenized_forget = tokenizer(dummy_text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    # 실제로는 forget_samples 리스트를 배치 단위로 처리해야 함
    forget_dataloader = [(tokenized_forget, tokenized_forget['input_ids'])] # (batch, labels) 형태 예시

    dummy_text_r = f"This is dummy {language} retain text."
    tokenized_retain = tokenizer(dummy_text_r, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    # 실제로는 retain_samples 리스트를 배치 단위로 처리해야 함
    retain_dataloader = [(tokenized_retain, tokenized_retain['input_ids'])] # (batch, labels) 형태 예시

    return forget_dataloader, retain_dataloader
    # =======================

# --- 2. 모델 및 토크나이저 로드 ---
print(f"Loading model '{MODEL_NAME}' and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# Seq2Seq 모델 예시 (Causal LM은 AutoModelForCausalLM 사용)
pretrained_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
pretrained_model.eval() # 원본 모델은 평가 모드로 유지 (가중치 변경 방지)
original_state_dict = copy.deepcopy(pretrained_model.state_dict()) # 원본 가중치 저장

# --- 3. 언러닝 손실 함수 정의 ---
# Seq2Seq 모델의 경우, labels 인자를 주면 자동으로 CrossEntropyLoss 계산
# Lf는 forget set에 대한 loss를 최대화 (-loss 최소화), Lr은 retain set에 대한 loss 최소화
def calculate_unlearning_loss(model, batch_forget, labels_forget, batch_retain, labels_retain, lambda_lr):
    # Lr: Retain Set에 대한 일반적인 손실 (최소화 목표)
    outputs_retain = model(**batch_retain, labels=labels_retain)
    loss_r = outputs_retain.loss

    # Lf: Forget Set에 대한 손실 (최대화 목표 -> 음수 손실을 최소화)
    outputs_forget = model(**batch_forget, labels=labels_forget)
    # 간단한 Lf 구현: 일반 손실 값에 음수를 취함 (이 값을 최소화하면 원래 손실은 최대화됨)
    loss_f = -outputs_forget.loss
    # 다른 Lf 전략:
    # - 예측 확률과 균등 분포 간 KL Divergence 최소화
    # - 원래 모델 예측과의 KL Divergence 최소화 (Gradient Ascent 방식)

    combined_loss = loss_f + lambda_lr * loss_r
    return combined_loss, loss_f, loss_r

# --- 4. 언어별 Task Vector 계산 ---
task_vectors = {}
for lang in LANGUAGES:
    print(f"\n--- Calculating Task Vector for language: {lang} ---")

    # 4.1 데이터 로드 및 준비
    forget_loader, retain_loader = load_and_tokenize_data(lang, tokenizer)
    # 실제로는 두 로더의 길이를 맞추거나, 번갈아 사용하는 로직 필요
    # 여기서는 간단히 각 로더에 샘플이 하나씩 있다고 가정

    # 4.2 모델 복사 및 학습 설정
    model_for_task = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    model_for_task.load_state_dict(original_state_dict) # 원본 가중치로 시작
    model_for_task.train() # 학습 모드 설정
    optimizer = optim.AdamW(model_for_task.parameters(), lr=LEARNING_RATE)

    # 4.3 짧은 파인튜닝 (Task Vector 생성을 위해)
    for epoch in range(EPOCHS_PER_LANG):
        print(f"  Epoch {epoch+1}/{EPOCHS_PER_LANG}")
        model_for_task.train()
        total_loss, total_lf, total_lr = 0, 0, 0
        num_batches = 0

        # 실제로는 DataLoader 사용 필요. 여기서는 예시로 각 로더에서 하나씩 가져옴
        # 두 로더의 배치를 번갈아 사용하거나, 미니배치 내에서 섞는 방식 고려
        try:
             batch_f, labels_f = next(iter(forget_loader))
             batch_r, labels_r = next(iter(retain_loader))
             num_batches += 1

             optimizer.zero_grad()
             # .to(DEVICE)는 데이터 로딩 함수에서 처리하는 것이 더 좋음
             loss, lf, lr = calculate_unlearning_loss(model_for_task,
                                                      batch_f, labels_f.to(DEVICE),
                                                      batch_r, labels_r.to(DEVICE),
                                                      LAMBDA_LR)
             loss.backward()
             optimizer.step()

             total_loss += loss.item()
             total_lf += lf.item()
             total_lr += lr.item()

        except StopIteration:
             # 실제 데이터 로더 사용 시 루프 종료 처리
             pass # 예시에서는 한 번만 실행

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_lf = total_lf / num_batches
            avg_lr = total_lr / num_batches
            print(f"  Avg Loss: {avg_loss:.4f}, Avg Lf: {avg_lf:.4f}, Avg Lr: {avg_lr:.4f}")
        else:
            print("  No batches processed.")


    # 4.4 Task Vector 계산 (finetuned_params - original_params)
    task_vector_lang = {}
    finetuned_state_dict = model_for_task.state_dict()
    with torch.no_grad():
        for name, param_finetuned in finetuned_state_dict.items():
            param_original = original_state_dict[name].to(DEVICE) # 원본도 같은 디바이스로
            task_vector_lang[name] = param_finetuned.data - param_original.data # .data 사용 주의
    task_vectors[lang] = task_vector_lang

    # 메모리 확보 (선택 사항)
    del model_for_task
    torch.cuda.empty_cache()

# --- 5. Task Vector 결합 ---
print("\n--- Combining Task Vectors ---")
combined_task_vector = {}
if task_vectors:
    # 모든 언어 Task Vector의 평균 계산
    param_names = task_vectors[LANGUAGES[0]].keys() # 첫 번째 언어의 파라미터 이름 기준
    for name in tqdm(param_names, desc="Combining vectors"):
        # 각 언어의 해당 파라미터 벡터를 스택
        vectors_for_param = [task_vectors[lang][name] for lang in LANGUAGES if name in task_vectors[lang]]
        if vectors_for_param:
            # 스택 후 평균 계산
            combined_task_vector[name] = torch.stack(vectors_for_param, dim=0).mean(dim=0)
    print("Task vectors combined.")
else:
    print("No task vectors were calculated.")

# --- 6. 최종 모델 생성 (원본 모델 + 결합된 Task Vector) ---
print("\n--- Creating Final Unlearned Model ---")
unlearned_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE) # 새 모델 로드
unlearned_model.load_state_dict(original_state_dict) # 원본 가중치 적용

if combined_task_vector:
    with torch.no_grad():
        updated_state_dict = unlearned_model.state_dict()
        for name, param in tqdm(updated_state_dict.items(), desc="Applying combined vector"):
            if name in combined_task_vector:
                # 원본 파라미터에 결합된 task vector를 더함 (ALPHA_MERGE로 스케일링 가능)
                param.data += ALPHA_MERGE * combined_task_vector[name] # .data 사용 주의
    unlearned_model.load_state_dict(updated_state_dict)
    print("Combined task vector applied to the model.")
else:
    print("No combined task vector to apply.")

unlearned_model.eval() # 최종 모델은 평가 모드로

# --- 7. 결과 (선택 사항) ---
# unlearned_model 을 저장하거나, 평가 데이터셋으로 성능 검증
# 예: unlearned_model.save_pretrained("./unlearned_mt5_model")
# tokenizer.save_pretrained("./unlearned_mt5_model")

print("\n--- Unlearning process finished ---")
# print("Evaluate the 'unlearned_model' on forget and retain sets for all languages.")