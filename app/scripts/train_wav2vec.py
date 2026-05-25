import os
import json
import torch
import librosa
from datasets import Dataset
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, TrainingArguments, Trainer

# 1. 절대 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(BASE_DIR, "data", "labels")
AUDIO_DIR = os.path.join(BASE_DIR, "data", "audios")

FINAL_OUTPUT_DIR = os.path.join(BASE_DIR, "wav2vec2-child-stt-final")
MODEL_OUTPUT_DIR = os.path.join(BASE_DIR, "wav2vec2-child-stt-model")

print(f"▶ 데이터 경로 확인:\n - JSON: {JSON_DIR}\n - AUDIO: {AUDIO_DIR}\n")

# 2. 데이터셋 준비 함수
def prepare_dataset(json_dir, audio_base_path):
    data_list = []
    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"경로를 찾을 수 없습니다: {json_dir}")
        
    for filename in os.listdir(json_dir):
        if filename.endswith(".json"):
            with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            clean_text = data["Transcription"]["LabelText"].replace(".", "").strip()
            
            data_list.append({
                "audio_path": os.path.join(audio_base_path, data["File"]["FileName"]),
                "sentence": clean_text,
            })
    
    dataset = Dataset.from_list(data_list)
    return dataset.train_test_split(test_size=0.1)

# 3. 모델 및 프로세서 로드
model_id = "kresnik/wav2vec2-large-xlsr-korean"
processor = Wav2Vec2Processor.from_pretrained(model_id)
model = Wav2Vec2ForCTC.from_pretrained(
    model_id,
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.0,
    mask_time_prob=0.05,
    layerdrop=0.1,
    ctc_loss_reduction="mean",
    pad_token_id=processor.tokenizer.pad_token_id,
    vocab_size=len(processor.tokenizer),
    ignore_mismatched_sizes=True, 
)

model.freeze_feature_encoder()

# 4. 데이터셋 로드 및 전처리 수행
dataset = prepare_dataset(JSON_DIR, AUDIO_DIR)

def prepare_example(batch):
    speech, _ = librosa.load(batch["audio_path"], sr=16000)
    batch["input_values"] = processor(speech, sampling_rate=16000).input_values[0]
    
    # [버전 보정] 최신 문법 구조 반영
    batch["labels"] = processor(text=batch["sentence"]).input_ids
    return batch

print("▶ 데이터 전처리를 시작합니다 (map)...")
dataset = dataset.map(prepare_example, remove_columns=dataset.column_names["train"])

# 5. 최신 규격에 맞춘 Data Collator 클래스
@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입출력 특징 분리 추출
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # [버전 보정] 에러를 내던 augmentation=False 옵션 완벽 제거
        batch = self.processor.pad(input_features, return_tensors="pt")
        
        # [버전 보정] 에러를 내던 with as_target_processor() 문법을 제거하고 최신 labels 인자로 처리
        labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")

        # 손실 계산을 위한 패딩 토큰 마스킹 (-100)
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# 6. 학습 환경 설정 (최신 라이브러리 규격 맞춤)
training_args = TrainingArguments(
    output_dir=MODEL_OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    eval_strategy="steps",        # evaluation_strategy에서 변경 완료
    num_train_epochs=10,
    fp16=True, 
    save_steps=500,
    eval_steps=500,
    logging_steps=100,
    learning_rate=5e-5,
    warmup_steps=300,
    save_total_limit=1, 
)

# 7. Trainer 선언 및 학습
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
)

print("▶ [확인] 모든 관문 통과! 진짜 학습을 시작합니다...")
trainer.train()

# 8. 최종 결과 안전 저장
trainer.save_model(FINAL_OUTPUT_DIR)
print("🎉 [성공] 아동 음성 데이터 Fine-Tuning 완료 및 저장 성공!")