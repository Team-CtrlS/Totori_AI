# STT 벤치마크 — 난독 아동 발음 오류 검출

## 파일 구조
```
stt_benchmark/
├── stt_benchmark.py    # 전체 벤치마크 (Whisper + wav2vec2 + 앙상블)
├── quick_test.py       # 단일 파일 빠른 테스트
├── references.json     # 기준 텍스트 (파일명 → 정답)
└── README.md
```

## 설치
```bash
pip install openai-whisper transformers librosa torch torchaudio
```

---

## 실행 방법

### 1. 빠른 단일 파일 테스트
```bash
python quick_test.py sample.wav "강아지가 마당에서 뛰어놀았습니다"
```

### 2. 전체 벤치마크 (단일 파일)
```bash
python stt_benchmark.py \
  --audio_file sample.wav \
  --ref_text "강아지가 마당에서 뛰어놀았습니다"
```

### 3. 디렉토리 배치 실행 + 기준 파일
```bash
python stt_benchmark.py \
  --audio_dir ./samples \
  --ref_file references.json \
  --output_dir ./results
```

### 4. 특정 모델만 실행
```bash
# Whisper만
python stt_benchmark.py --audio_file sample.wav --no_wav2vec2 --no_ensemble

# wav2vec2만
python stt_benchmark.py --audio_file sample.wav --no_whisper --no_ensemble
```

---

## 비교 모델/파라미터

| 태그 | 모델 | 전략 | 핵심 파라미터 |
|------|------|------|--------------|
| `whisper_raw_ultra` | medium | 발음 그대로 최우선 | temp=1.0, beam=1, logprob=-1.5 |
| `whisper_raw` | medium | 기존 raw (베이스라인) | temp=0.7, beam=1 |
| `whisper_no_fallback` | medium | Greedy + 느슨한 임계값 | temp=0.0, beam=5, logprob=-2.0 |
| `whisper_large_raw` | large-v3 | 강력 모델 + raw | temp=0.7, beam=1 |
| `whisper_large_no_fallback` | large-v3 | 강력 모델 + greedy | temp=0.0, beam=5 |
| `whisper_balanced` | medium | 비교군 (보정 있음) | temp=0.2, beam=5, prev_ctx=True |
| `wav2vec2_*` | xlsr-korean | CTC 디코딩 (보정 無) | - |
| `ensemble` | medium + wav2vec2 | 신뢰도 기반 선택 | logprob + conf 비교 |

---

## 핵심 파라미터 해설

### Whisper
| 파라미터 | 역할 | 발음 오류 검출 관점 |
|---------|------|------------------|
| `temperature` | 샘플링 다양성 | ↑높을수록 보정 약해짐 |
| `beam_size` | 탐색 너비 | 1이면 greedy → 보정 최소 |
| `condition_on_previous_text` | 이전 문맥 반영 | False → 독립 인식 |
| `logprob_threshold` | 낮은 확률 토큰 거부 | 낮출수록 더 많은 토큰 허용 |
| `compression_ratio_threshold` | 반복/이상 감지 | 높일수록 이상 발화 허용 |

### wav2vec2 vs Whisper
| 구분 | Whisper | wav2vec2 |
|------|---------|----------|
| 언어 모델 보정 | 있음 (큰 영향) | 없음 (CTC만) |
| 발음 오류 보존 | 보정으로 일부 손실 | 발음 그대로 |
| 한국어 품질 | 우수 | 파인튜닝 모델 의존 |
| 속도 | 빠름 | 빠름 |

---

## 결과 해석 가이드

- **CER (Character Error Rate)**: 낮을수록 기준 텍스트에 가까움
- **발음 오류 검출 목적**: CER이 오히려 높은 모델이 "오류를 잘 잡는 것"일 수 있음
  → 반드시 출력 텍스트를 육안으로 확인할 것
- **권장 순서**:
  1. `quick_test.py`로 빠르게 출력 육안 확인
  2. 오류 잡히는 모델 후보 2~3개 선정
  3. `stt_benchmark.py`로 다수 샘플에 배치 검증
  4. CER + 육안 검토 병행하여 최종 모델 선택
