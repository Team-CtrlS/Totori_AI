import os
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from typing import Dict, Any

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_NAME = os.path.abspath(os.path.join(CURRENT_DIR, "..", "scripts", "wav2vec2-child-stt-final"))

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

_processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
_model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device).eval()

def transcribe_wav2vec(
    audio_path: str,
    preset: str = "raw"
) -> Dict[str, Any]:
    audio, _ = librosa.load(audio_path, sr=16000)
    duration = len(audio) / 16000

    # 추론
    inputs = _processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(device)

    with torch.no_grad():
        logits = _model(input_values).logits

    # 타임스탬프를 포함한 디코딩
    predicted_ids = torch.argmax(logits, dim=-1)
    outputs = _processor.batch_decode(predicted_ids, output_char_offsets=True)
    
    text = outputs.text[0]
    offsets = outputs.char_offsets[0]

    time_offset = 0.02 

    # 결과
    words_list = []
    for char_info in offsets:
        char_text = char_info["char"]
        if char_text == " ": continue 
        
        words_list.append({
            "word": char_text,
            "start": round(char_info["start_offset"] * time_offset, 2),
            "end": round(char_info["end_offset"] * time_offset, 2),
            "probability": None  # Greedy에서는 개별 확률 매핑이 복잡하여 None 처리
        })

    segments = [{
        "id": 0,
        "start": words_list[0]["start"] if words_list else 0.0,
        "end": words_list[-1]["end"] if words_list else round(duration, 2),
        "text": text.strip(),
        "words": words_list
    }]

    return {
        "model": MODEL_NAME,
        "language": "ko",
        "text": text.strip(),
        "segments": segments
    }