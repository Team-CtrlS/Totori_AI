"""
quick_test.py — 빠른 단일 파일 테스트
======================================
벤치마크 전체를 돌리기 전에, 특정 모델/파라미터만 골라서
빠르게 결과를 확인할 때 사용.

사용:
  python quick_test.py sample.wav "아이가 소리내어 읽었어요"
"""

import sys
import time
import json
import torch
import warnings
warnings.filterwarnings("ignore")


def get_device():
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = get_device()


# ─────────────────────────────────────────────
# 개별 Whisper 파라미터 집합
# ─────────────────────────────────────────────
CONFIGS = [
    {
        "tag": "raw_ultra",
        "desc": "temperature 최대, beam 없음, 보정 없음",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=1.0, beam_size=1, best_of=1,
            condition_on_previous_text=False,
            logprob_threshold=-1.5,
        )
    },
    {
        "tag": "raw",
        "desc": "temperature 0.7, beam 없음",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.7, beam_size=1, best_of=1,
            condition_on_previous_text=False,
        )
    },
    {
        "tag": "no_fallback",
        "desc": "greedy + logprob 느슨하게",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.0, beam_size=5, best_of=1,
            condition_on_previous_text=False,
            logprob_threshold=-2.0,
        )
    },
    {
        "tag": "balanced",
        "desc": "기존 balanced (비교군)",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.2, beam_size=5, best_of=1,
            condition_on_previous_text=True,
        )
    },
]


def test_whisper(audio_path: str, reference: str = ""):
    import whisper
    _models = {}

    print(f"\n{'═'*60}")
    print(f"[Whisper 테스트]  파일: {audio_path}")
    print(f"{'═'*60}")

    for cfg in CONFIGS:
        mn = cfg["model"]
        if mn not in _models:
            print(f"  모델 로딩: {mn}...")
            _models[mn] = whisper.load_model(mn, device=DEVICE)

        model = _models[mn]
        params = cfg["params"].copy()
        if DEVICE == "mps":
            params.pop("word_timestamps", None)
        else:
            params["word_timestamps"] = True

        t0 = time.perf_counter()
        try:
            result = model.transcribe(audio_path, fp16=False, **params)
            text = (result.get("text") or "").strip()

            # 세그먼트별 logprob 출력 (발음 확신도 파악용)
            segs = result.get("segments", [])
            avg_lp = sum(s.get("avg_logprob", 0) for s in segs) / len(segs) if segs else 0

        except Exception as e:
            text = f"[ERROR] {e}"
            avg_lp = -99

        elapsed = time.perf_counter() - t0

        print(f"\n  ▸ [{cfg['tag']}] {cfg['desc']}")
        print(f"    출력  : '{text}'")
        print(f"    logprob: {avg_lp:.3f}  |  {elapsed:.2f}s")

        if reference:
            from stt_benchmark import cer, wer
            print(f"    CER   : {cer(text, reference):.4f}")
            print(f"    WER   : {wer(text, reference):.4f}")

    # 세그먼트 상세 (마지막 결과 기준)
    if reference:
        print(f"\n  기준 텍스트: '{reference}'")


def test_wav2vec2(audio_path: str, reference: str = ""):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    import librosa
    import numpy as np

    MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"
    infer_dev = "cpu" if DEVICE == "mps" else DEVICE

    print(f"\n{'═'*60}")
    print(f"[wav2vec2 테스트]  파일: {audio_path}")
    print(f"{'═'*60}")
    print(f"  모델 로딩: {MODEL_ID}...")

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(infer_dev)
    model.eval()

    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    input_values = inputs.input_values.to(infer_dev)

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(input_values).logits

    probs = torch.softmax(logits, dim=-1)
    avg_conf = float(probs.max(dim=-1).values.mean())

    predicted_ids = torch.argmax(logits, dim=-1)
    text = processor.batch_decode(predicted_ids)[0].strip()
    elapsed = time.perf_counter() - t0

    print(f"\n  출력    : '{text}'")
    print(f"  avg_conf: {avg_conf:.4f}  |  {elapsed:.2f}s")

    if reference:
        from stt_benchmark import cer, wer
        print(f"  CER     : {cer(text, reference):.4f}")
        print(f"  WER     : {wer(text, reference):.4f}")
        print(f"  기준    : '{reference}'")


def main():
    if len(sys.argv) < 2:
        print("사용법: python quick_test.py <audio.wav> [기준텍스트]")
        sys.exit(1)

    audio_path = sys.argv[1]
    reference = sys.argv[2] if len(sys.argv) > 2 else ""

    print(f"디바이스: {DEVICE}")

    test_whisper(audio_path, reference)

    try:
        test_wav2vec2(audio_path, reference)
    except ImportError:
        print("\n[wav2vec2] transformers 미설치 — pip install transformers librosa")


if __name__ == "__main__":
    main()
