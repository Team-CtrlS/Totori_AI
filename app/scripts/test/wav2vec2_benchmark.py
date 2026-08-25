"""
wav2vec2_benchmark.py
=====================
wav2vec2로 할 수 있는 모든 조합 테스트
  1. 모델 변형 (3종)
  2. CTC 디코딩 방식 (greedy / beam search)
  3. 오디오 전처리 조합 (5종)
  → 총 최대 30개 조합 비교

사용법:
  python wav2vec2_benchmark.py --audio_dir ./samples --ref_file references.json
  python wav2vec2_benchmark.py --audio_file sample.m4a --ref_text "캉아디가 마다에서 뒤어놀았스니다"
"""

import os
import json
import time
import argparse
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Optional

import torch
import numpy as np

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# 디바이스
# ══════════════════════════════════════════════
def get_device():
    if torch.cuda.is_available():      return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE = get_device()
INFER_DEVICE = "cpu" if DEVICE == "mps" else DEVICE  # wav2vec2는 mps 불안정
print(f"[Device] {DEVICE}  (추론: {INFER_DEVICE})")


# ══════════════════════════════════════════════
# 1. 테스트할 모델 목록
# ══════════════════════════════════════════════
WAV2VEC2_MODELS = [
    {
        "id": "kresnik/wav2vec2-large-xlsr-korean",
        "tag": "xlsr_korean",
        "desc": "XLSR 한국어 파인튜닝 (현재 사용 중)",
    },
    {
        "id": "facebook/wav2vec2-large-xlsr-53",
        "tag": "xlsr_53_multilang",
        "desc": "XLSR 다국어 베이스 (한국어 파인튜닝 없음, 비교용)",
    },
    # 아래 모델은 허깅페이스에서 접근 가능한 경우만 동작
    {
        "id": "snoop2head/wav2vec2-xls-r-300m-ko",
        "tag": "xls_r_300m_ko",
        "desc": "XLS-R 300M 한국어",
    },
]


# ══════════════════════════════════════════════
# 2. 오디오 전처리 파이프라인
# ══════════════════════════════════════════════
def load_raw(audio_path: str) -> np.ndarray:
    """전처리 없음 — 원본 그대로"""
    import librosa
    y, _ = librosa.load(audio_path, sr=16000, mono=True)
    return y.astype(np.float32)


def load_normalized(audio_path: str) -> np.ndarray:
    """음량 정규화만 — 최대진폭 기준"""
    y = load_raw(audio_path)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y


def load_rms_normalized(audio_path: str) -> np.ndarray:
    """RMS 정규화 — 평균 음량 기준 (더 자연스러운 볼륨 보정)"""
    y = load_raw(audio_path)
    rms = np.sqrt(np.mean(y ** 2))
    target_rms = 0.1
    if rms > 0:
        y = y * (target_rms / rms)
    y = np.clip(y, -1.0, 1.0)
    return y


def load_denoised(audio_path: str) -> np.ndarray:
    """노이즈 제거 + 음량 정규화"""
    try:
        import noisereduce as nr
    except ImportError:
        print("  [경고] noisereduce 미설치 → pip install noisereduce")
        print("         노이즈 제거 없이 정규화만 적용")
        return load_normalized(audio_path)

    y = load_raw(audio_path)
    y = nr.reduce_noise(y=y, sr=16000, stationary=False, prop_decrease=0.8)
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def load_trimmed(audio_path: str) -> np.ndarray:
    """무음 구간 제거 + 정규화 — 앞뒤 침묵이 긴 경우 효과적"""
    import librosa
    y = load_raw(audio_path)
    # top_db: 이 값보다 조용한 구간을 무음으로 간주 (낮을수록 덜 잘림)
    y_trimmed, _ = librosa.effects.trim(y, top_db=25)
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak
    return y_trimmed.astype(np.float32)


def load_preemphasis(audio_path: str) -> np.ndarray:
    """Pre-emphasis 필터 + 정규화 — 고주파 강조, 자음 인식에 유리"""
    y = load_raw(audio_path)
    # 일반적인 pre-emphasis 계수: 0.97
    y_pe = np.append(y[0], y[1:] - 0.97 * y[:-1])
    peak = np.max(np.abs(y_pe))
    if peak > 0:
        y_pe = y_pe / peak
    return y_pe.astype(np.float32)


PREPROCESSING_CONFIGS = [
    {"tag": "raw",          "fn": load_raw,          "desc": "전처리 없음"},
    {"tag": "normalized",   "fn": load_normalized,   "desc": "최대진폭 정규화"},
    {"tag": "rms_norm",     "fn": load_rms_normalized,"desc": "RMS 정규화"},
    {"tag": "denoised",     "fn": load_denoised,     "desc": "노이즈제거 + 정규화"},
    {"tag": "trimmed",      "fn": load_trimmed,      "desc": "무음제거 + 정규화"},
    {"tag": "preemphasis",  "fn": load_preemphasis,  "desc": "Pre-emphasis + 정규화"},
]


# ══════════════════════════════════════════════
# 3. CTC 디코딩 방식
# ══════════════════════════════════════════════
def decode_greedy(logits: torch.Tensor, processor) -> str:
    """Greedy — 매 프레임 argmax (현재 방식)"""
    predicted_ids = torch.argmax(logits, dim=-1)
    return processor.batch_decode(predicted_ids)[0].strip()


def decode_beam(logits: torch.Tensor, processor, beam_width: int = 10) -> str:
    """Beam search CTC 디코딩 — pyctcdecode 필요"""
    try:
        from pyctcdecode import build_ctcdecoder
    except ImportError:
        # fallback to greedy
        return decode_greedy(logits, processor)

    vocab = processor.tokenizer.get_vocab()
    # id → token 순서로 정렬
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    labels = [token for token, _ in sorted_vocab]

    num_logits = logits.shape[-1]
    if len(labels) > num_logits:
        labels = labels[:num_logits]
    elif len(labels) < num_logits:
        labels = labels + ["<unk>"] * (num_logits - len(labels))

    decoder = build_ctcdecoder(labels)
    log_probs = torch.log_softmax(logits, dim=-1)
    logits_np = log_probs.cpu().numpy()[0]

    text = decoder.decode(logits_np, beam_width=beam_width)
    return text.strip()


DECODING_CONFIGS = [
    {"tag": "greedy",     "fn": decode_greedy,                        "desc": "Greedy argmax"},
    {"tag": "beam10",     "fn": lambda l, p: decode_beam(l, p, 10),  "desc": "Beam search (width=10)"},
    {"tag": "beam50",     "fn": lambda l, p: decode_beam(l, p, 50),  "desc": "Beam search (width=50)"},
]


# ══════════════════════════════════════════════
# 4. 지표
# ══════════════════════════════════════════════
def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def cer(hyp, ref):
    if not ref: return 0.0
    return _edit_distance(list(hyp), list(ref)) / len(ref)

def wer(hyp, ref):
    if not ref: return 0.0
    h, r = hyp.split(), ref.split()
    if not r: return 0.0
    return _edit_distance(h, r) / len(r)


# ══════════════════════════════════════════════
# 5. 결과 구조
# ══════════════════════════════════════════════
@dataclass
class W2VResult:
    tag: str            # "xlsr_korean | denoised | beam10" 형태
    model_tag: str
    preproc_tag: str
    decode_tag: str
    audio_file: str
    hypothesis: str
    reference: str
    cer: float
    wer: float
    latency_sec: float
    audio_duration_sec: float = 0.0
    extra: dict = field(default_factory=dict)


def print_result(r: W2VResult):
    cer_s = f"CER={r.cer:.4f}" if r.reference else "CER=N/A"
    wer_s = f"WER={r.wer:.4f}" if r.reference else "WER=N/A"
    print(f"  [{r.tag:55s}] {cer_s}  {wer_s}  {r.latency_sec:.2f}s")
    print(f"    → '{r.hypothesis}'")


# ══════════════════════════════════════════════
# 6. 메인 벤치마크
# ══════════════════════════════════════════════
class Wav2Vec2Benchmark:

    def __init__(self):
        self._model_cache = {}  # model_id → (processor, model)

    def _get_model(self, model_id: str):
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if model_id not in self._model_cache:
            print(f"\n  [로딩] {model_id} ...")
            try:
                processor = Wav2Vec2Processor.from_pretrained(model_id)
                model = Wav2Vec2ForCTC.from_pretrained(model_id)
                model.to(INFER_DEVICE).eval()
                self._model_cache[model_id] = (processor, model)
                print(f"  [완료] {model_id}")
            except Exception as e:
                print(f"  [실패] {model_id}: {e}")
                self._model_cache[model_id] = None
        return self._model_cache[model_id]

    def _infer(self, audio: np.ndarray, processor, model) -> tuple[torch.Tensor, float]:
        inputs = processor(
            audio,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(INFER_DEVICE)
        t0 = time.perf_counter()
        with torch.no_grad():
            logits = model(input_values).logits
        latency = time.perf_counter() - t0
        return logits, latency

    def run_single(
        self,
        audio_path: str,
        reference: str = "",
        model_filter: Optional[list] = None,
        preproc_filter: Optional[list] = None,
        decode_filter: Optional[list] = None,
    ) -> list[W2VResult]:

        models    = [m for m in WAV2VEC2_MODELS
                     if model_filter is None or m["tag"] in model_filter]
        preprocs  = [p for p in PREPROCESSING_CONFIGS
                     if preproc_filter is None or p["tag"] in preproc_filter]
        decoders  = [d for d in DECODING_CONFIGS
                     if decode_filter is None or d["tag"] in decode_filter]

        results = []

        for model_cfg in models:
            loaded = self._get_model(model_cfg["id"])
            if loaded is None:
                # 모델 로드 실패 — 스킵
                continue
            processor, model = loaded

            for preproc_cfg in preprocs:
                # 오디오 로드 (전처리 포함)
                try:
                    audio = preproc_cfg["fn"](audio_path)
                    duration = len(audio) / 16000
                except Exception as e:
                    print(f"  [전처리 실패] {preproc_cfg['tag']}: {e}")
                    continue

                # 추론은 1번만 (디코딩만 다르게)
                try:
                    logits, latency = self._infer(audio, processor, model)
                except Exception as e:
                    print(f"  [추론 실패] {model_cfg['tag']} + {preproc_cfg['tag']}: {e}")
                    continue

                for decode_cfg in decoders:
                    tag = f"{model_cfg['tag']} | {preproc_cfg['tag']} | {decode_cfg['tag']}"
                    try:
                        hyp = decode_cfg["fn"](logits, processor)
                    except Exception as e:
                        hyp = f"[ERROR: {e}]"

                    r = W2VResult(
                        tag=tag,
                        model_tag=model_cfg["tag"],
                        preproc_tag=preproc_cfg["tag"],
                        decode_tag=decode_cfg["tag"],
                        audio_file=audio_path,
                        hypothesis=hyp,
                        reference=reference,
                        cer=round(cer(hyp, reference), 4) if reference else -1,
                        wer=round(wer(hyp, reference), 4) if reference else -1,
                        latency_sec=round(latency, 3),
                        audio_duration_sec=round(duration, 2),
                        extra={
                            "model_desc": model_cfg["desc"],
                            "preproc_desc": preproc_cfg["desc"],
                            "decode_desc": decode_cfg["desc"],
                            "rtf": round(latency / duration, 3) if duration > 0 else -1,
                        },
                    )
                    results.append(r)
                    print_result(r)

        return results

    def run_batch(
        self,
        audio_files: list[str],
        references: dict[str, str],
        **kwargs,
    ) -> list[W2VResult]:
        all_results = []
        for af in audio_files:
            ref = references.get(Path(af).name, "")
            print(f"\n{'─'*60}")
            print(f"파일: {af}")
            print(f"기준: '{ref}'" if ref else "기준: (없음)")
            print(f"{'─'*60}")
            all_results.extend(self.run_single(af, ref, **kwargs))
        return all_results


# ══════════════════════════════════════════════
# 7. 결과 저장 & 요약
# ══════════════════════════════════════════════
def save_results(results: list[W2VResult], out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")


def print_summary(results: list[W2VResult]):
    valid = [r for r in results if r.cer >= 0]
    if not valid:
        print("\n[요약] reference 없음 — CER 계산 불가")
        return

    # 조합별 평균 CER (여러 파일 기준)
    from collections import defaultdict
    tag_scores: dict[str, list] = defaultdict(list)
    for r in valid:
        tag_scores[r.tag].append(r.cer)

    avg_scores = {tag: np.mean(cers) for tag, cers in tag_scores.items()}
    sorted_tags = sorted(avg_scores.items(), key=lambda x: x[1])

    print("\n" + "═" * 80)
    print(f"{'조합 (모델 | 전처리 | 디코딩)':55s} {'평균CER':>8} {'샘플수':>6}")
    print("─" * 80)
    for tag, avg_cer in sorted_tags:
        n = len(tag_scores[tag])
        print(f"{tag:55s} {avg_cer:>8.4f} {n:>6}")
    print("═" * 80)

    best_tag, best_cer = sorted_tags[0]
    print(f"\n✅ 최저 평균 CER: [{best_tag}]  {best_cer:.4f}")

    # 전처리별 평균
    preproc_scores: dict[str, list] = defaultdict(list)
    for r in valid:
        preproc_scores[r.preproc_tag].append(r.cer)

    print("\n[전처리 방식별 평균 CER]")
    for tag, cers in sorted(preproc_scores.items(), key=lambda x: np.mean(x[1])):
        print(f"  {tag:15s}: {np.mean(cers):.4f}")

    # 디코딩별 평균
    decode_scores: dict[str, list] = defaultdict(list)
    for r in valid:
        decode_scores[r.decode_tag].append(r.cer)

    print("\n[디코딩 방식별 평균 CER]")
    for tag, cers in sorted(decode_scores.items(), key=lambda x: np.mean(x[1])):
        print(f"  {tag:15s}: {np.mean(cers):.4f}")

    # 모델별 평균
    model_scores: dict[str, list] = defaultdict(list)
    for r in valid:
        model_scores[r.model_tag].append(r.cer)

    print("\n[모델별 평균 CER]")
    for tag, cers in sorted(model_scores.items(), key=lambda x: np.mean(x[1])):
        print(f"  {tag:30s}: {np.mean(cers):.4f}")


def print_audio_info(audio_path: str):
    """오디오 품질 진단 — 실행 전 확인용"""
    import librosa
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration = len(y) / sr
    peak = float(np.max(np.abs(y)))
    rms = float(np.sqrt(np.mean(y ** 2)))
    silence_ratio = float(np.mean(np.abs(y) < 0.01))

    print(f"\n[오디오 진단] {audio_path}")
    print(f"  길이       : {duration:.2f}초")
    print(f"  최대진폭   : {peak:.4f}  {'⚠ 너무 작음' if peak < 0.05 else '✅'}")
    print(f"  RMS(음량)  : {rms:.4f}  {'⚠ 너무 작음(0.01 이하)' if rms < 0.01 else '✅'}")
    print(f"  무음비율   : {silence_ratio:.1%}  {'⚠ 무음 많음' if silence_ratio > 0.5 else '✅'}")


# ══════════════════════════════════════════════
# 8. CLI
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(description="wav2vec2 Benchmark")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--audio_file",  help="단일 오디오 파일")
    group.add_argument("--audio_dir",   help="오디오 디렉토리")
    p.add_argument("--ref_text",    default="", help="단일 파일 기준 텍스트")
    p.add_argument("--ref_file",    default="", help="JSON {파일명: 기준텍스트}")
    p.add_argument("--output_dir",  default="./results_w2v")
    p.add_argument("--exts", nargs="+", default=[".wav", ".mp3", ".m4a", ".flac"])

    # 필터 — 일부만 돌리고 싶을 때
    p.add_argument("--models",   nargs="+",
                   choices=[m["tag"] for m in WAV2VEC2_MODELS],
                   help="테스트할 모델 태그 (기본: 전체)")
    p.add_argument("--preprocs", nargs="+",
                   choices=[c["tag"] for c in PREPROCESSING_CONFIGS],
                   help="테스트할 전처리 (기본: 전체)")
    p.add_argument("--decoders", nargs="+",
                   choices=[d["tag"] for d in DECODING_CONFIGS],
                   help="테스트할 디코딩 (기본: 전체)")
    p.add_argument("--diagnose", action="store_true",
                   help="오디오 품질 진단만 출력하고 종료")
    return p.parse_args()


def main():
    args = parse_args()

    # 오디오 목록
    if args.audio_file:
        audio_files = [args.audio_file]
    else:
        audio_files = sorted([
            str(p) for p in Path(args.audio_dir).rglob("*")
            if p.suffix.lower() in args.exts
        ])

    if not audio_files:
        print("오디오 파일 없음")
        return

    # 진단만
    if args.diagnose:
        for af in audio_files:
            print_audio_info(af)
        return

    # 기준 텍스트
    if args.ref_file and os.path.exists(args.ref_file):
        with open(args.ref_file, encoding="utf-8") as f:
            references = json.load(f)
    elif args.ref_text and args.audio_file:
        references = {Path(args.audio_file).name: args.ref_text}
    else:
        references = {}

    # 설치 안내
    try:
        import noisereduce
    except ImportError:
        print("[안내] noisereduce 미설치 → denoised 전처리 비활성화")
        print("       pip install noisereduce  으로 설치 가능\n")

    try:
        import pyctcdecode
    except ImportError:
        print("[안내] pyctcdecode 미설치 → beam search 디코딩 greedy로 대체")
        print("       pip install pyctcdecode  으로 설치 가능\n")

    bench = Wav2Vec2Benchmark()
    results = bench.run_batch(
        audio_files=audio_files,
        references=references,
        model_filter=args.models,
        preproc_filter=args.preprocs,
        decode_filter=args.decoders,
    )

    out_path = os.path.join(args.output_dir, "wav2vec2_results.json")
    save_results(results, out_path)
    print_summary(results)


if __name__ == "__main__":
    main()
