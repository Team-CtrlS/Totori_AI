"""
stt_benchmark_v2.py
===================
목적: 아동 발음 오류를 그대로 잡아내는 STT 모델/파라미터 비교
구성:
  [전처리]  RMS 정규화 → 모든 모델에 공통 적용
  [Whisper] 4종 파라미터 비교
  [wav2vec2] xlsr_korean | raw | greedy
             xlsr_korean | preemphasis | greedy
  [앙상블A] 어절 단위 교차 검증 (같으면 채택, 다르면 wav2vec2 우선)
  [앙상블B] 보정 신뢰도 기반 선택 (엔트로피 보정 후 스케일 통일)

사용법:
  python stt_benchmark_v2.py --audio_dir ./samples --ref_file references.json
  python stt_benchmark_v2.py --audio_file sample.m4a --ref_text "캉아디가 마다에서 뒤어놀았스니다"
"""

import os
import json
import time
import argparse
import tempfile
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any

import torch
import numpy as np

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════
# 디바이스
# ══════════════════════════════════════════════
def get_device() -> str:
    if torch.cuda.is_available():         return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"

DEVICE     = get_device()
INFER_DEV  = "cpu" if DEVICE == "mps" else DEVICE   # wav2vec2는 mps 불안정
print(f"[Device] {DEVICE}  (wav2vec2 추론: {INFER_DEV})")


# ══════════════════════════════════════════════
# 1. 오디오 전처리
# ══════════════════════════════════════════════
def load_and_preprocess(audio_path: str) -> tuple[np.ndarray, str]:
    """
    공통 전처리: 16kHz mono 로드 → RMS 정규화
    Returns:
        audio_np  : 정규화된 numpy 배열 (wav2vec2용)
        tmp_path  : 정규화된 오디오를 저장한 임시 wav 경로 (Whisper용)
    """
    import librosa, soundfile as sf

    y, _ = librosa.load(audio_path, sr=16000, mono=True)

    # RMS 정규화 (목표 RMS = 0.1)
    rms = np.sqrt(np.mean(y ** 2))
    if rms > 0:
        y = y * (0.1 / rms)
    y = np.clip(y, -1.0, 1.0).astype(np.float32)

    # Whisper는 파일 경로만 받으므로 임시 wav 저장
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, 16000)
    tmp.close()

    return y, tmp.name


def apply_preemphasis(audio: np.ndarray, coef: float = 0.97) -> np.ndarray:
    """고주파(자음) 강조 필터"""
    y = np.append(audio[0], audio[1:] - coef * audio[:-1])
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak
    return y.astype(np.float32)


def cleanup_tmp(path: str):
    try:
        os.unlink(path)
    except Exception:
        pass


# ══════════════════════════════════════════════
# 2. 평가 지표
# ══════════════════════════════════════════════
def _edit_dist(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = tmp
    return dp[n]

def cer(hyp: str, ref: str) -> float:
    if not ref: return 0.0
    return _edit_dist(list(hyp), list(ref)) / len(ref)

def wer(hyp: str, ref: str) -> float:
    if not ref: return 0.0
    h, r = hyp.split(), ref.split()
    return _edit_dist(h, r) / len(r) if r else 0.0


# ══════════════════════════════════════════════
# 3. 결과 구조
# ══════════════════════════════════════════════
@dataclass
class STTResult:
    model_tag:    str
    audio_file:   str
    hypothesis:   str
    reference:    str
    latency_sec:  float
    cer:          float = -1.0
    wer:          float = -1.0
    extra:        dict  = field(default_factory=dict)

def _score(r: STTResult, reference: str) -> STTResult:
    if reference:
        r.cer = round(cer(r.hypothesis, reference), 4)
        r.wer = round(wer(r.hypothesis, reference), 4)
    return r

def _print(r: STTResult):
    c = f"CER={r.cer:.4f}" if r.cer >= 0 else "CER=N/A"
    w = f"WER={r.wer:.4f}" if r.wer >= 0 else "WER=N/A"
    print(f"  [{r.model_tag:42s}] {c}  {w}  {r.latency_sec:.2f}s")
    print(f"    → '{r.hypothesis}'")


# ══════════════════════════════════════════════
# 4. Whisper (4종)
# ══════════════════════════════════════════════
WHISPER_GRID = [
    {
        "tag": "whisper_raw",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.7, beam_size=1, best_of=1,
            condition_on_previous_text=False,
        ),
    },
    {
        "tag": "whisper_no_fallback",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.0, beam_size=5, best_of=1,
            condition_on_previous_text=False,
        ),
    },
    {
        "tag": "whisper_large_raw",
        "model": "large-v3",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.7, beam_size=1, best_of=1,
            condition_on_previous_text=False,
        ),
    },
    {
        "tag": "whisper_balanced",
        "model": "medium",
        "params": dict(
            language="ko", task="transcribe",
            temperature=0.2, beam_size=5, best_of=1,
            condition_on_previous_text=True,
        ),
    },
]


class WhisperTester:
    _cache: dict[str, Any] = {}

    def _get(self, name: str):
        import whisper
        if name not in self._cache:
            print(f"  [Whisper] 로딩: {name} ...")
            self._cache[name] = whisper.load_model(name, device=DEVICE)
        return self._cache[name]

    def run(self, tmp_path: str, audio_file: str, reference: str) -> list[STTResult]:
        results = []
        for cfg in WHISPER_GRID:
            model  = self._get(cfg["model"])
            params = cfg["params"].copy()
            if DEVICE == "mps":
                params.pop("word_timestamps", None)

            t0 = time.perf_counter()
            try:
                raw = model.transcribe(tmp_path, fp16=False, **params)
                hyp = (raw.get("text") or "").strip()
            except Exception as e:
                hyp = f"[ERROR: {e}]"
            latency = time.perf_counter() - t0

            r = STTResult(
                model_tag=cfg["tag"], audio_file=audio_file,
                hypothesis=hyp, reference=reference,
                latency_sec=round(latency, 3),
                extra={"model": cfg["model"]},
            )
            _score(r, reference)
            results.append(r)
            _print(r)
        return results

    def transcribe_single(self, tmp_path: str) -> tuple[str, float]:
        """앙상블용: whisper_raw 설정으로 1회 추론 + avg_logprob 반환"""
        model = self._get("medium")
        params = dict(
            language="ko", task="transcribe",
            temperature=0.7, beam_size=1, best_of=1,
            condition_on_previous_text=False,
        )
        raw  = model.transcribe(tmp_path, fp16=False, **params)
        hyp  = (raw.get("text") or "").strip()
        segs = raw.get("segments", [])
        avg_lp = float(np.mean([s.get("avg_logprob", -1.0) for s in segs])) if segs else -1.0
        return hyp, avg_lp


# ══════════════════════════════════════════════
# 5. wav2vec2 (raw greedy / preemphasis greedy)
# ══════════════════════════════════════════════
WAV2VEC2_MODEL_ID = "kresnik/wav2vec2-large-xlsr-korean"

class Wav2Vec2Tester:
    _processor = None
    _model     = None

    def _get(self):
        if self._model is None:
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            print(f"  [wav2vec2] 로딩: {WAV2VEC2_MODEL_ID} ...")
            self._processor = Wav2Vec2Processor.from_pretrained(WAV2VEC2_MODEL_ID)
            self._model     = Wav2Vec2ForCTC.from_pretrained(WAV2VEC2_MODEL_ID)
            self._model.to(INFER_DEV).eval()
        return self._processor, self._model

    def _infer_logits(self, audio: np.ndarray) -> torch.Tensor:
        processor, model = self._get()
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            return model(inputs.input_values.to(INFER_DEV)).logits

    def _decode(self, logits: torch.Tensor) -> str:
        processor, _ = self._get()
        ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(ids)[0].strip()

    def _entropy_conf(self, logits: torch.Tensor) -> float:
        """엔트로피 기반 보정 신뢰도 (앙상블 B용)"""
        probs   = torch.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        return float(torch.exp(-entropy))

    def run(self, audio_raw: np.ndarray, audio_file: str, reference: str) -> list[STTResult]:
        results = []

        # ── raw greedy
        t0     = time.perf_counter()
        logits = self._infer_logits(audio_raw)
        hyp    = self._decode(logits)
        latency = time.perf_counter() - t0
        r = STTResult(
            model_tag="wav2vec2 | raw | greedy", audio_file=audio_file,
            hypothesis=hyp, reference=reference,
            latency_sec=round(latency, 3),
            extra={"preproc": "raw"},
        )
        _score(r, reference); results.append(r); _print(r)

        # ── preemphasis greedy
        audio_pe = apply_preemphasis(audio_raw)
        t0       = time.perf_counter()
        logits_pe = self._infer_logits(audio_pe)
        hyp_pe    = self._decode(logits_pe)
        latency   = time.perf_counter() - t0
        r = STTResult(
            model_tag="wav2vec2 | preemphasis | greedy", audio_file=audio_file,
            hypothesis=hyp_pe, reference=reference,
            latency_sec=round(latency, 3),
            extra={"preproc": "preemphasis"},
        )
        _score(r, reference); results.append(r); _print(r)

        return results

    def infer_with_conf(self, audio: np.ndarray) -> tuple[str, float]:
        """앙상블용: (텍스트, 엔트로피 보정 신뢰도) 반환"""
        logits = self._infer_logits(audio)
        hyp    = self._decode(logits)
        conf   = self._entropy_conf(logits)
        return hyp, conf


# ══════════════════════════════════════════════
# 6. 앙상블 A — 어절 단위 교차 검증
# ══════════════════════════════════════════════
def ensemble_a(
    w_text: str, v_text: str,
    audio_file: str, reference: str,
    latency: float,
    extra_info: dict,
) -> STTResult:
    """
    어절(공백 분리) 단위로 두 모델 출력 비교
      - 같으면 → 그대로 채택 (두 모델 모두 확신)
      - 다르면 → wav2vec2 우선 (언어모델 보정 없음)
      - 길이 차이 나는 나머지 → wav2vec2 기준
    """
    w_words = w_text.split()
    v_words = v_text.split()
    merged  = []
    match_count = 0

    max_len = max(len(w_words), len(v_words))
    for i in range(max_len):
        ww = w_words[i] if i < len(w_words) else None
        vw = v_words[i] if i < len(v_words) else None

        if ww is None:
            merged.append(vw)           # wav2vec2 남은 어절
        elif vw is None:
            merged.append(ww)           # Whisper 남은 어절
        elif ww == vw:
            merged.append(ww)           # 일치 → 채택
            match_count += 1
        else:
            merged.append(vw)           # 불일치 → wav2vec2 우선

    hyp = " ".join(merged)
    match_ratio = match_count / max_len if max_len > 0 else 0.0

    r = STTResult(
        model_tag="ensemble_A (어절교차)", audio_file=audio_file,
        hypothesis=hyp, reference=reference,
        latency_sec=round(latency, 3),
        extra={
            **extra_info,
            "strategy": "word_cross_validation",
            "match_ratio": round(match_ratio, 3),
            "whisper_text": w_text,
            "wav2vec2_text": v_text,
        },
    )
    _score(r, reference)
    return r


# ══════════════════════════════════════════════
# 7. 앙상블 B — 보정 신뢰도 기반 선택
# ══════════════════════════════════════════════
def ensemble_b(
    w_text: str, w_lp: float,
    v_text: str, v_conf: float,
    audio_file: str, reference: str,
    latency: float,
    extra_info: dict,
) -> STTResult:
    """
    두 모델의 신뢰도를 같은 0~1 스케일로 보정 후 비교
      - Whisper:  w_score = exp(avg_logprob)
                  avg_logprob -0.3 → 0.74  (높은 확신)
                  avg_logprob -0.7 → 0.50  (보통)
                  avg_logprob -1.2 → 0.30  (낮은 확신)
      - wav2vec2: v_score = exp(-entropy)
                  (softmax 분포가 뾰족할수록 높음, 항상 0.99이던 max_prob 대신 사용)
    → 더 높은 쪽 선택. 동점(0.05 이내)이면 wav2vec2 우선.
    """
    w_score = float(np.exp(w_lp))          # 0~1, Whisper 확신도
    v_score = v_conf                        # 0~1, wav2vec2 엔트로피 보정 확신도

    if w_score > v_score + 0.05:
        hyp      = w_text
        strategy = f"whisper_wins  (w={w_score:.3f} > v={v_score:.3f})"
    else:
        hyp      = v_text
        strategy = f"wav2vec2_wins (v={v_score:.3f} >= w={w_score:.3f})"

    r = STTResult(
        model_tag="ensemble_B (보정신뢰도)", audio_file=audio_file,
        hypothesis=hyp, reference=reference,
        latency_sec=round(latency, 3),
        extra={
            **extra_info,
            "strategy": strategy,
            "whisper_score": round(w_score, 4),
            "wav2vec2_score": round(v_score, 4),
            "whisper_text": w_text,
            "wav2vec2_text": v_text,
        },
    )
    _score(r, reference)
    return r


# ══════════════════════════════════════════════
# 8. 앙상블 실행기
# ══════════════════════════════════════════════
class EnsembleTester:
    def __init__(self, w_tester: WhisperTester, v_tester: Wav2Vec2Tester):
        self.w = w_tester
        self.v = v_tester

    def run(
        self,
        tmp_path: str, audio_raw: np.ndarray,
        audio_file: str, reference: str,
    ) -> list[STTResult]:
        print("  [Ensemble] Whisper 추론 중...")
        t0 = time.perf_counter()
        w_text, w_lp = self.w.transcribe_single(tmp_path)

        print("  [Ensemble] wav2vec2 추론 중...")
        v_text, v_conf = self.v.infer_with_conf(audio_raw)
        latency = time.perf_counter() - t0

        base_extra = {
            "whisper_logprob": round(w_lp, 4),
            "wav2vec2_entropy_conf": round(v_conf, 4),
        }

        r_a = ensemble_a(w_text, v_text, audio_file, reference, latency, base_extra)
        r_b = ensemble_b(w_text, w_lp, v_text, v_conf, audio_file, reference, latency, base_extra)

        _print(r_a)
        _print(r_b)
        return [r_a, r_b]


# ══════════════════════════════════════════════
# 9. 배치 실행
# ══════════════════════════════════════════════
def run_benchmark(
    audio_files: list[str],
    references: dict[str, str],
    run_whisper:  bool = True,
    run_wav2vec2: bool = True,
    run_ensemble: bool = True,
    output_dir:   str  = "./results",
) -> list[STTResult]:

    w_tester = WhisperTester()
    v_tester = Wav2Vec2Tester()
    e_tester = EnsembleTester(w_tester, v_tester)

    all_results: list[STTResult] = []

    for audio_path in audio_files:
        ref = references.get(Path(audio_path).name, "")
        print(f"\n{'─'*60}")
        print(f"파일: {audio_path}")
        print(f"기준: '{ref}'" if ref else "기준: (없음)")
        print(f"{'─'*60}")

        # ── 공통 전처리 (RMS 정규화)
        print("  [전처리] RMS 정규화 중...")
        try:
            audio_np, tmp_path = load_and_preprocess(audio_path)
        except Exception as e:
            print(f"  [전처리 실패] {e} — 원본 파일로 대체")
            import librosa
            audio_np, _ = librosa.load(audio_path, sr=16000, mono=True)
            tmp_path = audio_path  # 원본 경로 그대로

        try:
            if run_whisper:
                print("[Whisper]")
                all_results.extend(w_tester.run(tmp_path, audio_path, ref))

            if run_wav2vec2:
                print("[wav2vec2]")
                all_results.extend(v_tester.run(audio_np, audio_path, ref))

            if run_ensemble:
                print("[앙상블]")
                all_results.extend(e_tester.run(tmp_path, audio_np, audio_path, ref))

        finally:
            if tmp_path != audio_path:
                cleanup_tmp(tmp_path)

    # 저장
    out_path = Path(output_dir) / "benchmark_v2_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_results], f, ensure_ascii=False, indent=2)
    print(f"\n[저장] {out_path}")

    _print_summary(all_results)
    return all_results


# ══════════════════════════════════════════════
# 10. 요약 출력
# ══════════════════════════════════════════════
def _print_summary(results: list[STTResult]):
    from collections import defaultdict

    valid = [r for r in results if r.cer >= 0]
    if not valid:
        print("\n[요약] reference 없음 — CER 계산 불가")
        return

    # 모델별 평균 CER (여러 파일)
    scores: dict[str, list] = defaultdict(list)
    for r in valid:
        scores[r.model_tag].append(r.cer)

    print("\n" + "═" * 72)
    print(f"{'모델':42s} {'평균CER':>8} {'평균WER':>8} {'샘플':>4}")
    print("─" * 72)

    wer_scores: dict[str, list] = defaultdict(list)
    for r in valid:
        wer_scores[r.model_tag].append(r.wer)

    for tag, cers in sorted(scores.items(), key=lambda x: np.mean(x[1])):
        avg_c = np.mean(cers)
        avg_w = np.mean(wer_scores[tag])
        print(f"{tag:42s} {avg_c:>8.4f} {avg_w:>8.4f} {len(cers):>4}")

    print("═" * 72)
    best_tag = min(scores, key=lambda t: np.mean(scores[t]))
    print(f"\n✅ 최저 평균 CER: [{best_tag}]  {np.mean(scores[best_tag]):.4f}")

    # 앙상블 A vs B 비교 요약
    a_scores = scores.get("ensemble_A (어절교차)", [])
    b_scores = scores.get("ensemble_B (보정신뢰도)", [])
    if a_scores and b_scores:
        print(f"\n[앙상블 비교]")
        print(f"  A (어절교차)   평균 CER: {np.mean(a_scores):.4f}")
        print(f"  B (보정신뢰도) 평균 CER: {np.mean(b_scores):.4f}")
        winner = "A" if np.mean(a_scores) <= np.mean(b_scores) else "B"
        print(f"  → 앙상블 {winner} 우세")


# ══════════════════════════════════════════════
# 11. CLI
# ══════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--audio_file")
    g.add_argument("--audio_dir")
    p.add_argument("--ref_text",   default="")
    p.add_argument("--ref_file",   default="")
    p.add_argument("--output_dir", default="./results")
    p.add_argument("--exts", nargs="+", default=[".wav", ".mp3", ".m4a", ".flac"])
    p.add_argument("--no_whisper",   action="store_true")
    p.add_argument("--no_wav2vec2",  action="store_true")
    p.add_argument("--no_ensemble",  action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

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

    if args.ref_file and os.path.exists(args.ref_file):
        with open(args.ref_file, encoding="utf-8") as f:
            references = json.load(f)
    elif args.ref_text and args.audio_file:
        references = {Path(args.audio_file).name: args.ref_text}
    else:
        references = {}

    # soundfile 설치 확인
    try:
        import soundfile
    except ImportError:
        print("[필수] soundfile 미설치 → pip install soundfile")
        return

    run_benchmark(
        audio_files=audio_files,
        references=references,
        run_whisper=not args.no_whisper,
        run_wav2vec2=not args.no_wav2vec2,
        run_ensemble=not args.no_ensemble,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()