import json
import asyncio
from collections import Counter

from app.core.redis_client import get_redis
from app.services.whisper_loader import transcribe_with_timestamps
from app.services.phoneme_analyzer import PhonemeAnalyzerService
from app.services.josa_analyzer import JosaAnalyzerService, JosaEvent
from app.utils.alignment_utils import get_alignment, levenshtein

PHONEME_LEVELS = {"L1", "L2", "L3"}
JOSA_LEVELS = {"L4", "L5", "L6"}
REDIS_TTL = 10800 # 3시간

class ReadingService:
    def __init__(self):
        self._phoneme = PhonemeAnalyzerService()
        self._josa = JosaAnalyzerService()

    # 낭독 음성 분석 후 redis 저장
    async def analyze_and_store(
            self,
            audio_path: str,
            original_text: str,
            child_id: int,
            book_id: int,
            level: str,
    ) -> dict:
        if level not in PHONEME_LEVELS | JOSA_LEVELS:
            raise ValueError(f"유효하지 않은 레벨입니다: {level}")
        
        result = await asyncio.to_thread(transcribe_with_timestamps, audio_path, "raw")
        stt_text = result["text"]

        if not stt_text.strip():
            return
            
        errors = self._extract_errors(original_text, stt_text)
        wcpm = self._calc_wcpm(original_text, result, errors)

        del original_text, stt_text

        await self._store_to_redis(child_id, book_id, errors, wcpm)
        
    # 오류 추출
    def _extract_errors(self, original_text: str, stt_text: str) -> list[dict]:
        aligned_org, aligned_stt = get_alignment(
            self._phoneme._clean(original_text),
            self._phoneme._clean(stt_text),
            )
        josa_events = self._josa.analyze(original_text, stt_text)
        josa_by_stem: dict[str, JosaEvent] = {e.stem: e for e in josa_events}

        errors: list[dict] = []

        for org_w, stt_w in zip(aligned_org, aligned_stt):
            if org_w == stt_w:
                continue

            if org_w == "-" or stt_w == "-":
                word = org_w if org_w != "-" else stt_w
                errors.append({"type": "phoneme", "pattern": f"단어 {word} 읽기 오류", "word": word})
                continue

            phoneme_errs = self._word_phoneme_errors(org_w, stt_w)
            josa_events = _find_josa_event(org_w, josa_by_stem)

            if phoneme_errs and josa_events:
                # 겹치는 경우 어간/조사 경계로 주오류 판별
                if self._error_is_in_particle(org_w, stt_w, josa_events.stem):
                    errors.append(_josa_to_dict(josa_events))

                else:
                    errors.extend(phoneme_errs)
                josa_by_stem.pop(josa_events.stem, None)

            elif phoneme_errs:
                errors.extend(phoneme_errs)

            elif josa_events:
                errors.append(_josa_to_dict(josa_events))
                josa_by_stem.pop(josa_events.stem, None)
        
        for event in josa_by_stem.values():
            errors.append(_josa_to_dict(event))

        return errors
    
    # 음소 오류 반환
    def _word_phoneme_errors(self, org_word: str, stt_word: str) -> list[dict]:
        a1, a2 = get_alignment(
            self._phoneme._to_jamo(org_word),
            self._phoneme._to_jamo(stt_word),
        )
        return [
            {"type": "phoneme", "pattern": pattern, "word": org_word}
            for pattern in self._phoneme._get_errors(a1, a2)
        ]
    
    # 음소 오류가 조사 위치면 True, 어간 위치면 False
    def _error_is_in_particle(self, org_word: str, stt_word: str, stem: str) -> bool:
        n = len(self._phoneme._to_jamo(stem))
        org_jamo = self._phoneme._to_jamo(org_word)
        stt_jamo = self._phoneme._to_jamo(stt_word)

        stem_edit     = levenshtein(org_jamo[:n], stt_jamo[:n])
        particle_edit = levenshtein(org_jamo[n:], stt_jamo[n:] if len(stt_jamo) > n else [])

        if stem_edit == 0:
            return True
        if particle_edit == 0:
            return False
        return particle_edit > stem_edit

    # wcpm 계산
    def _calc_wcpm(self, original_text: str, stt_result: dict, errors: list[dict]) -> float | None:
        segments = stt_result.get("segments", [])
        if not segments:
            return None
        
        duration_sec = segments[-1]["end"] - segments[0]["start"]
        if duration_sec <= 0:
            return None
        
        total_words = len(self._phoneme._clean(original_text))
        if total_words == 0:
            return None
        
        error_words = {
            e["word"] if e["type"] == "phoneme" else e["stem"]
            for e in errors
        }
        correct_words = max(0, total_words - len(error_words))
        return round(correct_words / (duration_sec / 60), 1)

    async def _store_to_redis(self, child_id: int, book_id: int, errors: list[dict], wcpm: float | None) -> None:
        r = get_redis()
        async with r.pipeline() as pipe:
            for error in errors:
                pipe.rpush(_key(child_id, book_id), json.dumps(error, ensure_ascii=False))
            if wcpm is not None:
                pipe.rpush(_wcpm_key(child_id, book_id), str(wcpm))
            pipe.expire(_key(child_id, book_id), REDIS_TTL)
            pipe.expire(_wcpm_key(child_id, book_id), REDIS_TTL)
            await pipe.execute()

    async def get_errors(self, child_id: int, book_id: int) -> list[dict]:
        raw = await get_redis().lrange(_key(child_id, book_id), 0, -1)
        return [json.loads(e) for e in raw]

    async def delete_errors(self, child_id: int, book_id: int) -> None:
        await get_redis().delete(_key(child_id, book_id), _wcpm_key(child_id, book_id))

    # 동화 완료 시 모든 오류 패턴 및 wcpm 반환하고 redis에서 삭제
    async def get_all_and_delete(self, child_id: int, book_id: int) -> dict:
        r = get_redis()
        errors_raw, wcpm_raw = await asyncio.gather(
            r.lrange(_key(child_id, book_id), 0, -1),
            r.lrange(_wcpm_key(child_id, book_id), 0, -1),
        )
        await r.delete(_key(child_id, book_id), _wcpm_key(child_id, book_id))

        errors = [json.loads(e) for e in errors_raw]
        wcpm_values = [float(v) for v in wcpm_raw]
        avg_wcpm    = round(sum(wcpm_values) / len(wcpm_values), 1) if wcpm_values else None

        return {"errors": errors, "avg_wcpm": avg_wcpm}

    # 퀴즈 생성용 Top 오류 추출
    def get_top_phoneme_error(self, errors: list[dict]) -> tuple[str, str]:
        phoneme_errors = [e for e in errors if e["type"] == "phoneme"]
        if not phoneme_errors:
            raise ValueError("저장된 음소 오류가 없습니다.")
        
        top_pattern = Counter(e["pattern"] for e in phoneme_errors).most_common(1)[0][0]
        top_word    = next(e["word"] for e in phoneme_errors if e["pattern"] == top_pattern)
        return top_pattern, top_word

    def get_top_josa_error(self, errors: list[dict]) -> JosaEvent:
        josa_errors = [e for e in errors if e["type"] == "josa"]
        if not josa_errors:
            raise ValueError("저장된 조사 오류가 없습니다.")
        
        top_key = Counter(
            (e["kind"], e["target_josa"], e["stt_josa"]) for e in josa_errors
        ).most_common(1)[0][0]
        
        top = next(e for e in josa_errors if (e["kind"], e["target_josa"], e["stt_josa"]) == top_key)
        
        return JosaEvent(
            kind=top["kind"], stem=top["stem"],
            target_josa=top["target_josa"], stt_josa=top["stt_josa"],
        )

        
def _key(child_id: int, book_id: int) -> str:
    return f"reading_errors:{child_id}:{book_id}"

def _wcpm_key(child_id: int, book_id: int) -> str:
    return f"reading_wcpm:{child_id}:{book_id}"

def _find_josa_event(word: str, josa_by_stem: dict[str, JosaEvent]) -> JosaEvent | None:
    for stem, event in josa_by_stem.items():
        if word.startswith(stem) and len(word) > len(stem):
            return event
    return None

def _josa_to_dict(event: JosaEvent) -> dict:
    return {
        "type": "josa",
        "kind": event.kind,
        "stem": event.stem,
        "target_josa": event.target_josa,
        "stt_josa": event.stt_josa,
    }

_instance: ReadingService | None = None

def get_reading_service() -> ReadingService:
    global _instance
    if _instance is None:
        _instance = ReadingService()
    return _instance