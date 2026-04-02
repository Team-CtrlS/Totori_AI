import re
from collections import Counter
from jamo import h2j, j2hcj
from typing import Tuple
from app.utils.alignment_utils import get_alignment

class PhonemeAnalyzerService:
    def _clean(self, text: str) -> list[str]:
        return re.sub(r'[^\w\s]', '', text).split()
    
    def _to_jamo(self, text: str) -> list[str]:
        return list(j2hcj(h2j(text)))
    
    def _get_errors(self, a1: list, a2: list) -> list[str]:
        errors = []
        for c1, c2 in zip(a1, a2):
            if c1 != c2:
                if c2 == "-":
                    errors.append(f"{c1} 탈락")
                elif c1 == "-":
                    errors.append(f"{c2} 첨가")
                else:
                    errors.append(f"{c1} -> {c2} 대치")
        return errors
    
    def analyze(self, original_text: str, stt_text: str) -> Tuple[list[str], list[str]]:
        org_words = self._clean(original_text)
        stt_words = self._clean(stt_text)
        aligned_org, aligned_stt = get_alignment(org_words, stt_words)

        error_reports, error_words = [], []
        for o_word, s_word in zip(aligned_org, aligned_stt):
            if o_word != s_word:
                if o_word == "-" or s_word == "-":
                    word = o_word if o_word != "-" else s_word
                    error_reports.append(f"단어 {word} 읽기 오류")
                    error_words.append(word)
                else:
                    a1, a2 = get_alignment(self._to_jamo(o_word), self._to_jamo(s_word))
                    for err in self._get_errors(a1, a2):
                        error_reports.append(err)
                        error_words.append(o_word)

        return error_reports, error_words
    
    def get_top_error(self, error_reports: list[str], error_words: list[str]) -> Tuple[str, str, int]:
        if not error_reports:
            raise ValueError("분석된 오류가 없습니다.")

        report_to_word: dict[str, str] = {}
        for report, word in zip(error_reports, error_words):
            if report not in report_to_word:
                report_to_word[report] = word

        top_pattern, top_count = Counter(error_reports).most_common(1)[0]
        return top_pattern, report_to_word[top_pattern], top_count