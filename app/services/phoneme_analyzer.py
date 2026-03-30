import re
from collections import Counter
from jamo import h2j, j2hcj
from typing import Tuple

class PhonemeAnalyzerService:
    def clean_text(self, text: str) -> list[str]:
        text = re.sub(r'[^\w\s]', '', text)
        return text.split()
    
    def get_jamo_list(self, text: str) -> list[str]:
        jamo_str = j2hcj(h2j(text))
        return list(jamo_str)
    
    def get_alignment(self, original: str, stt: str) -> Tuple[str, str]:
        m, n = len(original), len(stt)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if original[i - 1] == stt[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + 1)
        
        align1, align2 = [], []
        i, j = m, n
        while i > 0 or j > 0:
            if i > 0 and j > 0 and original[i - 1] == stt[j - 1]:
                align1.append(original[i - 1])
                align2.append(stt[j - 1])
                i -= 1; j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
                align1.append(original[i - 1])
                align2.append(stt[j - 1])
                i -= 1; j -= 1
            elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
                align1.append(original[i - 1])
                align2.append("-")
                i -= 1
            else:
                align1.append("-")
                align2.append(stt[j - 1])
                j -= 1

        return align1[::-1], align2[::-1]
        
    def get_error_report(self, a1: list, a2: list) -> list[str]:
        errors = []
        for char1, char2 in zip(a1, a2):
            if char1 != char2:
                if char2 == "-":
                    errors.append(f"{char1} 탈락")
                elif char1 == "-":
                    errors.append(f"{char2} 첨가")
                else:
                    errors.append(f"{char1} -> {char2} 대치")
        return errors
        
    def analyze(self, original_text: str, stt_text: str) -> Tuple[list[str], list[str]]:
        org_words = self.clean_text(original_text)
        stt_words = self.clean_text(stt_text)

        aligned_org, aligned_stt = self.get_alignment(org_words, stt_words)

        total_errors = []
        error_words = []

        for o_word, s_word in zip(aligned_org, aligned_stt):
            if o_word != s_word:
                if o_word == "-" or s_word == "-":
                    total_errors.append(f"단어 {o_word if o_word != '-' else s_word} 읽기 오류")
                    error_words.append(o_word if o_word != "-" else s_word)
                else:
                    org_jamo = self.get_jamo_list(o_word)
                    stt_jamo = self.get_jamo_list(s_word)
                    a1, a2 = self.get_alignment(org_jamo, stt_jamo)
                    errors = self.get_error_report(a1, a2)
                    for error in errors:
                        total_errors.append(error)
                        error_words.append(o_word) 

        return total_errors, error_words
        
    def get_top_error(self, error_reports: list[str], error_words: list[str]) -> Tuple[str, str, int]:
        if not error_reports:
            raise ValueError("분석된 오류가 없습니다. 낭독 텍스트를 확인해주세요.")

         # 오류 패턴 → 해당 단어 매핑 (첫 번째 매칭 기준)
        report_to_word: dict[str, str] = {}
        for i, report in enumerate(error_reports):
            if report not in report_to_word and i < len(error_words):
                report_to_word[report] = error_words[i]

        counter = Counter(error_reports)
        top_pattern, top_count = counter.most_common(1)[0]
        target_word = report_to_word.get(top_pattern, error_words[0] if error_words else "")

        return top_pattern, target_word, top_count