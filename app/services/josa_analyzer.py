import re, os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import Counter
from konlpy.tag import Mecab

os.environ["MECABRC"] = "/opt/homebrew/etc/mecabrc"

@dataclass
class StemJosa:
    stem: str
    josa: Optional[str]

@dataclass
class JosaEvent:
    kind: str
    stem: str
    target_josa: Optional[str]
    stt_josa: Optional[str]

class JosaAnalyzerService:
    def __init__(self):
        self.tagger = Mecab(dicpath="/opt/homebrew/opt/mecab-ko-dic/lib/mecab/dic/mecab-ko-dic")

    # 형태소 분석
    @staticmethod
    def _is_noun_like(tag: str) -> bool:
        return tag.startswith("NN") or tag in ("NP", "NR")

    @staticmethod
    def _is_josa(tag: str) -> bool:
        return tag.startswith("J")

    def _extract_stemjosa(self, text: str) -> List[StemJosa]:
        tokens: List[Tuple[str, str]] = self.tagger.pos(text)
        pairs: List[StemJosa] = []
        i = 0
        while i < len(tokens):
            morph, tag = tokens[i]
            if self._is_noun_like(tag):
                josa = None
                if i + 1 < len(tokens) and self._is_josa(tokens[i + 1][1]):
                    josa = tokens[i + 1][0]
                    i += 2
                else:
                    i += 1
                pairs.append(StemJosa(stem=morph, josa=josa))
            else:
                i += 1
        return pairs

    # 정렬
    @staticmethod
    def _levenshtein(s1: str, s2: str) -> int:
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
        return dp[m][n]

    def _align(
        self,
        target: List[StemJosa],
        stt: List[StemJosa],
    ) -> List[Tuple[Optional[StemJosa], Optional[StemJosa]]]:
        lenT, lenS = len(target), len(stt)

        def sub_cost(a: StemJosa, b: StemJosa) -> int:
            return min(3, self._levenshtein(a.stem, b.stem))

        dp = [[0] * (lenS + 1) for _ in range(lenT + 1)]
        bt = [[None] * (lenS + 1) for _ in range(lenT + 1)]

        for i in range(1, lenT + 1):
            dp[i][0] = i
            bt[i][0] = ("DEL", i - 1, 0)
        for j in range(1, lenS + 1):
            dp[0][j] = j
            bt[0][j] = ("INS", 0, j - 1)

        for i in range(1, lenT + 1):
            for j in range(1, lenS + 1):
                c_sub = dp[i - 1][j - 1] + sub_cost(target[i - 1], stt[j - 1])
                c_del = dp[i - 1][j] + 1
                c_ins = dp[i][j - 1] + 1
                best = min(c_sub, c_del, c_ins)
                dp[i][j] = best
                if best == c_sub:
                    bt[i][j] = ("SUB", i - 1, j - 1)
                elif best == c_del:
                    bt[i][j] = ("DEL", i - 1, j)
                else:
                    bt[i][j] = ("INS", i, j - 1)

        aligned = []
        i, j = lenT, lenS
        while i > 0 or j > 0:
            op, pi, pj = bt[i][j]
            if op == "SUB":
                aligned.append((target[pi], stt[pj]))
            elif op == "DEL":
                aligned.append((target[pi], None))
            else:
                aligned.append((None, stt[pj]))
            i, j = pi, pj

        aligned.reverse()
        return aligned

    # 이벤트 감지
    @staticmethod
    def _detect_events(aligned) -> List[JosaEvent]:
        events = []
        for t, s in aligned:
            if t is not None and s is None:
                if t.josa is not None:
                    events.append(JosaEvent("DELETION", t.stem, t.josa, None))
            elif t is None and s is not None:
                if s.josa is not None:
                    events.append(JosaEvent("INSERTION", s.stem, None, s.josa))
            else:
                if t.josa is not None and s.josa is None:
                    events.append(JosaEvent("DELETION", t.stem, t.josa, None))
                elif t.josa is not None and s.josa is not None and t.josa != s.josa:
                    events.append(JosaEvent("SUBSTITUTION", t.stem, t.josa, s.josa))
                elif t.josa is None and s.josa is not None:
                    events.append(JosaEvent("INSERTION", t.stem, None, s.josa))
        return events

    def analyze(self, original_text: str, stt_text: str) -> List[JosaEvent]:
        target_pairs = self._extract_stemjosa(original_text)
        stt_pairs = self._extract_stemjosa(stt_text)
        aligned = self._align(target_pairs, stt_pairs)
        return self._detect_events(aligned)

    def get_top_event(self, events: List[JosaEvent]) -> JosaEvent:
        if not events:
            raise ValueError("분석된 조사 오류가 없습니다. 낭독 텍스트를 확인해주세요.")

        counter = Counter(
            (e.kind, e.target_josa, e.stt_josa) for e in events
        )
        top_key, _ = counter.most_common(1)[0]

        for e in events:
            if (e.kind, e.target_josa, e.stt_josa) == top_key:
                return e