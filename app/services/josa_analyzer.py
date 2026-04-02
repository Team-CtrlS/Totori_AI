import re, os
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import Counter
from konlpy.tag import Mecab
from app.utils.alignment_utils import levenshtein

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

    def _extract(self, text: str) -> List[StemJosa]:
        tokens = self.tagger.pos(text)
        pairs, i = [], 0
        while i < len(tokens):
            morph, tag = tokens[i]
            if tag.startswith("NN") or tag in ("NP", "NR"):
                josa = None
                if i + 1 < len(tokens) and tokens[i + 1][1].startswith("J"):
                    josa = tokens[i + 1][0]
                    i += 2
                else:
                    i += 1
                pairs.append(StemJosa(stem=morph, josa=josa))
            else:
                i += 1
        return pairs
    
    def _align(self, target: List[StemJosa], stt: List[StemJosa]):
        lenT, lenS = len(target), len(stt)

        def sub_cost(a, b):
            return min(3, levenshtein(list(a.stem), list(b.stem)))

        dp = [[0] * (lenS + 1) for _ in range(lenT + 1)]
        bt = [[None] * (lenS + 1) for _ in range(lenT + 1)]
        for i in range(1, lenT + 1):
            dp[i][0] = i; bt[i][0] = ("DEL", i - 1, 0)
        for j in range(1, lenS + 1):
            dp[0][j] = j; bt[0][j] = ("INS", 0, j - 1)

        for i in range(1, lenT + 1):
            for j in range(1, lenS + 1):
                c_sub = dp[i-1][j-1] + sub_cost(target[i-1], stt[j-1])
                c_del = dp[i-1][j] + 1
                c_ins = dp[i][j-1] + 1
                best = min(c_sub, c_del, c_ins)
                dp[i][j] = best
                bt[i][j] = ("SUB", i-1, j-1) if best == c_sub else \
                            ("DEL", i-1, j) if best == c_del else \
                            ("INS", i, j-1)
            
        aligned, i, j = [], lenT, lenS
        while i > 0 or j > 0:
            op, pi, pj = bt[i][j]
            aligned.append((target[pi] if op != "INS" else None, stt[pj] if op != "DEL" else None))
            i, j = pi, pj
        return aligned[::-1]
    
    def _detect(self, aligned) -> List[JosaEvent]:
        events = []
        for t, s in aligned:
            if t and not s:
                if t.josa: events.append(JosaEvent("DELETION", t.stem, t.josa, None))
            elif not t and s:
                if s.josa: events.append(JosaEvent("INSERTION", s.stem, None, s.josa))
            else:
                if t.josa and not s.josa:
                    events.append(JosaEvent("DELETION", t.stem, t.josa, None))
                elif t.josa and s.josa and t.josa != s.josa:
                    events.append(JosaEvent("SUBSTITUTION", t.stem, t.josa, s.josa))
                elif not t.josa and s.josa:
                    events.append(JosaEvent("INSERTION", t.stem, None, s.josa))
        return events
    
    def analyze(self, original_text: str, stt_text: str) -> List[JosaEvent]:
        return self._detect(self._align(self._extract(original_text), self._extract(stt_text)))

    def get_top_event(self, events: List[JosaEvent]) -> JosaEvent:
        if not events:
            raise ValueError("분석된 조사 오류가 없습니다.")
        top_key = Counter((e.kind, e.target_josa, e.stt_josa) for e in events).most_common(1)[0][0]
        return next(e for e in events if (e.kind, e.target_josa, e.stt_josa) == top_key)