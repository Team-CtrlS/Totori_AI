from konlpy.tag import Okt
from typing import List, Dict, Any
from app.common.lexicon import FILLER_WORDS, BANNED_KEYWORDS, THEME_HINTS

class InterestRefinerService:
    def __init__(self):
        self.okt = Okt()

    def _is_safe_word(self, word: str) -> bool:
        if word in BANNED_KEYWORDS:
            return False
        
        return True

    def refine(self, text:str) -> List[str]:
        if not text:
            return []
        
        # 명사 추출
        nouns = self.okt.nouns(text)

        refine_keywords = []

        for noun in nouns:
            if len(noun) < 1:
                continue

            if noun in FILLER_WORDS: # 불용어 제거
                continue

            if not self._is_safe_word(noun):
                print(f"유해 키워드 감지 및 제거됨: {noun}")
                continue

            refine_keywords.append(noun)

            refine_keywords = list(set(refine_keywords))

        return refine_keywords