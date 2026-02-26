import re
from ..common.lexicon import FILLER_WORDS

def basic_clean(text: str) -> str:
    if not text:
        return []
    
    # 특수문자 제거
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    # 공백 기준 분리
    words = text.split()

    # 불용어 제거
    cleaned_words = [
        words for word in words
        if word not in FILLER_WORDS
    ]

    return " ".join(cleaned_words)