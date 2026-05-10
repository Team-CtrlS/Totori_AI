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

# 퀴즈 정답 비교용 텍스트 정규화
def normalize_for_quiz(text: str) -> str:
    if not text:
        return ""
    
    text = re.sub(r"[^가-힣a-zA-Z0-9]", "", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()