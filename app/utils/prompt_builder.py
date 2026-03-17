from app.schemas.profile_schema import ConstraintResponse

def build_story_prompt(constraints: ConstraintResponse) -> dict:
    # 역할과 출력 형식 고정
    system_prompt = (
        "너는 난독증 아동을 위한 전문 한국어 동화 작가이자 특수교사야.\n"
        "아동의 읽기 수준과 취약한 발음에 맞춰 아주 세심하게 동화를 작성해야 해.\n"
        "반복을 피하고, 장면마다 기승전결이 분명한 이야기를 만들어.\n"
        "모든 문장은 반드시 '-요'로 끝나. ('다/합니다/해' 금지)\n"
        "모든 출력은 반드시 JSON 형식으로만 작성하고, 지정된 제약 조건을 완벽하게 지켜줘."
    )

    themes_str = ", ".join(constraints.theme_keywords) if constraints.theme_keywords else "자유 주제"
    phonemes_str = ", ".join(constraints.focus_phonemes) if constraints.focus_phonemes else "없음"
    min_len = getattr(constraints, "min_sentence_len", None)

    # 제약값 넣어서 지시
    user_prompt = f"""
아래의 [제약 조건]을 엄격하게 지켜서 아동 맞춤형 동화를 생성해줘.

[핵심 목표]
- 재미있고 변화가 있는 전개로, 비슷한 내용 반복을 피해야 해.
- 동화에는 기승전결(상황-문제-시도-결과)이 반드시 드러나야 해.
- 모든 문장은 반드시 '-요'로 끝나야 해.

[제약 조건]
1. 대상 연령: {constraints.age_group}
2. 동화 주제(관심사): {themes_str}
3. 어휘 난이도: {constraints.vocab_level}
4. 문장 길이: 한 페이지당 무조건 1개의 문장만 작성할 것. (한 문장당 절대 {constraints.max_sentence_len}어절을 넘지 말 것)
{"5. 문장 길이(최소): 한 문장당 최소 " + str(min_len) + "어절 이상으로 써." if min_len else ""}
6. 전체 분량: 총 {constraints.total_pages}페이지 (즉, 총 {constraints.total_pages}개의 문장)
7. 장면(Scene) 구성: 3~4개의 페이지(문장)를 묶어서 하나의 Scene으로 구성할 것. 각 Scene마다 삽화를 그리기 위한 구체적인 영어 프롬프트를 1개씩만 생성할 것.
8. 취약 발음 연습: '{phonemes_str}' 발음이 포함된 단어를 이야기가 어색하지 않은 선에서 자주 등장시킬 것.

[대사 사용 규칙]
- 이야기 전체에서 1~3개의 페이지만 대사를 포함해.
- 대사가 포함된 페이지도 여전히 "1페이지 = 1문장" 규칙을 지켜.
- 대사는 반드시 큰따옴표("")를 사용해.
- 대사는 반드시 '-요'로 끝나지 않아도 돼.
- 대사는 인물의 감정이나 문제 해결에 기여해야 해.

[출력 형식(JSON)]
{{
  "title": "동화 제목",
  "pages": [
    {{
        "pageOrder": 1,
        "imagePrompt": "A cute puppy looking at a sparkling star in the night sky, children's book illustration, warm and soft style",
        "sentences": ["토토가 반짝이는 별을 봐요."]
    }},
    {{
        "pageOrder": 2,
        "imagePrompt": "A cute puppy reaching toward a bright star, children's book illustration, dreamy and warm tone",
        "sentences": ["토토는 별을 잡고 싶어요."]
    }}, ...
  ], ...
}}

[중요]
- pages 배열 길이는 반드시 {constraints.total_pages}개여야 해.
- 각 page의 sentences 배열에는 반드시 문장 1개만 넣어.
- 설명, 해설, 마크다운 없이 JSON만 출력해.

"""
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }