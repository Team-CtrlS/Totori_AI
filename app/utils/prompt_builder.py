from app.schemas.profile_schema import ConstraintResponse

def build_story_prompt(constraints: ConstraintResponse) -> dict:
    # 역할과 출력 형식 고정
    system_prompt = (
        "너는 난독증 아동을 위한 전문 한국어 동화 작가이자 특수 교사야.\n"
        "아동의 읽기 수준과 취약한 발음에 맞춰 아주 세심하게 동화를 작성해야 해.\n"
        "모든 출력은 반드시 JSON 형식으로만 작성하고, 지정된 제약 조건을 완벽하게 지켜줘."
    )

    themes_str = ", ".join(constraints.theme_keywords) if constraints.theme_keywords else "자유 주제"
    phonemes_str = ", ".join(constraints.focus_phonemes) if constraints.focus_phonemes else "없음"

    # 제약값 넣어서 지시
    user_prompt = f"""
아래의 [제약 조건]을 엄격하게 지켜서 아동 맞춤형 동화를 생성해줘.

[제약 조건]
1. 대상 연령: {constraints.age_group}
2. 동화 주제(관심사): {themes_str}
3. 어휘 난이도: {constraints.vocab_level}
4. 문장 길이: 한 페이지당 무조건 1개의 문장만 작성할 것. (한 문장당 절대 {constraints.max_sentence_len}어절을 넘지 말 것)
5. 전체 분량: 총 {constraints.total_pages}페이지 (즉, 총 {constraints.total_pages}개의 문장)
6. 장면(Scene) 구성: 3~4개의 페이지(문장)를 묶어서 하나의 Scene으로 구성할 것. 각 Scene마다 삽화를 그리기 위한 영어 프롬프트를 1개씩만 생성할 것.
7. 취약 발음 연습: '{phonemes_str}' 발음이 포함된 단어를 이야기가 어색하지 않은 선에서 자주 등장시킬 것.

[출력 형식(JSON)]
{{
    "title": "동화 제목",
    "scenes": [
        {{
            "scene_number": 1,
            "image_prompt": "이 scene의 삽화를 위한 영어 프롬프트",
            "pages": [
                {{
                    "page_number": 1,
                    "text": "페이지에 들어갈 문장 1개"
                }},
                {{
                    "page_number": 2,
                    "text": "페이지에 들어갈 문장 1개"
                }},
                {{
                    "page_number": 3,
                    "text": "페이지에 들어갈 문장 1개"
                }}
            ]
        }},
        ...
    ]
}}
"""
    
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }