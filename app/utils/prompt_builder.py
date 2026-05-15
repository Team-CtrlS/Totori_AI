from app.schemas.profile_schema import ConstraintResponse

def build_story_prompt(constraints: ConstraintResponse) -> dict:
    system_prompt = (
        "너는 난독증 아동을 위한 전문 한국어 동화 작가이자 특수교사야.\n"
        "아동의 읽기 수준과 취약한 발음에 맞춰 아주 세심하게 동화를 작성해야 해.\n"
        "반복을 피하고, 장면마다 기승전결이 분명한 이야기를 만들어.\n"
        "\n"
        "[문장 종결 규칙]\n"
        "- 서술 문장은 반드시 자연스러운 해요체로 끝나야 해.\n"
        "- 문장 끝은 반드시 한 번의 '요'로만 끝나야 해.\n"
        "\n"
        "모든 출력은 반드시 JSON 형식으로만 작성하고, 지정된 제약 조건을 완벽하게 지켜줘."
    )

    themes_str = ", ".join(constraints.theme_keywords) if constraints.theme_keywords else "자유 주제"
    phonemes = constraints.focus_phonemes or []
    min_len = getattr(constraints, "min_sentence_len", None)
    scene_count = constraints.scene_count
    sentences_per_scene = constraints.sentences_per_scene

    phoneme_instruction = (
        f"8. 취약 발음 연습: {', '.join(phonemes)} 발음이 포함된 단어를 이야기가 어색하지 않은 선에서 등장시킬 것."
        if phonemes
        else "8. 취약 발음 연습: 별도의 취약 발음 조건이 없으므로, '없음'이라는 단어를 이야기 안에 넣지 말 것."
    )

    user_prompt = f"""
아래의 [제약 조건]을 엄격하게 지켜서 아동 맞춤형 동화를 생성해줘.

[핵심 목표]
- 재미있고 변화가 있는 전개로, 비슷한 내용 반복을 피해야 해.
- 동화에는 기승전결(상황-문제-시도-결과)이 반드시 드러나야 해.

[제약 조건]
1. 대상 연령: {constraints.age_group}
2. 동화 주제(관심사): {themes_str}
3. 어휘 난이도: {constraints.vocab_level}
4. 문장 길이: 한 문장당 절대 {constraints.max_sentence_len}어절을 넘지 말 것.
{"5. 문장 길이(최소): 한 문장당 최소 " + str(min_len) + "어절 이상으로 쓸 것." if min_len else ""}
6. 전체 분량: 총 {constraints.total_pages}개의 문장
7. 장면(Scene) 구성:
- 반드시 정확히 {scene_count}개의 scene만 생성할 것.
- 각 scene에는 반드시 {sentences_per_scene}개의 문장만 넣을 것.
- 모든 scene의 sentences 원소 개수 총합은 반드시 {constraints.total_pages}개여야 할 것.
- "pages" 배열은 실제 페이지 배열이 아니라 scene 배열이다.
- 실제 페이지 수는 모든 scene의 sentences 안에 있는 문장 수 총합으로 계산한다.
{phoneme_instruction}
9. 캐릭터 외형 일관성: 이야기 전체의 주인공은 동일한 인물이어야 하며, 모든 scene의 image_prompt에는 주인공의 핵심 외형 정보를 공통적으로 유지해서 넣을 것.
   - 예: 나이대, 성별, 머리색/머리스타일, 옷차림, 대표 소지품, 전체 화풍
   - scene이 바뀌어도 주인공의 외형, 의상은 바뀌면 안 돼.
   - 바뀌는 것은 배경, 행동, 표정, 구도만 허용돼.

[문장 작성 규칙]
- text 필드에는 반드시 문장 1개만 넣어.
- 서술 문장은 반드시 자연스러운 해요체로 작성해.
- 서술 문장의 마지막 글자는 반드시 '요'여야 해.
- 단, 마지막에 '요'를 두 번 쓰면 안 돼.

[대사 사용 규칙]
- 이야기 전체에서 1~3개의 문장만 대사를 포함해.
- 대사가 포함된 문장도 여전히 "text 하나 = 문장 1개" 규칙을 지켜.
- 대사는 반드시 큰따옴표("")를 사용해.
- 대사만 단독으로 넣지 말고, 필요하면 짧은 서술어와 함께 자연스럽게 작성해.
- 대사 뒤의 서술 문장은 자연스러운 해요체로 끝나야 해.
- 예: "같이 놀자" 하고 준호가 말해요
- 예: 꼬마 공룡이 "나도 친구가 필요해" 하고 말해요

[출력 형식(JSON)]
{{
  "title": "동화 제목",
  "cover_image_prompt": "표지 삽화를 위한 영어 프롬프트",
  "pages": [
    {{
      "page_order": 1,
      "image_prompt": "이 scene의 삽화를 위한 영어 프롬프트",
      "sentences": [
          {{"text": "페이지에 들어갈 문장 1개", "audioS3Key": null, "durationMs": null}},
          {{"text": "페이지에 들어갈 문장 1개", "audioS3Key": null, "durationMs": null}},
          {{"text": "페이지에 들어갈 문장 1개", "audioS3Key": null, "durationMs": null}}
      ]
    }}
  ]
}}

[중요]
- pages 배열의 각 원소는 하나의 scene이야.
- 각 scene에는 반드시 "page_order", "image_prompt", "sentences" 필드가 있어야 해.
- page_order는 scene의 순서를 의미하며 1부터 시작해 순서대로 증가해야 해.
- 각 scene의 image_prompt는 반드시 영어로 작성해.
- 각 scene의 sentences 배열의 각 원소는 반드시 {{"text": "문장", "audioS3Key": null, "durationMs": null}} 형태의 객체여야 해.
- 모든 scene의 sentences 안에 들어 있는 문장의 총합은 반드시 {constraints.total_pages}개여야 해.
- 설명, 해설, 마크다운, 코드블록 없이 JSON만 출력해.
- 모든 scene의 image_prompt에는 반드시 동일한 주인공의 외형 정보가 반복되어야 해.
- 모든 scene의 image_prompt에는 다음 요소를 자연스럽게 포함해:
  1) character age
  2) gender presentation
  3) hair color and hairstyle
  4) outfit
  5) signature item or accessory
  6) illustration style
"""

    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt
    }