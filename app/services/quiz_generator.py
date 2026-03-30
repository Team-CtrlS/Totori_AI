import json, os
from openai import AsyncOpenAI

class QuizGeneratorService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI API KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o"

    async def generate_quiz_words(self, target_word: str, error_pattern: str) -> list[str]:
        system_prompt = (
            "당신은 난독 아동을 위한 언어 치료사입니다.\n"
            "아동의 음운 오류 패턴을 분석하여 해당 패턴을 집중적으로 연습할 수 있는 단어를 추천합니다.\n\n"
            "[오류 패턴 해석 방법]\n"
            "- 'X 탈락': 자음 X가 있어야 할 자리에서 빠뜨리는 오류. 해당 자음이 받침 또는 초성으로 포함된 단어를 추천.\n"
            "- 'X 첨가': 없어야 할 자음 X를 추가로 발음하는 오류. 해당 자음이 없는 단어를 추천.\n"
            "- 'X -> Y 대치': 자음/모음 X를 Y로 잘못 발음하는 오류. X가 포함된 단어를 추천.\n\n"
            "[추천 규칙]\n"
            "1. 반드시 첫 번째 단어는 아이가 틀린 원래 단어 그대로 출력.\n"
            "2. 나머지 3개는 오류 패턴을 연습할 수 있는 단어로, 해당 자음/모음이 실제로 포함된 단어여야 함.\n"
            "3. 4개의 단어는 모두 서로 달라야 함. 중복 절대 금지.\n"
            "4. 모든 단어는 초등학교 1~2학년 수준의 친숙한 단어여야 함.\n"
            "5. 단어는 명사 또는 동사 원형으로만 구성.\n\n"
            "결과는 반드시 아래 JSON 형식으로만 출력:\n"
            '{"words": ["단어1", "단어2", "단어3", "단어4"]}'
        )

        user_prompt = (
            f"아이가 방금 틀린 원래 단어: {target_word}\n"
            f"발견된 오류 패턴: {error_pattern}\n"
            "위 규칙에 따라 '{target_word}'를 포함한 단어 4개를 추천해주세요. 중복 없이."
        )

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        words = result.get("words", [])
        if not words or words[0] != target_word:
            words = [target_word] + [w for w in words if w != target_word][:3]

        return words[:4]