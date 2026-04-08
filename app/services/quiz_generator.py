import json, os
from openai import AsyncOpenAI
from app.services.josa_analyzer import JosaEvent

class QuizGeneratorService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI API KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o-mini"

    def _dedupe(self, items: list[str], first: str) -> list[str]:
        seen, result = set(), []
        for item in [first] + [x for x in items if x != first]:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result[:4]
    
    async def _call_gpt(self, system_prompt: str, user_prompt: str, key: str) -> list[str]:
        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content).get(key, [])
    
    # 음운 퀴즈
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
            "4. 모든 단어는 초등학교-중학교 수준의 친숙한 단어여야 함. 실생활에서 사용하는 단어여야 함.\n"
            "5. 단어는 명사 또는 동사 원형으로만 구성.\n\n"
            "결과는 반드시 아래 JSON 형식으로만 출력:\n"
            '{"words": ["단어1", "단어2", "단어3", "단어4"]}'
        )

        user_prompt = (
            f"아이가 방금 틀린 원래 단어: {target_word}\n"
            f"오류 패턴: {error_pattern}\n"
            f"위 규칙에 따라 '{target_word}'를 포함한 단어 4개를 중복 없이 추천해주세요."
        )
        words = await self._call_gpt(system_prompt, user_prompt, "words")
        return self._dedupe(words, target_word)
    
    # 조사 퀴즈
    async def generate_josa_quiz(self, event: JosaEvent) -> list[str]:
        if event.kind == "DELETION":
            error_desc = f"조사 '{event.target_josa}'를 빠뜨리는 오류 (예: '{event.stem}' 뒤에 {event.target_josa}'가 와야 함)"
            practice_josa = event.target_josa
        elif event.kind == "SUBSTITUTION":
            error_desc = f"조사 '{event.target_josa}'를 '{event.stt_josa}'로 잘못 읽는 오류 ('{event.stem}{event.target_josa}' -> '{event.stem}{event.stt_josa}')"
            practice_josa = event.target_josa
        else:
            error_desc = f"없어야 할 조사 '{event.stt_josa}'를 추가로 붙이는 오류"
            practice_josa = event.stt_josa

        system_prompt = (
            "당신은 난독 아동을 위한 언어 치료사입니다.\n"
            "아동이 조사를 잘못 읽는 오류 패턴을 바탕으로 연습용 퀴즈 문장을 만들어야 합니다.\n\n"
            "[문장 생성 규칙]\n"
            "1. 반드시 정확히 4개의 문장을 생성.\n"
            "2. 각 문장은 반드시 두 어절(체언+조사 / 용언)로만 구성. 예: '나비가 난다.' '음식을 먹는다.'\n"
            "3. 4개 문장 모두 연습할 조사가 반드시 포함되어야 함.\n"
            "4. 문장은 서로 중복되지 않아야 함.\n"
            "5. 초등학교-중학교 수준의 쉬운 어휘만 사용.\n"
            "6. 각 문장은 마침표로 끝냄.\n\n"
            "결과는 반드시 아래 JSON 형식으로만 출력:\n"
            '{"sentences": ["문장1", "문장2", "문장3", "문장4"]}'
        )

        user_prompt = (
            f"오류: {error_desc}\n"
            f"연습 조사: '{practice_josa}'\n"
            f"예) 조사 '가': ['나비가 난다.', '꽃이 핀다.', '새가 난다.', '별이 빛난다.']\n"
            f"예) 조사 '을': ['음식을 먹는다.', '책을 읽는다.', '공을 찬다.', '물을 마신다.']\n"
            f"조사 '{practice_josa}'가 들어간 두 어절 문장 4개를 중복 없이 만들어줘."
        )
        
        sentences = await self._call_gpt(system_prompt, user_prompt, "sentences")
        seen, result = set(), []
        for s in sentences:
            if s not in seen:
                seen.add(s); result.append(s)
        return result[:4]