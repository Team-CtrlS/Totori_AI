import json, os
from openai import AsyncOpenAI
from app.services.josa_analyzer import JosaEvent

class JosaQuizGeneratorService:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI API KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = "gpt-4o"

    async def generate_quiz_sentences(self, event: JosaEvent) -> list[str]:
        # 오류 유형별 설명 구성
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
             f"오류 정보: {error_desc}\n"
            f"연습할 조사: '{practice_josa}'\n\n"
            f"예시) 조사가 '가'라면: ['나비가 난다.', '꽃이 핀다.', '새가 난다.', '별이 빛난다.']\n"
            f"예시) 조사가 '을'이라면: ['음식을 먹는다.', '책을 읽는다.', '공을 찬다.', '물을 마신다.']\n\n"
            f"위 규칙에 따라 조사 '{practice_josa}'를 포함한 두 어절 문장 4개를 만들어주세요."
        )

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)
        sentences = result.get("sentences", [])

        seen = set()
        deduped = []
        for s in sentences:
            if s not in seen:
                seen.add(s)
                deduped.append(s)
        
        return deduped[:4]