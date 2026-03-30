import os
from openai import AsyncOpenAI
from pydantic import ValidationError
from app.schemas.story_schema import StoryResponse

class LLMStoryGeneratorService:
    def __init__(self):
        # 환경 변수에서 API 키 불러오기
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI API KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
        # 비동기 클라이언트 생성
        self.client = AsyncOpenAI(api_key=api_key)

        # 모델 지정
        self.model = "gpt-4o"

    async def generate_story(self, system_prompt: str, user_prompt: str) -> StoryResponse:
        try:
            # OPENAI API 호출
            response = await self.client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            # GPT 결과물 추출
            result_text = response.choices[0].message.content
            
            # 유효성 검증
            parsed_story = StoryResponse.model_validate_json(result_text)
            return parsed_story
        
        except ValidationError as e:
            # GPT가 양식 어겼을 때
            print(f"데이터 형식 불일치 에러 발생: {e}")
            raise ValueError("LLM이 지정한 동화 JSON 형식을 지키지 않았습니다.")
        
        except Exception as e:
            print(f"OpenAI API 통신 중 에러 발생: {e}")
            raise RuntimeError(f"동화 생성 중 서버 통신 오류가 발생했습니다: {str(e)}")