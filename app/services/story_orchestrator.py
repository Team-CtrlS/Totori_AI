from app.schemas.story_schema import GenerateStoryRequest, StoryResponse
from app.schemas.profile_schema import ProfileRequest
from app.services.interest_refiner import InterestRefinerService
from app.services.constraint_generator import ConstraintGeneratorService
from app.services.story_generator import LLMStoryGeneratorService
from app.utils.prompt_builder import build_story_prompt

class StoryOrchestratorService:
    def __init__(self):
        self.refiner_service = InterestRefinerService()
        self.constraint_service = ConstraintGeneratorService()
        self.llm_service = LLMStoryGeneratorService()

    async def run_pipeline(self, request: GenerateStoryRequest) -> StoryResponse:
        # STT 관심사 정제
        self.refined_keywords = self.refiner_service.refine(request.stt_text)

        # 제약값 생성
        profile_req = ProfileRequest(
            level=request.level,
            interests=self.refined_keywords,
            recent_wcpm=request.recent_wcpm,
            weak_phonemes=request.weak_phonemes
        )
        constraints = self.constraint_service.generate(profile_req)

        # 프롬프트 빌더
        prompts = build_story_prompt(constraints)

        # LLM 동화 생성
        final_story = await self.llm_service.generate_story(
            system_prompt=prompts["system_prompt"],
            user_prompt=prompts["user_prompt"]
        )

        return final_story