from fastapi import APIRouter, Depends, HTTPException
from app.schemas.story_schema import GenerateStoryRequest, StoryResponse
from app.services.story_orchestrator import StoryOrchestratorService

router = APIRouter(
    prefix="/api/story",
    tags=["Story Generator"]
)

def get_orchestrator():
    return StoryOrchestratorService()

@router.post("/story_generate", response_model=StoryResponse)
async def generate_story(
    request: GenerateStoryRequest,
    orchestrator: StoryOrchestratorService = Depends(get_orchestrator)
):
    try:
        final_story = await orchestrator.run_pipeline(request)
        return final_story
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"동화 생성 중 알 수 없는 오류 발생: {str(e)}")