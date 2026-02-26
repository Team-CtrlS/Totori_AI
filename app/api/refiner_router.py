from fastapi import APIRouter, Depends
from app.schemas.refiner_schema import InterestRequest, InterestResponse
from app.services.interest_refiner import InterestRefinerService

router = APIRouter(
    prefix="/api/refiner",
    tags=["Interest Refiner"]
)

_refiner_service_instance = InterestRefinerService()

def get_refiner_service():
    return _refiner_service_instance

@router.post("/extract", response_model=InterestResponse)
async def extract_interests(
    request: InterestRequest,
    service: InterestRefinerService = Depends(get_refiner_service)
):
    # 정제된 키워드 받기
    keywords = service.refine(request.text)

    # 결과 비었는지 확인
    is_empty = len(keywords) == 0

    # InterestResponse에 맞춰 데이터 변환
    return InterestResponse(
        original_text=request.text,
        keywords=keywords,
        is_empty=is_empty
    )