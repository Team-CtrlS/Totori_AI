from fastapi import APIRouter, HTTPException
from app.schemas.quiz_schema import QuizRequest, QuizResponse
from app.services.quiz_generator import QuizGeneratorService
from app.services.reading_service import get_reading_service

router = APIRouter(
    prefix="/ai/quiz",
    tags=["Quiz Generator"]
)

_quiz_generator   = QuizGeneratorService()
_reading_service = get_reading_service()

PHONEME_LEVELS = {"L1", "L2", "L3"}
JOSA_LEVELS    = {"L4", "L5", "L6"}

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    # 레벨 검증
    if request.level not in PHONEME_LEVELS | JOSA_LEVELS:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 레벨입니다: {request.level}")

    try:
        errors = await _reading_service.get_errors(request.book_id)

        if request.level in PHONEME_LEVELS:
            pattern, word = _reading_service.get_top_phoneme_error(errors)
            quiz_items = await _quiz_generator.generate_quiz_words(word, pattern)
        else:
            event = _reading_service.get_top_josa_error(errors)
            quiz_items = await _quiz_generator.generate_josa_quiz(event)

        return QuizResponse(quiz_items=quiz_items)

    except ValueError as e:
        # 오류 패턴이 없는 경우 (아이가 완벽하게 읽은 구간)
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")