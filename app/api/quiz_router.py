from fastapi import APIRouter, HTTPException
from app.schemas.quiz_schema import QuizRequest, QuizResponse, PhonemeErrorDetail, JosaErrorDetail
from app.services.phoneme_analyzer import PhonemeAnalyzerService
from app.services.josa_analyzer import JosaAnalyzerService
from app.services.quiz_generator import QuizGeneratorService

router = APIRouter(
    prefix="/ai/quiz",
    tags=["Quiz Generator"]
)

_phoneme_analyzer = PhonemeAnalyzerService()
_josa_analyzer    = JosaAnalyzerService()
_quiz_generator   = QuizGeneratorService()

PHONEME_LEVELS = {"L1", "L2", "L3"}
JOSA_LEVELS    = {"L4", "L5", "L6"}

@router.post("/generate", response_model=QuizResponse)
async def generate_quiz(request: QuizRequest):
    # 레벨 검증
    if request.level not in PHONEME_LEVELS | JOSA_LEVELS:
        raise HTTPException(status_code=400, detail=f"유효하지 않은 레벨입니다: {request.level}")
    
    try:
        original_text = " ".join(request.original_text)
        stt_text = " ".join(request.stt_text)

        if request.level in PHONEME_LEVELS:
            reports, words = _phoneme_analyzer.analyze(original_text, stt_text)
            pattern, target_word, count = _phoneme_analyzer.get_top_error(reports, words)
            quiz_items = await _quiz_generator.generate_quiz_words(target_word, pattern)
        else:
            events = _josa_analyzer.analyze(original_text, stt_text)
            top_event = _josa_analyzer.get_top_event(events)
            quiz_items = await _quiz_generator.generate_josa_quiz(top_event)
        return QuizResponse(quiz_items=quiz_items)
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")
