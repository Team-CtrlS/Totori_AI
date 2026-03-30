from fastapi import APIRouter, HTTPException
from app.schemas.quiz_schema import QuizRequest, QuizResponse, PhonemeErrorDetail
from app.services.phoneme_analyzer import PhonemeAnalyzerService
from app.services.quiz_generator import QuizGeneratorService

router = APIRouter(
    prefix="/api/quiz",
    tags=["Quiz Generator"]
)

@router.post("/phoneme", response_model=QuizResponse)
async def generate_phoneme_quiz(request: QuizRequest):
    # 레벨 검증
    if request.level not in ("L1", "L2", "L3"):
        raise HTTPException(
            status_code=400,
            detail=f"음운 오류 퀴즈는 L1~L3 레벨만 지원합니다. 요청 레벨: {request.level}"
        )
    
    try:
        # 음운 오류 분석
        analyzer = PhonemeAnalyzerService()
        error_reports, error_words = analyzer.analyze(request.original_text, request.stt_text)

        # 최빈 오류 추출
        error_pattern, target_word, error_count = analyzer.get_top_error(error_reports, error_words)

        # 퀴즈 단어 생성
        quiz_generator = QuizGeneratorService()
        quiz_words = await quiz_generator.generate_quiz_words(target_word, error_pattern)

        return QuizResponse(
            error_detail=PhonemeErrorDetail(
                error_pattern=error_pattern,
                target_word=target_word,
                error_count=error_count,
            ),
        quiz_words=quiz_words,
        )
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")
