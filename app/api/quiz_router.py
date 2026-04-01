from fastapi import APIRouter, HTTPException
from app.schemas.quiz_schema import QuizRequest, QuizResponse, PhonemeErrorDetail, JosaErrorDetail, JosaQuizRequest, JosaQuizResponse
from app.services.phoneme_analyzer import PhonemeAnalyzerService
from app.services.quiz_generator import QuizGeneratorService
from app.services.josa_analyzer import JosaAnalyzerService
from app.services.josa_quiz_generator import JosaQuizGeneratorService

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

@router.post("/josa", response_model=JosaQuizResponse)
async def generate_josa_quiz(request: JosaQuizRequest):
    if request.level not in ("L4", "L5", "L6"):
        raise HTTPException(status_code=400, detail=f"조사 오류 퀴즈는 L4~L6 레벨만 지원합니다. 요청 레벨: {request.level}")
    
    try:
        analyzer = JosaAnalyzerService()
        events = analyzer.analyze(request.original_text, request.stt_text)
        top_event = analyzer.get_top_event(events)

        generator = JosaQuizGeneratorService()
        quiz_sentences = await generator.generate_quiz_sentences(top_event)

        return JosaQuizResponse(
            error_detail = JosaErrorDetail(
                kind = top_event.kind,
                stem = top_event.stem,
                target_josa = top_event.target_josa,
                stt_josa = top_event.stt_josa,
            ),
            quiz_sentences = quiz_sentences,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"퀴즈 생성 중 오류 발생: {str(e)}")