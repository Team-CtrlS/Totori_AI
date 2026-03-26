from app.common.level_consts import LEVEL_CONSTRAINTS
from app.schemas.profile_schema import ProfileRequest, ConstraintResponse

class ConstraintGeneratorService:
    def generate(self, request: ProfileRequest) -> ConstraintResponse:
        # 레벨 유효성 검사
        if request.level not in LEVEL_CONSTRAINTS:
            raise ValueError(f"지원하지 않는 레벨입니다: {request.level}")
        
        base_consts = LEVEL_CONSTRAINTS[request.level]

        # wcpm 조정
        if request.recent_wcpm is not None:
            final_wcpm = request.recent_wcpm
        else:
            final_wcpm = base_consts["target_wcpm"]

        # 취약 발음 처리
        final_phonemes = request.weak_phonemes if request.weak_phonemes else []

        # 결과 반환
        return ConstraintResponse(
            level_name=request.level,
            age_group=base_consts["age_group"],
            max_sentence_len=base_consts["max_sentence_len"],
            vocab_level=base_consts["vocab_level"],
            total_pages=base_consts["total_pages"],
            theme_keywords=request.interests,
            focus_phonemes=final_phonemes or [],
            scene_count=base_consts["scene_count"],
            sentences_per_scene=base_consts["sentences_per_scene"],
            adjusted_target_wcpm=final_wcpm
        )