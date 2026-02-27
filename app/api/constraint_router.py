from fastapi import APIRouter, Depends, HTTPException
from app.schemas.profile_schema import ProfileRequest, ConstraintResponse
from app.services.constraint_generator import ConstraintGeneratorService

router = APIRouter(
    prefix="/api/constraints",
    tags=["Constraint Generator"]
)

_constraint_service_instance = ConstraintGeneratorService()

def get_constraint_service():
    return _constraint_service_instance

@router.post("/generate", response_model=ConstraintResponse)
async def generate_constraint(
    request: ProfileRequest,
    service: ConstraintGeneratorService = Depends(get_constraint_service)
):
    try:
        result = service.generate(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))