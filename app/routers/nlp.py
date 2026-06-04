from fastapi import APIRouter, HTTPException
from app.models.schemas import NLPInput
from app.services.nlp_service import analyse_nlp

router = APIRouter()

@router.post("/nlp")
async def analyse_text_report(data: NLPInput):
    """Analyse text reports — biopsy, PHQ-9/GAD-7/DASS-21 psych assessments, discharge summaries."""
    try:
        return analyse_nlp(
            report_type=data.report_type,
            report_text=data.report_text,
            patient_age=data.patient_age,
            patient_gender=data.patient_gender,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")
