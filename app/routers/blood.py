from fastapi import APIRouter, HTTPException
from app.models.schemas import BloodTestInput
from app.services.blood_service import analyse_blood

router = APIRouter()

@router.post("/blood")
async def analyse_blood_report(data: BloodTestInput):
    """Analyse blood test parameters — CBC, LFT, KFT, Thyroid, Diabetes, Lipid."""
    try:
        return analyse_blood(
            parameters=data.parameters,
            report_type=data.report_type,
            patient_age=data.patient_age,
            patient_gender=data.patient_gender,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Blood analysis failed: {str(e)}")
