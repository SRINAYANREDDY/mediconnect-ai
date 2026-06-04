from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from app.services.xray_service import analyse_xray

router = APIRouter()

@router.post("/xray")
async def analyse_xray_report(
    file: UploadFile = File(...),
    report_type: str = Form(default="Chest X-Ray"),
    patient_age: Optional[int] = Form(default=None),
    patient_gender: Optional[str] = Form(default=None),
    clinical_notes: Optional[str] = Form(default=None),
):
    """Analyse X-Ray, CT Scan, or MRI image file (JPG, PNG, DICOM)."""
    image_bytes = await file.read()
    if len(image_bytes) < 500:
        raise HTTPException(status_code=400, detail="File too small — please upload a valid image.")
    try:
        return analyse_xray(
            image_bytes=image_bytes,
            report_type=report_type,
            patient_age=patient_age,
            patient_gender=patient_gender,
            clinical_notes=clinical_notes,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"X-Ray analysis failed: {str(e)}")
