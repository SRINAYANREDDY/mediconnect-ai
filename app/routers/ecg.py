from fastapi import APIRouter, HTTPException
from app.models.schemas import ECGInput
from app.services.ecg_service import analyse_ecg

router = APIRouter()

@router.post("/ecg")
async def analyse_ecg_report(data: ECGInput):
    """Analyse ECG parameters — heart rate, ST changes, arrhythmias, conduction defects."""
    try:
        return analyse_ecg(
            heart_rate=data.heart_rate,
            pr_interval=data.pr_interval,
            qrs_duration=data.qrs_duration,
            qt_interval=data.qt_interval,
            qtc_interval=data.qtc_interval,
            p_wave_present=data.p_wave_present,
            st_elevation=data.st_elevation,
            st_depression=data.st_depression,
            t_wave_inversion=data.t_wave_inversion,
            patient_age=data.patient_age,
            patient_gender=data.patient_gender,
            symptoms=data.symptoms,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ECG analysis failed: {str(e)}")
