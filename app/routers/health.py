from fastapi import APIRouter
import platform, time

router = APIRouter()
START_TIME = time.time()

@router.get("/health")
async def health_check():
    return {
        "success": True,
        "status": "healthy",
        "service": "MediConnect AI",
        "version": "1.0.0",
        "uptime_seconds": round(time.time() - START_TIME),
        "python": platform.python_version(),
        "capabilities": {
            "blood_analysis":       True,
            "xray_analysis":        True,
            "ecg_analysis":         True,
            "nlp_biopsy":           True,
            "nlp_psych_assessment": True,
        }
    }
