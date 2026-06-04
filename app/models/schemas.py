from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

# ── ENUMS ─────────────────────────────────────────────
class Severity(str, Enum):
    NORMAL  = "normal"
    MONITOR = "monitor"
    CONSULT = "consult"
    URGENT  = "urgent"

class ReportType(str, Enum):
    BLOOD_CBC      = "CBC"
    BLOOD_LFT      = "LFT"
    BLOOD_KFT      = "KFT"
    BLOOD_THYROID  = "Thyroid"
    BLOOD_DIABETES = "Diabetes"
    BLOOD_LIPID    = "Lipid"
    XRAY_CHEST     = "Chest X-Ray"
    XRAY_BONE      = "Bone X-Ray"
    CT_HEAD        = "CT Head"
    CT_CHEST       = "CT Chest"
    MRI_BRAIN      = "MRI Brain"
    ECG            = "ECG"
    BIOPSY         = "Biopsy"
    PSYCH          = "Psychological Assessment"

# ── SHARED ────────────────────────────────────────────
class DetectedCondition(BaseModel):
    name:       str
    confidence: float = Field(..., ge=0.0, le=1.0)
    severity:   Severity

class Parameter(BaseModel):
    name:      str
    value:     str
    unit:      str
    reference: str
    status:    str  # normal | low | high | critical

class AIResult(BaseModel):
    success:             bool
    report_type:         str
    department:          str
    severity:            Severity
    detected_conditions: List[DetectedCondition]
    parameters:          List[Parameter]
    patient_summary:     str
    doctor_summary:      str
    recommendations:     List[str]
    similar_cases_count: int
    recovery_rate:       float
    confidence:          float
    processing_time_ms:  int

# ── BLOOD TEST ─────────────────────────────────────────
class BloodTestInput(BaseModel):
    report_type: str = "CBC"
    parameters: Dict[str, float] = Field(
        ...,
        example={
            "haemoglobin": 9.2,
            "wbc": 11200,
            "platelets": 240000,
            "rbc": 3.8,
            "mcv": 72.0,
            "mch": 22.0,
            "mchc": 30.5,
            "neutrophils": 65.0,
            "lymphocytes": 28.0,
            "eosinophils": 3.0,
            # LFT
            "alt": 45.0,
            "ast": 38.0,
            "alp": 120.0,
            "bilirubin_total": 1.2,
            "albumin": 3.8,
            # KFT
            "creatinine": 1.0,
            "urea": 32.0,
            "uric_acid": 5.5,
            # Thyroid
            "tsh": 3.2,
            "t3": 1.2,
            "t4": 8.5,
            # Diabetes
            "fasting_glucose": 95.0,
            "hba1c": 5.4,
            # Lipid
            "total_cholesterol": 185.0,
            "ldl": 110.0,
            "hdl": 52.0,
            "triglycerides": 140.0
        }
    )
    patient_age:    Optional[int]   = None
    patient_gender: Optional[str]   = None

# ── X-RAY ──────────────────────────────────────────────
class XRayInput(BaseModel):
    report_type: str = "Chest X-Ray"
    # Image passed as base64 in API route via file upload
    patient_age:    Optional[int] = None
    patient_gender: Optional[str] = None
    clinical_notes: Optional[str] = None

# ── ECG ────────────────────────────────────────────────
class ECGInput(BaseModel):
    heart_rate:       Optional[float] = None
    pr_interval:      Optional[float] = None
    qrs_duration:     Optional[float] = None
    qt_interval:      Optional[float] = None
    qtc_interval:     Optional[float] = None
    p_wave_present:   Optional[bool]  = True
    st_elevation:     Optional[float] = 0.0
    st_depression:    Optional[float] = 0.0
    t_wave_inversion: Optional[bool]  = False
    patient_age:      Optional[int]   = None
    patient_gender:   Optional[str]   = None
    symptoms:         Optional[List[str]] = None

# ── NLP ────────────────────────────────────────────────
class NLPInput(BaseModel):
    report_type: str  # "Biopsy" | "Psychological Assessment" | "Discharge Summary"
    report_text: str  = Field(..., min_length=20, max_length=10000)
    patient_age:    Optional[int] = None
    patient_gender: Optional[str] = None
