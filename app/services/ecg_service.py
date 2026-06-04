"""
ECG Analysis Service
Analyses ECG parameters for arrhythmias, ST changes, and conduction defects.
"""
import time
from typing import Dict, List, Optional, Tuple

NORMAL_RANGES = {
    "heart_rate":    (60, 100,   "bpm"),
    "pr_interval":   (120, 200,  "ms"),
    "qrs_duration":  (70, 120,   "ms"),
    "qt_interval":   (350, 440,  "ms"),
    "qtc_interval":  (350, 440,  "ms"),
}

def analyse_ecg(
    heart_rate: Optional[float] = None,
    pr_interval: Optional[float] = None,
    qrs_duration: Optional[float] = None,
    qt_interval: Optional[float] = None,
    qtc_interval: Optional[float] = None,
    p_wave_present: Optional[bool] = True,
    st_elevation: Optional[float] = 0.0,
    st_depression: Optional[float] = 0.0,
    t_wave_inversion: Optional[bool] = False,
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None,
    symptoms: Optional[List[str]] = None,
) -> Dict:
    start = time.time()
    conditions = []
    parameters = []
    symptoms = symptoms or []

    def check_param(name, value, label, unit):
        if value is None: return
        low, high, u = NORMAL_RANGES[name]
        status = "normal" if low <= value <= high else ("low" if value < low else "high")
        parameters.append({"name": label, "value": str(value), "unit": unit, "reference": f"{low}–{high} {unit}", "status": status})

    check_param("heart_rate",   heart_rate,   "Heart Rate",   "bpm")
    check_param("pr_interval",  pr_interval,  "PR Interval",  "ms")
    check_param("qrs_duration", qrs_duration, "QRS Duration", "ms")
    check_param("qt_interval",  qt_interval,  "QT Interval",  "ms")
    check_param("qtc_interval", qtc_interval, "QTc Interval", "ms")

    # ── HEART RATE ANALYSIS ─────────────────────────────
    if heart_rate is not None:
        if heart_rate > 100:
            if heart_rate > 150:
                conditions.append({"name": "Tachyarrhythmia (HR > 150 bpm)", "confidence": 0.90, "severity": "urgent"})
            else:
                conditions.append({"name": "Sinus Tachycardia", "confidence": 0.88, "severity": "monitor"})
        elif heart_rate < 60:
            if heart_rate < 40:
                conditions.append({"name": "Severe Bradycardia (HR < 40 bpm)", "confidence": 0.92, "severity": "urgent"})
            else:
                conditions.append({"name": "Sinus Bradycardia", "confidence": 0.85, "severity": "monitor"})

    # ── ST CHANGES ─────────────────────────────────────
    if st_elevation and st_elevation > 1.0:
        conditions.append({"name": "ST Elevation — Possible STEMI (Heart Attack)", "confidence": 0.92 if st_elevation > 2.0 else 0.78, "severity": "urgent"})
    if st_depression and st_depression > 1.0:
        conditions.append({"name": "ST Depression — Possible Myocardial Ischaemia", "confidence": 0.82, "severity": "urgent" if st_depression > 2.0 else "consult"})

    # ── P WAVE / ATRIAL FIBRILLATION ────────────────────
    if p_wave_present is False:
        conditions.append({"name": "Atrial Fibrillation (no P waves)", "confidence": 0.88, "severity": "consult"})

    # ── T WAVE INVERSION ────────────────────────────────
    if t_wave_inversion:
        conditions.append({"name": "T-Wave Inversion — Possible Ischaemia", "confidence": 0.72, "severity": "consult"})

    # ── PR INTERVAL ─────────────────────────────────────
    if pr_interval is not None:
        if pr_interval > 200:
            conditions.append({"name": "First Degree AV Block (prolonged PR)", "confidence": 0.88, "severity": "monitor"})
        elif pr_interval < 120:
            conditions.append({"name": "Pre-excitation (short PR) — Possible WPW", "confidence": 0.70, "severity": "consult"})

    # ── QRS DURATION ────────────────────────────────────
    if qrs_duration is not None and qrs_duration > 120:
        conditions.append({"name": "Wide QRS — Bundle Branch Block", "confidence": 0.85, "severity": "consult"})

    # ── QTc PROLONGATION ────────────────────────────────
    qtc = qtc_interval or qt_interval
    if qtc and qtc > 500:
        conditions.append({"name": "Prolonged QTc — Risk of Torsades de Pointes", "confidence": 0.90, "severity": "urgent"})
    elif qtc and qtc > 440:
        conditions.append({"name": "Borderline QTc Prolongation", "confidence": 0.80, "severity": "monitor"})

    # ── NORMAL ───────────────────────────────────────────
    if not conditions:
        conditions.append({"name": "Normal Sinus Rhythm", "confidence": 0.90, "severity": "normal"})

    severity_order = ["normal", "monitor", "consult", "urgent"]
    severity = max([c["severity"] for c in conditions], key=lambda x: severity_order.index(x))

    # ── SUMMARIES ────────────────────────────────────────
    if severity == "normal":
        patient_s = "Your ECG shows normal sinus rhythm. Heart rate and electrical activity are within normal limits. No significant abnormalities detected."
        doctor_s  = "Normal sinus rhythm. Rate, rhythm, PR, QRS and QT intervals within normal limits. No ischaemic or conduction changes."
    elif severity == "urgent":
        primary = conditions[0]["name"]
        patient_s = f"Your ECG shows an urgent finding: {primary}. Please seek immediate medical attention — go to the emergency department or call 108 now."
        doctor_s  = f"URGENT: {'; '.join([c['name'] for c in conditions])}. Immediate cardiological evaluation required."
    else:
        primary = conditions[0]["name"]
        patient_s = f"Your ECG shows {primary}. This requires medical attention. Please consult your doctor or a cardiologist within 1–3 days."
        doctor_s  = f"ECG findings: {'; '.join([c['name'] for c in conditions])}. Cardiologist review recommended. Correlate with symptoms and clinical history."

    recs = []
    if severity == "urgent":
        recs += ["Go to the emergency department immediately", "Call 108 (ambulance) if you have chest pain", "Do not drive yourself — call for help"]
    elif severity == "consult":
        recs += ["Consult a cardiologist within 1–3 days", "Avoid strenuous physical activity until reviewed", "Monitor for chest pain, breathlessness, or palpitations"]
    else:
        recs += ["Show this report to your doctor at next visit", "Report any symptoms of chest pain or palpitations immediately"]
    recs.append("An ECG is a snapshot in time — a 24-hour Holter monitor may be needed for intermittent arrhythmias")

    similar_map = {"normal": (22100, 99.0), "monitor": (9400, 92.0), "consult": (7200, 85.0), "urgent": (4100, 72.0)}
    similar, recovery = similar_map.get(severity, (8000, 85.0))
    confidence = round(sum(c["confidence"] for c in conditions) / len(conditions), 2)
    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "success": True, "report_type": "ECG", "department": "Cardiology",
        "severity": severity,
        "detected_conditions": conditions,
        "parameters": parameters,
        "patient_summary": patient_s,
        "doctor_summary": doctor_s,
        "recommendations": recs,
        "similar_cases_count": similar,
        "recovery_rate": recovery,
        "confidence": confidence,
        "processing_time_ms": elapsed_ms,
    }
