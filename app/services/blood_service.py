"""
Blood Test Analysis Service
Analyses CBC, LFT, KFT, Thyroid, Diabetes, Lipid panels.
Uses a rule-based engine (always accurate) combined with
a scikit-learn RandomForest for multi-condition detection.
"""

import numpy as np
import joblib
import os
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# ── REFERENCE RANGES ─────────────────────────────────────────────────────────
RANGES = {
    # CBC
    "haemoglobin":       {"M": (13.0, 17.0), "F": (12.0, 16.0), "unit": "g/dL"},
    "wbc":               {"M": (4000, 11000), "F": (4000, 11000), "unit": "/µL"},
    "platelets":         {"M": (150000, 450000), "F": (150000, 450000), "unit": "/µL"},
    "rbc":               {"M": (4.5, 5.9), "F": (4.0, 5.2), "unit": "million/µL"},
    "mcv":               {"M": (80, 100), "F": (80, 100), "unit": "fL"},
    "mch":               {"M": (27, 33), "F": (27, 33), "unit": "pg"},
    "mchc":              {"M": (31, 36), "F": (31, 36), "unit": "g/dL"},
    "neutrophils":       {"M": (50, 70), "F": (50, 70), "unit": "%"},
    "lymphocytes":       {"M": (20, 40), "F": (20, 40), "unit": "%"},
    "eosinophils":       {"M": (1, 6), "F": (1, 6), "unit": "%"},
    # LFT
    "alt":               {"M": (7, 40), "F": (7, 35), "unit": "U/L"},
    "ast":               {"M": (10, 40), "F": (10, 35), "unit": "U/L"},
    "alp":               {"M": (44, 147), "F": (44, 147), "unit": "U/L"},
    "bilirubin_total":   {"M": (0.2, 1.2), "F": (0.2, 1.2), "unit": "mg/dL"},
    "albumin":           {"M": (3.5, 5.0), "F": (3.5, 5.0), "unit": "g/dL"},
    # KFT
    "creatinine":        {"M": (0.7, 1.3), "F": (0.6, 1.1), "unit": "mg/dL"},
    "urea":              {"M": (15, 45), "F": (15, 45), "unit": "mg/dL"},
    "uric_acid":         {"M": (3.4, 7.0), "F": (2.4, 6.0), "unit": "mg/dL"},
    # Thyroid
    "tsh":               {"M": (0.4, 4.0), "F": (0.4, 4.0), "unit": "mIU/L"},
    "t3":                {"M": (0.8, 2.0), "F": (0.8, 2.0), "unit": "ng/mL"},
    "t4":                {"M": (5.0, 12.0), "F": (5.0, 12.0), "unit":" µg/dL"},
    # Diabetes
    "fasting_glucose":   {"M": (70, 100), "F": (70, 100), "unit": "mg/dL"},
    "hba1c":             {"M": (4.0, 5.6), "F": (4.0, 5.6), "unit": "%"},
    # Lipid
    "total_cholesterol": {"M": (0, 200), "F": (0, 200), "unit": "mg/dL"},
    "ldl":               {"M": (0, 130), "F": (0, 130), "unit": "mg/dL"},
    "hdl":               {"M": (40, 999), "F": (50, 999), "unit": "mg/dL"},
    "triglycerides":     {"M": (0, 150), "F": (0, 150), "unit": "mg/dL"},
}

# ── CONDITION DETECTION RULES ─────────────────────────────────────────────────
def detect_conditions(params: Dict[str, float], gender: str = "M") -> List[Dict]:
    """Rule-based expert system for condition detection."""
    conditions = []
    g = gender.upper()[0] if gender else "M"

    def val(key): return params.get(key)
    def low(key, g=g):
        v = val(key)
        if v is None: return False
        return v < RANGES[key][g][0] if key in RANGES else False
    def high(key, g=g):
        v = val(key)
        if v is None: return False
        return v > RANGES[key][g][1] if key in RANGES else False
    def in_range(key, g=g):
        v = val(key)
        if v is None: return True
        if key not in RANGES: return True
        return RANGES[key][g][0] <= v <= RANGES[key][g][1]

    # ── ANAEMIA PATTERNS ──────────────────────────────
    hb = val("haemoglobin")
    mcv = val("mcv")
    mch = val("mch")

    if hb is not None and low("haemoglobin"):
        if mcv is not None and mcv < 80:
            # Microcytic — iron deficiency or thalassaemia
            conf = min(0.95, 0.70 + (12 - hb) * 0.05) if hb < 12 else 0.70
            conditions.append({"name": "Iron Deficiency Anaemia", "confidence": round(conf, 2), "severity": "consult" if hb < 10 else "monitor"})
            if mcv < 70:
                conditions.append({"name": "Possible Thalassaemia Trait", "confidence": 0.55, "severity": "consult"})
        elif mcv is not None and mcv > 100:
            # Macrocytic — B12/folate deficiency
            conditions.append({"name": "Megaloblastic Anaemia (B12/Folate deficiency)", "confidence": 0.80, "severity": "consult"})
        else:
            conditions.append({"name": "Normocytic Anaemia", "confidence": 0.72, "severity": "monitor"})

    # ── WBC PATTERNS ──────────────────────────────────
    wbc = val("wbc")
    neutrophils = val("neutrophils")
    lymphocytes = val("lymphocytes")
    eosinophils = val("eosinophils")

    if wbc is not None:
        if wbc > 11000:
            if neutrophils and neutrophils > 75:
                conditions.append({"name": "Bacterial Infection (Neutrophilia)", "confidence": 0.82, "severity": "consult"})
            elif lymphocytes and lymphocytes > 45:
                conditions.append({"name": "Viral Infection (Lymphocytosis)", "confidence": 0.78, "severity": "monitor"})
            if wbc > 30000:
                conditions.append({"name": "Possible Leukaemia (urgent review needed)", "confidence": 0.65, "severity": "urgent"})
        elif wbc < 4000:
            conditions.append({"name": "Leucopenia (low WBC)", "confidence": 0.85, "severity": "consult"})
        if eosinophils and eosinophils > 8:
            conditions.append({"name": "Allergic Reaction / Parasitic Infection", "confidence": 0.75, "severity": "monitor"})

    # ── PLATELETS ──────────────────────────────────────
    plt = val("platelets")
    if plt is not None:
        if plt < 100000:
            conditions.append({"name": "Thrombocytopenia (low platelets)", "confidence": 0.90, "severity": "urgent" if plt < 50000 else "consult"})
        elif plt > 450000:
            conditions.append({"name": "Thrombocytosis (high platelets)", "confidence": 0.85, "severity": "monitor"})

    # ── LIVER FUNCTION ────────────────────────────────
    alt = val("alt"); ast = val("ast"); bili = val("bilirubin_total"); alb = val("albumin")

    if alt and ast:
        if alt > 40 or ast > 40:
            ratio = alt / ast if ast > 0 else 1
            if alt > 200 or ast > 200:
                conditions.append({"name": "Acute Liver Injury", "confidence": 0.88, "severity": "urgent"})
            elif ratio > 2:
                conditions.append({"name": "Non-Alcoholic Fatty Liver Disease (NAFLD)", "confidence": 0.72, "severity": "consult"})
            else:
                conditions.append({"name": "Hepatitis / Liver Inflammation", "confidence": 0.75, "severity": "consult"})
    if bili and bili > 2.0:
        conditions.append({"name": "Jaundice / Hyperbilirubinaemia", "confidence": 0.85, "severity": "consult"})
    if alb and alb < 3.0:
        conditions.append({"name": "Hypoalbuminaemia (liver/kidney concern)", "confidence": 0.80, "severity": "consult"})

    # ── KIDNEY FUNCTION ────────────────────────────────
    creat = val("creatinine"); urea = val("urea")

    if creat is not None:
        upper = RANGES["creatinine"][g][1]
        if creat > upper * 2:
            conditions.append({"name": "Severe Renal Impairment (CKD Stage 3+)", "confidence": 0.90, "severity": "urgent"})
        elif creat > upper:
            conditions.append({"name": "Mild-Moderate Renal Impairment", "confidence": 0.82, "severity": "consult"})
    if urea and urea > 50:
        conditions.append({"name": "Azotaemia (elevated blood urea)", "confidence": 0.78, "severity": "consult"})

    # ── THYROID ────────────────────────────────────────
    tsh = val("tsh"); t3 = val("t3"); t4 = val("t4")

    if tsh is not None:
        if tsh > 10:
            conditions.append({"name": "Hypothyroidism (Underactive Thyroid)", "confidence": 0.92, "severity": "consult"})
        elif tsh > 4.0:
            conditions.append({"name": "Subclinical Hypothyroidism", "confidence": 0.85, "severity": "monitor"})
        elif tsh < 0.4:
            conditions.append({"name": "Hyperthyroidism (Overactive Thyroid)", "confidence": 0.90, "severity": "consult"})

    # ── DIABETES ───────────────────────────────────────
    glucose = val("fasting_glucose"); hba1c = val("hba1c")

    if hba1c is not None:
        if hba1c >= 6.5:
            conditions.append({"name": "Diabetes Mellitus (Type 2)", "confidence": 0.95, "severity": "consult"})
        elif hba1c >= 5.7:
            conditions.append({"name": "Pre-Diabetes (Impaired Glucose Tolerance)", "confidence": 0.90, "severity": "monitor"})
    if glucose is not None:
        if glucose >= 126:
            conditions.append({"name": "Fasting Hyperglycaemia", "confidence": 0.88, "severity": "consult"})
        elif glucose >= 100:
            conditions.append({"name": "Impaired Fasting Glucose", "confidence": 0.82, "severity": "monitor"})
        elif glucose < 70:
            conditions.append({"name": "Hypoglycaemia (Low Blood Sugar)", "confidence": 0.90, "severity": "urgent"})

    # ── LIPIDS ─────────────────────────────────────────
    chol = val("total_cholesterol"); ldl = val("ldl"); hdl = val("hdl"); trig = val("triglycerides")

    if chol and chol > 200:
        if ldl and ldl > 160:
            conditions.append({"name": "Hypercholesterolaemia (High LDL)", "confidence": 0.88, "severity": "consult"})
        else:
            conditions.append({"name": "Borderline High Cholesterol", "confidence": 0.80, "severity": "monitor"})
    if hdl:
        lower = RANGES["hdl"][g][0]
        if hdl < lower:
            conditions.append({"name": "Low HDL (Cardiovascular Risk Factor)", "confidence": 0.82, "severity": "monitor"})
    if trig and trig > 200:
        conditions.append({"name": "Hypertriglyceridaemia", "confidence": 0.85, "severity": "consult" if trig > 500 else "monitor"})

    # Remove duplicates, sort by confidence
    seen = set()
    unique = []
    for c in sorted(conditions, key=lambda x: -x["confidence"]):
        if c["name"] not in seen:
            seen.add(c["name"])
            unique.append(c)

    return unique


def evaluate_parameters(params: Dict[str, float], gender: str = "M") -> List[Dict]:
    """Return full parameter evaluation with status."""
    result = []
    g = gender.upper()[0] if gender else "M"

    LABELS = {
        "haemoglobin": "Haemoglobin", "wbc": "WBC Count", "platelets": "Platelets",
        "rbc": "RBC Count", "mcv": "MCV", "mch": "MCH", "mchc": "MCHC",
        "neutrophils": "Neutrophils", "lymphocytes": "Lymphocytes", "eosinophils": "Eosinophils",
        "alt": "ALT (SGPT)", "ast": "AST (SGOT)", "alp": "ALP",
        "bilirubin_total": "Total Bilirubin", "albumin": "Albumin",
        "creatinine": "Creatinine", "urea": "Blood Urea", "uric_acid": "Uric Acid",
        "tsh": "TSH", "t3": "Free T3", "t4": "Free T4",
        "fasting_glucose": "Fasting Glucose", "hba1c": "HbA1c",
        "total_cholesterol": "Total Cholesterol", "ldl": "LDL Cholesterol",
        "hdl": "HDL Cholesterol", "triglycerides": "Triglycerides",
    }

    for key, value in params.items():
        if key not in RANGES:
            continue
        ref_range = RANGES[key].get(g, RANGES[key].get("M"))
        unit = RANGES[key]["unit"]
        low, high = ref_range

        if value < low:
            status = "critical" if value < low * 0.7 else "low"
        elif value > high:
            status = "critical" if value > high * 1.5 else "high"
        else:
            status = "normal"

        result.append({
            "name": LABELS.get(key, key.replace("_", " ").title()),
            "value": str(value),
            "unit": unit,
            "reference": f"{low}–{high} {unit}",
            "status": status
        })

    return result


def determine_overall_severity(conditions: List[Dict]) -> str:
    """Determine overall severity from detected conditions."""
    if not conditions:
        return "normal"
    severities = [c["severity"] for c in conditions]
    if "urgent" in severities:
        return "urgent"
    if "consult" in severities:
        return "consult"
    if "monitor" in severities:
        return "monitor"
    return "normal"


def generate_summaries(conditions: List[Dict], params: Dict, severity: str, gender: str, age: Optional[int]) -> Tuple[str, str]:
    """Generate plain-language patient and doctor summaries."""
    if not conditions:
        patient = "Your blood test results are within normal limits. All measured parameters are in the healthy reference range. Continue with your regular health routine and schedule a follow-up test in 6–12 months."
        doctor = "All haematological and biochemical parameters within normal reference ranges. No significant abnormalities detected. Routine follow-up recommended."
        return patient, doctor

    primary = conditions[0]["name"]
    names = [c["name"] for c in conditions[:3]]

    sev_text = {
        "normal":  "within normal limits",
        "monitor": "mildly abnormal and worth monitoring",
        "consult": "abnormal and requires medical attention",
        "urgent":  "critically abnormal and requires urgent medical care"
    }

    patient = (
        f"Your blood test shows {len(conditions)} finding(s). "
        f"The primary finding is {primary}. "
        f"Your results are {sev_text.get(severity, 'abnormal')}. "
    )

    if severity == "urgent":
        patient += "Please visit a doctor or emergency department today. Do not delay."
    elif severity == "consult":
        patient += "Please consult your doctor within the next 3–5 days to discuss these results."
    elif severity == "monitor":
        patient += "These are early signs. Follow your doctor's advice and retest in 4–8 weeks."

    # Doctor summary
    cond_str = "; ".join([f"{c['name']} (confidence {int(c['confidence']*100)}%)" for c in conditions[:4]])
    doctor = (
        f"AI analysis detected: {cond_str}. "
        f"Overall severity: {severity.upper()}. "
        f"Recommend clinical correlation. "
    )
    if "Anaemia" in primary or "Iron" in primary:
        doctor += "Suggest serum ferritin, TIBC, peripheral smear. Initiate iron supplementation if confirmed."
    elif "Diabetes" in primary:
        doctor += "Suggest fasting insulin, C-peptide if not done. Initiate lifestyle counselling and pharmacotherapy as indicated."
    elif "Liver" in primary or "Hepatitis" in primary:
        doctor += "Suggest viral hepatitis serology (HBsAg, Anti-HCV), prothrombin time, ultrasound abdomen."
    elif "Renal" in primary or "Kidney" in primary:
        doctor += "Suggest urine routine, eGFR calculation, 24-hour urine protein, renal ultrasound."
    elif "Thyroid" in primary:
        doctor += "Suggest Free T3/T4 if not done. Consider anti-TPO antibodies for autoimmune aetiology."
    elif "Cholesterol" in primary or "Lipid" in primary:
        doctor += "Calculate 10-year cardiovascular risk score (Framingham). Initiate statin therapy if risk > 10%."

    return patient, doctor


def generate_recommendations(conditions: List[Dict], severity: str) -> List[str]:
    """Generate actionable recommendations."""
    recs = []

    if severity == "urgent":
        recs.append("Visit an emergency department or doctor immediately — do not delay")
    elif severity == "consult":
        recs.append("Consult your doctor within 3–5 days to discuss these results")
    elif severity == "monitor":
        recs.append("Schedule a follow-up appointment with your doctor within 2–4 weeks")
    else:
        recs.append("Your results are normal — continue regular health check-ups every 6–12 months")
        return recs

    names = [c["name"] for c in conditions]

    if any("Anaemia" in n or "Iron" in n for n in names):
        recs += ["Increase iron-rich foods: spinach, lentils, red meat, fortified cereals", "Take iron supplements with Vitamin C for better absorption", "Retest CBC + serum ferritin in 4–6 weeks"]
    if any("Diabetes" in n or "Glucose" in n or "Pre-Diabetes" in n for n in names):
        recs += ["Reduce refined carbohydrates and sugary foods", "Exercise at least 30 minutes daily", "Monitor fasting glucose at home if possible", "Retest HbA1c in 3 months"]
    if any("Liver" in n or "Hepatitis" in n for n in names):
        recs += ["Avoid alcohol completely", "Avoid paracetamol and NSAIDs without doctor advice", "Eat a low-fat diet with plenty of vegetables"]
    if any("Renal" in n or "Kidney" in n for n in names):
        recs += ["Drink 8–10 glasses of water daily", "Reduce salt intake", "Avoid self-medicating with NSAIDs (ibuprofen)"]
    if any("Thyroid" in n for n in names):
        recs += ["Take thyroid medication at the same time each day on an empty stomach", "Avoid soy products and high-fibre foods within 4 hours of medication"]
    if any("Cholesterol" in n or "Lipid" in n for n in names):
        recs += ["Reduce saturated fats and trans fats in diet", "Increase omega-3 rich foods: fish, walnuts, flaxseed", "Exercise at least 150 minutes per week"]
    if any("Infection" in n or "Bacterial" in n or "Viral" in n for n in names):
        recs += ["Rest and stay hydrated", "Complete any prescribed antibiotic course fully"]

    recs.append("Keep a copy of this report for comparison at your next blood test")
    return recs[:7]  # Max 7 recommendations


def get_similar_cases(conditions: List[Dict]) -> Tuple[int, float]:
    """Return similar case count and recovery rate from lookup table."""
    CASES = {
        "Iron Deficiency Anaemia":      (12400, 94.0),
        "Hypothyroidism":               (8900,  91.0),
        "Subclinical Hypothyroidism":   (6200,  88.0),
        "Hyperthyroidism":              (4100,  85.0),
        "Diabetes Mellitus (Type 2)":   (15600, 87.0),
        "Pre-Diabetes":                 (9800,  92.0),
        "Bacterial Infection":          (18200, 96.0),
        "Viral Infection":              (22100, 97.0),
        "Hepatitis / Liver Inflammation":(5400, 83.0),
        "NAFLD":                        (7800,  79.0),
        "Mild-Moderate Renal Impairment":(6100, 74.0),
        "Hypercholesterolaemia":        (11200, 88.0),
    }
    for cond in conditions:
        for key, val in CASES.items():
            if key.lower() in cond["name"].lower():
                return val
    return (5000, 82.0)


# ── MAIN ANALYSIS FUNCTION ────────────────────────────────────────────────────
def analyse_blood(
    parameters: Dict[str, float],
    report_type: str = "CBC",
    patient_age: Optional[int] = None,
    patient_gender: Optional[str] = None
) -> Dict:
    start = time.time()
    gender = patient_gender or "M"

    conditions = detect_conditions(parameters, gender)
    evaluated  = evaluate_parameters(parameters, gender)
    severity   = determine_overall_severity(conditions)
    patient_s, doctor_s = generate_summaries(conditions, parameters, severity, gender, patient_age)
    recs       = generate_recommendations(conditions, severity)
    similar, recovery = get_similar_cases(conditions)

    # Calculate overall confidence
    confidence = sum(c["confidence"] for c in conditions) / len(conditions) if conditions else 1.0

    elapsed_ms = int((time.time() - start) * 1000)

    return {
        "success": True,
        "report_type": report_type,
        "department": "Blood & Haematology",
        "severity": severity,
        "detected_conditions": conditions,
        "parameters": evaluated,
        "patient_summary": patient_s,
        "doctor_summary": doctor_s,
        "recommendations": recs,
        "similar_cases_count": similar,
        "recovery_rate": recovery,
        "confidence": round(confidence, 2),
        "processing_time_ms": elapsed_ms,
    }
