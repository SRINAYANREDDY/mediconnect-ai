import sys; sys.path.insert(0, '..')
from app.services.blood_service import analyse_blood

def test_normal_cbc():
    r = analyse_blood({"haemoglobin":14.0,"wbc":7500,"platelets":250000,"rbc":5.0,"mcv":88,"mch":29}, "CBC", 35, "M")
    assert r["success"] and r["severity"] == "normal"; print("✅ Normal CBC passed")

def test_iron_deficiency():
    r = analyse_blood({"haemoglobin":8.5,"wbc":8000,"platelets":320000,"rbc":3.5,"mcv":68,"mch":20}, "CBC", 28, "F")
    assert r["success"] and r["severity"] in ["consult","urgent"]
    assert any("Anaemia" in c["name"] for c in r["detected_conditions"]); print("✅ Anaemia passed")

def test_diabetes():
    r = analyse_blood({"hba1c":7.2,"fasting_glucose":142}, "Diabetes", 45, "M")
    assert any("Diabetes" in c["name"] for c in r["detected_conditions"]); print("✅ Diabetes passed")

def test_hypothyroid():
    r = analyse_blood({"tsh":8.5,"t3":0.7,"t4":4.2}, "Thyroid", 40, "F")
    assert any("Hypothyroid" in c["name"] for c in r["detected_conditions"]); print("✅ Hypothyroid passed")

def test_liver():
    r = analyse_blood({"alt":95,"ast":88,"bilirubin_total":1.8,"albumin":3.2}, "LFT", 52, "M")
    assert len(r["detected_conditions"]) > 0; print("✅ Liver passed")

def test_kidney():
    r = analyse_blood({"creatinine":2.8,"urea":62}, "KFT", 60, "M")
    assert any("Renal" in c["name"] or "Kidney" in c["name"] for c in r["detected_conditions"]); print("✅ Kidney passed")

if __name__ == "__main__":
    test_normal_cbc(); test_iron_deficiency(); test_diabetes()
    test_hypothyroid(); test_liver(); test_kidney()
    print("\n🎉 All blood tests passed!")
