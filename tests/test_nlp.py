import sys; sys.path.insert(0, '..')
from app.services.nlp_service import analyse_nlp

def test_benign():
    r = analyse_nlp("Biopsy","Benign breast tissue. No evidence of malignancy. Fibroadenoma. Margins clear.",32,"F")
    assert r["success"] and r["severity"]=="normal"; print("✅ Benign biopsy passed")

def test_cancer():
    r = analyse_nlp("Biopsy","Invasive ductal carcinoma Grade III. ER positive. Lymphovascular invasion present. Margins positive.",44,"F")
    assert r["severity"] in ["consult","urgent"]; print("✅ Cancer biopsy passed")

def test_phq9_severe():
    r = analyse_nlp("Psychological Assessment","PHQ-9: 18. GAD-7: 14. Persistent hopelessness.",29,"M")
    assert r["severity"] in ["consult","urgent"]; print("✅ Severe depression passed")

def test_phq9_normal():
    r = analyse_nlp("Psychological Assessment","PHQ-9: 2. GAD-7: 1. Stable mood.",35,"F")
    assert r["severity"]=="normal"; print("✅ Normal psych passed")

if __name__ == "__main__":
    test_benign(); test_cancer(); test_phq9_severe(); test_phq9_normal()
    print("\n🎉 All NLP tests passed!")
