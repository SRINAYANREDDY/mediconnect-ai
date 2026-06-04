import sys; sys.path.insert(0, '..')
from app.services.ecg_service import analyse_ecg

def test_normal():
    r = analyse_ecg(heart_rate=72,pr_interval=160,qrs_duration=90,qt_interval=400,qtc_interval=410)
    assert r["success"] and r["severity"]=="normal"; print("✅ Normal ECG passed")

def test_stemi():
    r = analyse_ecg(heart_rate=95,st_elevation=2.5)
    assert r["severity"]=="urgent" and any("STEMI" in c["name"] or "Elevation" in c["name"] for c in r["detected_conditions"]); print("✅ STEMI passed")

def test_afib():
    r = analyse_ecg(heart_rate=110,p_wave_present=False)
    assert any("Fibrillation" in c["name"] for c in r["detected_conditions"]); print("✅ AFib passed")

def test_bradycardia():
    r = analyse_ecg(heart_rate=38)
    assert r["severity"]=="urgent"; print("✅ Bradycardia passed")

if __name__ == "__main__":
    test_normal(); test_stemi(); test_afib(); test_bradycardia()
    print("\n🎉 All ECG tests passed!")
