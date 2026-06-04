# MediConnect AI Microservice

Python FastAPI service providing AI-powered medical report analysis.

## What it analyses

| Endpoint | Report Types | Method |
|----------|-------------|--------|
| `/analyse/blood` | CBC, LFT, KFT, Thyroid, Diabetes, Lipid panels | Rule-based expert system |
| `/analyse/xray` | Chest X-Ray, Bone X-Ray, CT Scan, MRI | OpenCV + CNN (TF optional) |
| `/analyse/ecg` | ECG parameters | Rule-based cardiology engine |
| `/analyse/nlp` | Biopsy reports, PHQ-9/GAD-7/DASS-21 | Keyword NLP + regex |

## Setup

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy environment file
cp .env.example .env

# 4. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Docs available at:
# http://localhost:8000/docs     (Swagger UI)
# http://localhost:8000/redoc    (ReDoc)
# http://localhost:8000/health   (Health check)
```

## Test the service

```bash
# Run all tests
cd tests
python test_blood.py
python test_ecg.py
python test_nlp.py

# Test via curl — blood test
curl -X POST http://localhost:8000/analyse/blood \
  -H "Content-Type: application/json" \
  -H "X-AI-API-Key: mediconnect_ai_secret_key_change_in_production" \
  -d '{
    "report_type": "CBC",
    "parameters": {
      "haemoglobin": 9.2,
      "wbc": 11200,
      "platelets": 240000,
      "rbc": 3.8,
      "mcv": 72,
      "mch": 22
    },
    "patient_age": 28,
    "patient_gender": "F"
  }'

# Test ECG
curl -X POST http://localhost:8000/analyse/ecg \
  -H "Content-Type: application/json" \
  -H "X-AI-API-Key: mediconnect_ai_secret_key_change_in_production" \
  -d '{"heart_rate": 95, "st_elevation": 2.5, "pr_interval": 160}'

# Test NLP (biopsy)
curl -X POST http://localhost:8000/analyse/nlp \
  -H "Content-Type: application/json" \
  -H "X-AI-API-Key: mediconnect_ai_secret_key_change_in_production" \
  -d '{"report_type": "Biopsy", "report_text": "Invasive ductal carcinoma Grade II. ER positive. Margins clear.", "patient_age": 44, "patient_gender": "F"}'

# Test X-Ray (upload file)
curl -X POST http://localhost:8000/analyse/xray \
  -H "X-AI-API-Key: mediconnect_ai_secret_key_change_in_production" \
  -F "file=@chest_xray.jpg" \
  -F "report_type=Chest X-Ray" \
  -F "patient_age=45"
```

## Connect to Node.js backend

Copy `node_connector.js` to `mediconnect-backend/src/utils/aiConnector.js`

Add to your `.env` in the backend:
```
AI_SERVICE_URL=http://localhost:8000
AI_API_KEY=mediconnect_ai_secret_key_change_in_production
```

Then call from your report controller:
```javascript
const { analyseBlood, analyseXray, analyseECG, analyseNLP } = require('../utils/aiConnector');

// In reportController.js — replace simulateAIAnalysis() with:
const aiResult = await analyseBlood(parameters, reportType, patientAge, patientGender);
if (aiResult.success) {
  await Report.findByIdAndUpdate(report._id, {
    status: 'completed',
    aiAnalysis: { ...aiResult.data, processedAt: new Date() }
  });
}
```

## Architecture

```
mediconnect-backend (Node.js :5000)
        │
        │  HTTP POST + X-AI-API-Key header
        ▼
mediconnect-ai (Python FastAPI :8000)
        │
        ├── /analyse/blood  → blood_service.py  (rule-based expert system)
        ├── /analyse/xray   → xray_service.py   (OpenCV + TF CNN)
        ├── /analyse/ecg    → ecg_service.py    (cardiology rule engine)
        └── /analyse/nlp    → nlp_service.py    (keyword NLP + regex)
```

## Upgrade path (training real models)

For `/analyse/blood` — train a RandomForest on MIMIC-III or PhysioNet datasets:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Save
joblib.dump(model,  'data/models_saved/blood_model.joblib')
joblib.dump(scaler, 'data/models_saved/blood_scaler.joblib')
```

For `/analyse/xray` — fine-tune DenseNet121 on NIH ChestX-ray14 dataset:
```python
import tensorflow as tf

base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False)
model = tf.keras.Sequential([base, tf.keras.layers.GlobalAveragePooling2D(),
                              tf.keras.layers.Dense(15, activation='sigmoid')])
model.save('data/models_saved/xray_model.h5')
```
