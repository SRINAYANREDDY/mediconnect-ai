from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from dotenv import load_dotenv

load_dotenv()

from app.routers import blood, xray, ecg, nlp, health

app = FastAPI(
    title="MediConnect AI Service",
    description="AI-powered medical report analysis — blood tests, X-rays, ECG, NLP",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS ──────────────────────────────────────────────
origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:5000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API KEY MIDDLEWARE ────────────────────────────────
AI_API_KEY = os.getenv("AI_API_KEY", "mediconnect_ai_secret_key_change_in_production")

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Allow health check and docs without key
    if request.url.path in ["/", "/health", "/docs", "/redoc", "/openapi.json"]:
        return await call_next(request)
    key = request.headers.get("X-AI-API-Key")
    if key != AI_API_KEY:
        return JSONResponse(status_code=401, content={"success": False, "message": "Invalid AI API key"})
    return await call_next(request)

# ── GLOBAL ERROR HANDLER ─────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "message": f"AI service error: {str(exc)}"}
    )

# ── ROUTERS ──────────────────────────────────────────
app.include_router(health.router, tags=["Health"])
app.include_router(blood.router, prefix="/analyse", tags=["Blood Analysis"])
app.include_router(xray.router,  prefix="/analyse", tags=["X-Ray Analysis"])
app.include_router(ecg.router,   prefix="/analyse", tags=["ECG Analysis"])
app.include_router(nlp.router,   prefix="/analyse", tags=["NLP Analysis"])

@app.get("/")
async def root():
    return {
        "service": "MediConnect AI",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "blood":  "POST /analyse/blood",
            "xray":   "POST /analyse/xray",
            "ecg":    "POST /analyse/ecg",
            "nlp":    "POST /analyse/nlp",
            "health": "GET  /health"
        }
    }
