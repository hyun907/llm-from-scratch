"""
FastAPI 추론 서버.

실행:
    uvicorn api.main:app --reload --port 8000
    (프로젝트 루트에서 실행)
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import generate, is_loaded, load_model


# ============================================================
# 서버 시작/종료 시 실행
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 모델 로드
    success = load_model()
    if not success:
        print("[main] 모델 로드 실패. /health에서 model_loaded=false 반환됩니다.")
    yield
    # 종료 시 (필요한 정리 작업)


app = FastAPI(title="LLM From Scratch API", lifespan=lifespan)

# Next.js 개발 서버에서 오는 요청 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ============================================================
# 요청/응답 스키마
# ============================================================

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=100, ge=1, le=500)
    temperature: float = Field(default=0.8, ge=0.1, le=2.0)
    top_k: int = Field(default=40, ge=1, le=200)


class GenerateResponse(BaseModel):
    generated: str
    tokens_generated: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ============================================================
# 라우트
# ============================================================

@app.get("/health", response_model=HealthResponse)
def health():
    from .inference import _device
    return {
        "status": "ok",
        "model_loaded": is_loaded(),
        "device": _device,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    if not is_loaded():
        raise HTTPException(status_code=503, detail="모델이 로드되지 않았습니다.")

    try:
        generated, tokens_generated = generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"generated": generated, "tokens_generated": tokens_generated}
