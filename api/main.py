"""
FastAPI 추론 서버.

실행:
    uvicorn api.main:app --reload --port 8000
    (프로젝트 루트에서 실행)
"""

import os
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

# 허용할 origin을 환경변수로 관리 (쉼표로 구분)
# 예: ALLOWED_ORIGINS="http://localhost:3000,https://myapp.com"
_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
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
    repetition_penalty: float = Field(default=1.3, ge=1.0, le=3.0)


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

    # 프롬프트가 모델의 max_seq_len을 초과하면 서버 오류 대신 클라이언트 오류로 처리
    from .inference import _model, _tokenizer
    if _model and _tokenizer:
        token_len = len(_tokenizer.encode(req.prompt))
        if token_len >= _model.config.max_seq_len:
            raise HTTPException(
                status_code=422,
                detail=f"프롬프트가 너무 깁니다. ({token_len} 토큰, 최대 {_model.config.max_seq_len - 1})"
            )

    try:
        generated, tokens_generated = generate(
            prompt=req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"generated": generated, "tokens_generated": tokens_generated}
