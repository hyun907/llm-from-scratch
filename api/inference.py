"""
모델 로드 + 텍스트 생성.

서버 시작 시 한 번만 로드하고, 이후 요청은 로드된 모델을 재사용합니다.
"""

import sys
from pathlib import Path

import torch

# 프로젝트 루트를 sys.path에 추가 (model/ 패키지 임포트용)
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from model import GPT, GPTConfig
from model.tokenizer import BPETokenizer


# ============================================================
# 전역 상태 — 서버 시작 시 한 번만 초기화
# ============================================================

_model: GPT | None = None
_tokenizer: BPETokenizer | None = None
_device: str = "cpu"


def load_model(
    checkpoint_path: str | Path = ROOT / "checkpoints" / "model.pt",
    tokenizer_path: str | Path = ROOT / "data" / "tokenizer.json",
) -> bool:
    """
    체크포인트와 토크나이저를 로드합니다.
    성공하면 True, 실패하면 False를 반환합니다.
    """
    global _model, _tokenizer, _device

    checkpoint_path = Path(checkpoint_path)
    tokenizer_path = Path(tokenizer_path)

    if not checkpoint_path.exists():
        print(f"[inference] 체크포인트 없음: {checkpoint_path}")
        return False

    if not tokenizer_path.exists():
        print(f"[inference] 토크나이저 없음: {tokenizer_path}")
        return False

    # 디바이스 설정
    if torch.cuda.is_available():
        _device = "cuda"
    elif torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    print(f"[inference] 디바이스: {_device}")

    # 체크포인트 로드
    ckpt = torch.load(checkpoint_path, map_location=_device, weights_only=True)
    config = GPTConfig(**ckpt["config"])
    _model = GPT(config).to(_device)
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()

    print(f"[inference] 모델 로드 완료 (iter={ckpt['iter']}, val_loss={ckpt['val_loss']:.4f})")

    # 토크나이저 로드
    _tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"[inference] 토크나이저 로드 완료 (vocab_size={_tokenizer.vocab_size})")

    return True


def is_loaded() -> bool:
    return _model is not None and _tokenizer is not None


def generate(
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    repetition_penalty: float = 1.3,
) -> tuple[str, int]:
    """
    프롬프트를 받아 텍스트를 생성합니다.

    반환: (생성된 전체 텍스트, 생성된 토큰 수)
    """
    if not is_loaded():
        raise RuntimeError("모델이 로드되지 않았습니다. load_model()을 먼저 호출하세요.")

    # 프롬프트 인코딩
    input_ids = _tokenizer.encode(prompt)
    if not input_ids:
        input_ids = [0]  # 빈 프롬프트면 BOS 토큰으로 시작

    idx = torch.tensor([input_ids], dtype=torch.long, device=_device)
    prompt_len = idx.size(1)

    # 생성
    with torch.no_grad():
        output = _model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )

    # 생성된 부분만 디코딩 (프롬프트 제외)
    generated_ids = output[0].tolist()
    generated_text = _tokenizer.decode(generated_ids)
    tokens_generated = len(generated_ids) - prompt_len

    return generated_text, tokens_generated
