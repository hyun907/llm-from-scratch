"""
Transformer Block & GPT Model - From Scratch

GPT 아키텍처 (Decoder-only Transformer):

    Input IDs (B, T)
        │
        ├── Token Embedding  (B, T, n_embd)
        └── Position Embedding (B, T, n_embd)
        │
        ▼  (둘을 더함)
    ┌─────────────────────────────┐
    │  TransformerBlock × N        │
    │   ├── x + Attn(LayerNorm(x)) │  ← Pre-LN 방식 (GPT-2 이후 표준)
    │   └── x + FFN(LayerNorm(x))  │
    └─────────────────────────────┘
        │
        ▼
    LayerNorm
        │
        ▼
    Linear → Logits (B, T, vocab_size)

핵심 디테일:
- Pre-LayerNorm: 학습이 훨씬 안정적
- Weight Tying: token embedding과 lm_head가 가중치 공유 (메모리↓, 성능↑)
- GELU activation: ReLU보다 부드러운 곡선, GPT 표준
"""

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import MultiHeadSelfAttention


# ============================================================
# 모델 설정 (하이퍼파라미터를 한 곳에)
# ============================================================

@dataclass
class GPTConfig:
    vocab_size: int = 512        # 토크나이저 vocab 크기 (특수토큰 포함)
    max_seq_len: int = 256       # 처리 가능한 최대 시퀀스 길이
    n_layer: int = 6             # Transformer Block 개수
    n_head: int = 6              # Attention 헤드 개수
    n_embd: int = 384            # 임베딩 차원 (n_head로 나누어떨어져야 함)
    dropout: float = 0.1
    tie_weights: bool = True     # token embedding ↔ lm_head 가중치 공유


# ============================================================
# FeedForward (MLP) — Attention 사이사이의 "생각" 레이어
# ============================================================

class FeedForward(nn.Module):
    """
    표준 Transformer FFN: Linear → GELU → Linear → Dropout
    중간 차원은 보통 4 * n_embd (원 논문 관례).
    """

    def __init__(self, n_embd: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# Transformer Block — Attention + FFN with residual
# ============================================================

class TransformerBlock(nn.Module):
    """
    GPT-2 스타일 Pre-LayerNorm Block:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    LayerNorm을 residual 안쪽에 두는 게 핵심 (= Pre-LN).
    학습 초반 gradient 폭주를 막아줘서 deep network 학습이 가능해짐.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = MultiHeadSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.ffn = FeedForward(config.n_embd, dropout=config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ============================================================
# GPT — 전체 모델
# ============================================================

class GPT(nn.Module):
    """
    Decoder-only Transformer (GPT 계열).

    입력: token IDs (B, T)
    출력: logits (B, T, vocab_size)  ← 각 위치에서 "다음 토큰" 확률분포
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config

        # 임베딩
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        # Transformer 블록들
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layer)]
        )

        # 최종 LayerNorm + 출력 투영
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight Tying: 입력 임베딩과 출력 투영이 같은 행렬 공유
        # → 파라미터 수 vocab_size * n_embd 만큼 절약
        # → 입출력이 같은 의미공간에 살아서 학습도 잘 됨
        if config.tie_weights:
            self.lm_head.weight = self.token_emb.weight

        # 가중치 초기화 (GPT-2 표준)
        self.apply(self._init_weights)

        # residual 경로에 쌓이는 projection은 더 작게 초기화 (deep network 안정화)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("net.2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        idx:     (B, T) — 입력 토큰 ID
        targets: (B, T) — 다음 토큰 정답 (학습 시). 없으면 추론 모드.

        반환:
            logits: (B, T, vocab_size)
            loss:   targets가 있으면 cross-entropy, 없으면 None
        """
        B, T = idx.size()
        assert T <= self.config.max_seq_len, (
            f"입력 길이 {T}가 max_seq_len {self.config.max_seq_len}을 초과"
        )

        # 1) 임베딩 = 토큰 의미 + 위치 정보
        tok = self.token_emb(idx)                                    # (B, T, n_embd)
        pos = self.pos_emb(torch.arange(T, device=idx.device))       # (T, n_embd)
        x = self.drop(tok + pos)                                     # (B, T, n_embd)

        # 2) Transformer 블록 통과
        for block in self.blocks:
            x = block(x)

        # 3) 최종 정규화 + 어휘공간 투영
        x = self.ln_f(x)
        logits = self.lm_head(x)                                     # (B, T, vocab_size)

        # 4) 학습 시 loss 계산
        loss = None
        if targets is not None:
            # cross_entropy는 (N, C), (N,) 형태를 받음 → flatten
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,  # -1로 마스킹된 위치는 loss에서 제외
            )

        return logits, loss

    # ------------------------------------------------------------
    # Generate — 자기회귀 샘플링 (학습 안 했어도 동작은 검증 가능)
    # ------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        repetition_penalty: float = 1.0,
    ) -> torch.Tensor:
        """
        idx (B, T) 컨텍스트에서 시작해 max_new_tokens개의 토큰을 이어붙임.

        temperature:        1보다 크면 다양하게, 작으면 보수적으로
        top_k:              상위 k개 토큰 중에서만 샘플링 (None이면 전체)
        repetition_penalty: 1.0이면 효과 없음. 1.3~1.5 권장.
                            이미 등장한 토큰의 logit을 나눠서 재등장 확률을 낮춤.
                            - logit > 0: penalty로 나눔 → 값이 작아짐
                            - logit < 0: penalty를 곱함  → 절댓값이 커져 확률이 낮아짐

        주의: eval/train 모드 전환은 호출자가 직접 관리해야 합니다.
        이 함수 내부에서 self.eval()을 호출하지 않습니다.
        """
        for _ in range(max_new_tokens):
            # max_seq_len 넘으면 뒤쪽만 남김 (sliding window)
            idx_cond = idx[:, -self.config.max_seq_len :]

            logits, _ = self(idx_cond)
            # 마지막 위치의 분포만 사용 (다음 토큰을 예측해야 하므로)
            logits = logits[:, -1, :] / temperature

            # Repetition Penalty: 이미 등장한 토큰의 확률을 낮춤
            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = set(idx[b].tolist())
                    for token_id in seen:
                        if logits[b, token_id] > 0:
                            logits[b, token_id] /= repetition_penalty
                        else:
                            logits[b, token_id] *= repetition_penalty

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat([idx, next_id], dim=1)

        return idx

    # ------------------------------------------------------------
    # 유틸
    # ------------------------------------------------------------

    def num_params(self, non_embedding: bool = True) -> int:
        """
        파라미터 수. non_embedding=True면 position embedding은 빼고 셈
        (weight tying 덕분에 token embedding은 lm_head와 공유돼서 자동으로 한 번만 셈).
        """
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel()
        return n


# ============================================================
# 데모: 모델이 forward/loss/generate 모두 정상 동작하는지 검증
# ============================================================

if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"디바이스: {device}\n")

    # 작은 설정으로 빠르게 검증
    config = GPTConfig(
        vocab_size=512,
        max_seq_len=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        dropout=0.1,
    )
    model = GPT(config).to(device)

    print(f"모델 파라미터: {model.num_params():,}개 (non-embedding 기준)")
    print(f"전체 파라미터: {model.num_params(non_embedding=False):,}개\n")

    # --- forward shape 검증 ---
    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T), device=device)
    targets = torch.randint(0, config.vocab_size, (B, T), device=device)

    logits, loss = model(idx, targets)
    print(f"입력 shape:   {idx.shape}")
    print(f"logits shape: {logits.shape}  (예상: {(B, T, config.vocab_size)})")
    print(f"loss:         {loss.item():.4f}")

    assert logits.shape == (B, T, config.vocab_size), "logits shape 불일치"
    # 학습 전 random 모델의 cross-entropy는 ln(vocab_size) 근처여야 함
    expected_loss = math.log(config.vocab_size)
    print(f"예상 loss(랜덤): ~{expected_loss:.4f}  → 비슷하면 ✅\n")

    # --- weight tying 검증 ---
    if config.tie_weights:
        same = model.token_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()
        print(f"Weight tying: {'✅ 공유됨' if same else '❌ 분리됨'}")

    # --- causal 동작 재검증 (전체 모델 레벨) ---
    # 마지막 토큰 바꿔도 이전 위치의 logits은 같아야 함
    model.eval()
    with torch.no_grad():
        idx2 = idx.clone()
        idx2[:, -1] = (idx2[:, -1] + 1) % config.vocab_size
        l1, _ = model(idx)
        l2, _ = model(idx2)
    diff_prev = (l1[:, :-1] - l2[:, :-1]).abs().max().item()
    diff_last = (l1[:, -1] - l2[:, -1]).abs().max().item()
    print(f"마지막 토큰만 바꿨을 때 — 이전 logits 차이: {diff_prev:.6e} (0이어야 함)")
    print(f"                            마지막 logits 차이: {diff_last:.6e} (변화 있어야 함)")
    print("✅ Causal 동작 정상" if diff_prev < 1e-5 else "❌ Causal 실패")

    # --- generate 동작 검증 (학습 안 했으니 결과는 랜덤한 ID 시퀀스) ---
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(start, max_new_tokens=20, temperature=1.0, top_k=10)
    print(f"\ngenerate 출력 shape: {out.shape}  (예상: (1, 21))")
    print(f"생성된 토큰 ID 샘플: {out[0].tolist()}")
    print("✅ generate 정상 동작 (학습 전이라 내용은 랜덤)")
