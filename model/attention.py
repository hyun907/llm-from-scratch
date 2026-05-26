"""
Multi-Head Self-Attention - From Scratch

"Attention is All You Need" (Vaswani et al., 2017) 논문의 핵심 수식:

    Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V

GPT는 'Causal Self-Attention'을 사용:
- Self: Q, K, V 모두 같은 시퀀스에서 나옴
- Causal: 미래 토큰은 못 보게 마스킹 (autoregressive 생성용)
- Multi-Head: 여러 attention 헤드를 병렬로 → 다양한 관계 학습
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    GPT용 Causal Multi-Head Self-Attention.

    입력:  (batch, seq_len, n_embd)
    출력:  (batch, seq_len, n_embd)
    """

    def __init__(self, n_embd: int, n_head: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert n_embd % n_head == 0, "임베딩 차원은 헤드 수로 나누어떨어져야 함"

        self.n_embd = n_embd          # 임베딩 차원 (예: 384)
        self.n_head = n_head          # 헤드 개수 (예: 6)
        self.head_dim = n_embd // n_head  # 헤드당 차원 (384/6 = 64)

        # Q, K, V를 한 번에 계산하는 단일 선형변환
        # 입력: (B, T, n_embd) -> 출력: (B, T, 3*n_embd)
        # 이걸 나중에 Q, K, V로 split
        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd, bias=False)

        # 출력 투영 (헤드들을 다시 합친 후 적용)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=False)

        # Dropout
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal Mask: 상삼각 행렬 (미래 위치를 -inf로 만들기 위함)
        # register_buffer = 학습 안 되지만 모델과 함께 device 이동되는 텐서
        # shape: (1, 1, max_seq_len, max_seq_len)
        causal_mask = torch.tril(
            torch.ones(max_seq_len, max_seq_len, dtype=torch.bool)
        ).view(1, 1, max_seq_len, max_seq_len)
        self.register_buffer("causal_mask", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch, seq_len, n_embd

        # 1) Q, K, V 계산 후 헤드 차원으로 분리
        # qkv: (B, T, 3*C) -> 3개의 (B, T, C)로 split
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        # (B, T, C) -> (B, n_head, T, head_dim)
        # 헤드별로 따로 attention 계산 가능하게 차원 재배열
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 2) Attention score 계산: Q · Kᵀ / √d_k
        # (B, nH, T, hd) @ (B, nH, hd, T) -> (B, nH, T, T)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3) Causal Mask 적용: 미래 위치(상삼각)를 -inf로
        # mask가 False인 위치를 -inf로 → softmax 후 0이 됨
        attn_scores = attn_scores.masked_fill(
            self.causal_mask[:, :, :T, :T] == 0, float("-inf")
        )

        # 4) Softmax로 attention 가중치 확률화
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 5) Value에 가중치 적용
        # (B, nH, T, T) @ (B, nH, T, hd) -> (B, nH, T, hd)
        out = attn_weights @ v

        # 6) 헤드 합치기: (B, nH, T, hd) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # 7) 출력 투영
        out = self.out_proj(out)
        out = self.resid_dropout(out)
        return out


# ============================================================
# 데모: 직접 실행하면 attention이 잘 작동하는지 검증
# ============================================================

if __name__ == "__main__":
    # 환경 체크
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"디바이스: {device}\n")

    # 작은 모델로 테스트
    n_embd = 64
    n_head = 4
    max_seq_len = 16
    batch_size = 2
    seq_len = 10

    attn = MultiHeadSelfAttention(
        n_embd=n_embd, n_head=n_head, max_seq_len=max_seq_len
    ).to(device)

    # 랜덤 입력
    x = torch.randn(batch_size, seq_len, n_embd, device=device)
    print(f"입력 shape: {x.shape}")

    # Forward
    out = attn(x)
    print(f"출력 shape: {out.shape}")
    assert out.shape == x.shape, "입출력 shape이 같아야 함"
    print("✅ Shape 검증 통과\n")

    # Causal Mask 검증: 미래 정보가 누설되지 않는지
    # 첫 번째 토큰을 바꿔도, t=0 위치의 출력은 같아야 정상
    # 마지막 토큰을 바꾸면, t=마지막 위치만 출력이 바뀌어야 함
    x2 = x.clone()
    x2[:, -1, :] = torch.randn(batch_size, n_embd, device=device)  # 마지막 토큰만 변경

    attn.eval()  # dropout 끄기
    with torch.no_grad():
        out1 = attn(x)
        out2 = attn(x2)

    # 마지막 토큰을 제외한 모든 위치는 동일해야 함 (Causal Mask 작동)
    diff_except_last = (out1[:, :-1] - out2[:, :-1]).abs().max().item()
    diff_last = (out1[:, -1] - out2[:, -1]).abs().max().item()

    print(f"마지막 토큰만 바꿨을 때:")
    print(f"  이전 위치들의 출력 차이: {diff_except_last:.6f} (0에 가까워야 함)")
    print(f"  마지막 위치의 출력 차이: {diff_last:.6f} (변화 있어야 함)")

    if diff_except_last < 1e-5:
        print("✅ Causal Mask 정상 작동 (미래 정보 차단됨)")
    else:
        print("❌ Causal Mask 실패")

    # 파라미터 수 확인
    n_params = sum(p.numel() for p in attn.parameters())
    print(f"\nAttention 파라미터 수: {n_params:,}")
