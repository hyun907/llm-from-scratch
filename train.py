"""
GPT 학습 루프.

전제: 먼저 `python -m data.prepare --demo` (또는 본인 데이터로) 실행해
      data/{train.bin, val.bin, tokenizer.json, meta.json}이 있어야 함.

사용:
    python train.py                          # 기본 설정으로 학습
    python train.py --max-iters 5000         # 더 길게
    python train.py --n-layer 8 --n-embd 512 # 더 크게

체크포인트는 checkpoints/model.pt에 저장됨.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig
from model.tokenizer import BPETokenizer


# ============================================================
# 배치 추출 — memmap에서 random offset으로 시퀀스 추출
# ============================================================

def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    """
    무작위 위치 batch_size개를 골라 길이 block_size 시퀀스 추출.
    x: (B, T)         — 입력
    y: (B, T)         — 한 칸 미래(= 다음 토큰 정답)
    """
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # memmap 슬라이스를 int64로 복사 (PyTorch Embedding은 int64/Long 받음)
    x = torch.stack([
        torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix
    ])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


# ============================================================
# 평가
# ============================================================

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, device, n_iters):
    """train/val 각각 n_iters개 배치로 평균 loss 측정."""
    model.eval()
    out = {}
    for split, data in (("train", train_data), ("val", val_data)):
        if len(data) < block_size + 1:
            out[split] = float("nan")
            continue
        losses = []
        for _ in range(n_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


# ============================================================
# 옵티마이저 설정 — weight decay 그룹/no-decay 그룹 분리
# ============================================================

def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    """
    행렬 가중치(Linear, Embedding)에만 weight decay 적용.
    bias와 LayerNorm 파라미터는 decay 안 함 (정규화 의미 X, 학습 불안정 야기).
    """
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.dim() >= 2:
            decay.append(p)
        else:
            nodecay.append(p)

    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": nodecay, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay)
    n_nodecay = sum(p.numel() for p in nodecay)
    print(f"옵티마이저: decay 그룹 {n_decay:,} / no-decay 그룹 {n_nodecay:,}")
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


# ============================================================
# 학습률 스케줄 — linear warmup + cosine decay
# ============================================================

def get_lr(it: int, max_iters: int, warmup_iters: int, max_lr: float, min_lr: float):
    if it < warmup_iters:
        return max_lr * (it + 1) / warmup_iters
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# ============================================================
# 메인
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    # 경로
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="checkpoints")
    # 데이터/모델 형태
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=64,
                        help="시퀀스 길이 (=max_seq_len). 큰 코퍼스면 128~256 권장")
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.1)
    # 학습
    parser.add_argument("--max-iters", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    # 생성 미리보기
    parser.add_argument("--sample-tokens", type=int, default=120,
                        help="학습 후 생성 미리보기 토큰 수 (0이면 끔)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ----- 디바이스 -----
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"디바이스: {device}")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----- meta / data 로드 -----
    meta_path = data_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(
            f"❌ {meta_path}가 없어요. 먼저 `python -m data.prepare --demo` 실행하세요."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    vocab_size = meta["vocab_size"]

    train_data = np.memmap(data_dir / "train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap(data_dir / "val.bin", dtype=np.uint16, mode="r")

    print(f"vocab_size: {vocab_size}")
    print(f"train: {len(train_data):,} 토큰 / val: {len(val_data):,} 토큰")

    # 데이터가 block_size보다 작으면 학습 불가
    if len(train_data) < args.block_size + 1:
        raise SystemExit(
            f"❌ train 데이터({len(train_data)} 토큰)가 block_size({args.block_size})보다 작아요.\n"
            f"   → --block-size를 줄이거나 더 큰 코퍼스를 사용하세요."
        )

    # ----- 모델 -----
    config = GPTConfig(
        vocab_size=vocab_size,
        max_seq_len=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT(config).to(device)
    n_params = model.num_params()
    print(f"모델: n_layer={args.n_layer} n_head={args.n_head} n_embd={args.n_embd} "
          f"block_size={args.block_size}")
    print(f"파라미터: {n_params:,}개 ({n_params/1e6:.2f}M, non-embedding)")

    optimizer = configure_optimizer(model, args.learning_rate, args.weight_decay)

    # ----- 학습 루프 -----
    print(f"\n학습 시작: {args.max_iters} iters\n")
    t0 = time.time()
    best_val = float("inf")

    for it in range(args.max_iters + 1):
        # LR 업데이트
        lr = get_lr(it, args.max_iters, args.warmup_iters,
                    args.learning_rate, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # 평가 + 체크포인트
        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(
                model, train_data, val_data,
                args.block_size, args.batch_size, device, args.eval_iters,
            )
            dt = time.time() - t0
            print(f"iter {it:5d} | lr {lr:.2e} | "
                  f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                  f"elapsed {dt:6.1f}s")

            # val 개선되면 best 체크포인트 저장
            if losses["val"] < best_val and not math.isnan(losses["val"]):
                best_val = losses["val"]
                ckpt = {
                    "model_state": model.state_dict(),
                    "config": config.__dict__,
                    "iter": it,
                    "val_loss": losses["val"],
                }
                torch.save(ckpt, out_dir / "model.pt")

        if it == args.max_iters:
            break

        # 학습 스텝
        x, y = get_batch(train_data, args.block_size, args.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

    print(f"\n✅ 학습 완료. best val loss: {best_val:.4f}")
    print(f"   체크포인트: {out_dir / 'model.pt'}")

    # ----- 학습 결과 미리보기 -----
    if args.sample_tokens > 0:
        print("\n--- 생성 미리보기 ---")
        tok = BPETokenizer.load(data_dir / "tokenizer.json")
        bos_id = tok.special_tokens.get("<|bos|>")
        start_id = bos_id if bos_id is not None else 0
        start = torch.tensor([[start_id]], dtype=torch.long, device=device)
        out = model.generate(
            start, max_new_tokens=args.sample_tokens,
            temperature=0.8, top_k=40,
        )
        # bos 제외하고 디코딩
        text = tok.decode(out[0].tolist()[1:])
        print(text)


if __name__ == "__main__":
    main()
