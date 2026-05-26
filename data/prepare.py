"""
데이터 준비 파이프라인.

수행하는 일:
  1) 입력 텍스트 읽기 (data/input.txt 또는 --demo로 번들 코퍼스)
  2) BPE 토크나이저 학습
  3) 특수 토큰 등록
  4) 전체 텍스트 인코딩 → train/val split
  5) data/{train.bin, val.bin, tokenizer.json, meta.json} 저장

train.py가 이 파일들을 읽어 학습합니다.

사용:
    python -m data.prepare --demo                       # 번들 데모 코퍼스로 빠른 검증
    python -m data.prepare --input data/input.txt       # 본인 텍스트로 학습용 준비
    python -m data.prepare --input data/input.txt --vocab-size 4096
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from model.tokenizer import BPETokenizer


# 채팅 포맷 + 시퀀스 경계용 표준 특수 토큰
SPECIAL_TOKENS = ["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/input.txt",
                        help="학습 코퍼스 텍스트 파일 경로")
    parser.add_argument("--demo", action="store_true",
                        help="번들된 데모 코퍼스(data/demo_corpus.txt) 사용")
    parser.add_argument("--vocab-size", type=int, default=1024,
                        help="BPE 어휘 크기 (특수 토큰 포함 전)")
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="validation 비율 (0~1)")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ----- 1) 입력 텍스트 로드 -----
    if args.demo:
        input_path = data_dir / "demo_corpus.txt"
    else:
        input_path = Path(args.input)

    if not input_path.exists():
        print(f"❌ 입력 파일이 없습니다: {input_path}")
        print(f"   → {input_path}에 텍스트를 두거나, --demo 옵션을 쓰세요.")
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8")
    print(f"입력: {input_path}  ({len(text):,}자)")

    # ----- 2) BPE 학습 -----
    print(f"\nBPE 학습 (목표 vocab_size={args.vocab_size})...")
    tok = BPETokenizer()
    tok.train(text, vocab_size=args.vocab_size, verbose=True)

    # ----- 3) 특수 토큰 등록 -----
    tok.register_special_tokens(SPECIAL_TOKENS)
    print(f"특수 토큰 추가 후 vocab size: {tok.vocab_size}")

    # uint16에 담아야 하므로 65535 이하 보장
    assert tok.vocab_size < 2**16, "vocab_size가 65535 초과면 uint32로 바꿔야 함"

    # ----- 4) 토크나이저 저장 -----
    tokenizer_path = data_dir / "tokenizer.json"
    tok.save(tokenizer_path)
    print(f"토크나이저 저장: {tokenizer_path}")

    # ----- 5) 인코딩 -----
    print(f"\n전체 코퍼스 인코딩 중...")
    ids = tok.encode(text)
    n = len(ids)
    ratio = len(text) / n if n else 0
    print(f"인코딩 완료: {n:,} 토큰  (압축률 {ratio:.2f} 자/토큰)")

    # ----- 6) train/val split -----
    n_val = int(n * args.val_frac)
    train_ids = ids[:-n_val] if n_val > 0 else ids
    val_ids = ids[-n_val:] if n_val > 0 else []

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)
    train_arr.tofile(data_dir / "train.bin")
    val_arr.tofile(data_dir / "val.bin")
    print(f"train.bin: {len(train_arr):,} 토큰")
    print(f"val.bin:   {len(val_arr):,} 토큰")

    # ----- 7) 메타데이터 -----
    meta = {
        "vocab_size": tok.vocab_size,
        "n_train_tokens": len(train_arr),
        "n_val_tokens": len(val_arr),
        "special_tokens": tok.special_tokens,
        "source_file": str(input_path),
        "source_chars": len(text),
    }
    (data_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2)
    )

    print(f"\n✅ 준비 완료. data/ 폴더:")
    print(f"   tokenizer.json, train.bin, val.bin, meta.json")
    print(f"\n다음 단계: python train.py")


if __name__ == "__main__":
    main()
