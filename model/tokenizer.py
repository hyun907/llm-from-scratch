"""
BPE (Byte Pair Encoding) Tokenizer - From Scratch

알고리즘:
1. 텍스트를 UTF-8 바이트 단위로 분해
2. 가장 자주 등장하는 인접 바이트 쌍을 찾아 새 토큰으로 병합
3. 원하는 vocab size에 도달할 때까지 2번을 반복
4. 학습된 병합 규칙(merges)을 저장 → 이걸로 encode/decode

GPT-2, GPT-3, GPT-4 모두 이 방식의 변형(byte-level BPE)을 사용함.
"""

import json
import regex as re
from collections import Counter
from pathlib import Path


# GPT-2가 사용한 정규식 패턴 (단어/숫자/공백을 적절히 분리)
GPT2_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(self):
        # 학습된 병합 규칙: (token_a, token_b) -> merge_rank
        # rank가 낮을수록 먼저 학습된 = 자주 등장하는 쌍
        self.merges: dict[tuple[int, int], int] = {}

        # 토큰 ID -> 바이트열
        self.vocab: dict[int, bytes] = {}

        # 특수 토큰: 채팅용
        self.special_tokens: dict[str, int] = {}

    # ============================================================
    # 학습: 코퍼스에서 BPE 규칙 추출
    # ============================================================

    def train(self, text: str, vocab_size: int, verbose: bool = True):
        """텍스트로부터 vocab_size 크기의 BPE 어휘를 학습."""
        assert vocab_size >= 256, "최소 256 (모든 바이트값)"

        # 1단계: 텍스트를 GPT-2 정규식으로 chunk로 분할
        # ("안녕하세요" → ["안녕하세요"], "Hello world" → ["Hello", " world"])
        chunks = re.findall(GPT2_PATTERN, text)

        # 2단계: 각 chunk를 바이트 시퀀스로 변환 (UTF-8)
        # 예: "안녕" → b'\xec\x95\x88\xeb\x85\x95' → [236, 149, 136, 235, 133, 149]
        token_sequences: list[list[int]] = [
            list(chunk.encode("utf-8")) for chunk in chunks
        ]

        # 3단계: 초기 vocab = 모든 바이트값 (0~255)
        self.vocab = {i: bytes([i]) for i in range(256)}

        # 4단계: vocab_size에 도달할 때까지 가장 빈번한 쌍을 병합
        num_merges = vocab_size - 256
        for i in range(num_merges):
            # 모든 인접 쌍의 빈도 카운트
            pair_counts = Counter()
            for seq in token_sequences:
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] += 1

            if not pair_counts:
                break

            # 가장 빈번한 쌍 선택
            top_pair = max(pair_counts, key=pair_counts.get)
            new_id = 256 + i

            # 모든 시퀀스에서 이 쌍을 새 ID로 교체
            token_sequences = [
                self._merge(seq, top_pair, new_id) for seq in token_sequences
            ]

            # vocab과 merges 업데이트
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            if verbose and (i + 1) % 50 == 0:
                merged_text = self.vocab[new_id].decode("utf-8", errors="replace")
                print(
                    f"  병합 {i+1}/{num_merges}: "
                    f"{top_pair} → {new_id} "
                    f"({pair_counts[top_pair]}회) [{merged_text!r}]"
                )

        if verbose:
            print(f"✅ 학습 완료. 최종 vocab size: {len(self.vocab)}")

    @staticmethod
    def _merge(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
        """시퀀스에서 pair를 찾아 new_id로 교체."""
        result = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                result.append(new_id)
                i += 2
            else:
                result.append(seq[i])
                i += 1
        return result

    # ============================================================
    # 특수 토큰 등록 (채팅 포맷용)
    # ============================================================

    def register_special_tokens(self, tokens: list[str]):
        """<|user|>, <|assistant|> 같은 특수 토큰 추가."""
        base = len(self.vocab)
        for i, tok in enumerate(tokens):
            new_id = base + i
            self.special_tokens[tok] = new_id
            self.vocab[new_id] = tok.encode("utf-8")

    # ============================================================
    # Encode: 텍스트 → 토큰 ID 리스트
    # ============================================================

    def encode(self, text: str) -> list[int]:
        """학습된 merges로 텍스트를 토큰 ID 시퀀스로 변환."""
        # 특수 토큰이 있으면 먼저 split
        if self.special_tokens:
            # 정규식으로 special token을 경계로 split
            pattern = (
                "(" + "|".join(re.escape(s) for s in self.special_tokens) + ")"
            )
            parts = re.split(pattern, text)
        else:
            parts = [text]

        ids: list[int] = []
        for part in parts:
            if not part:
                continue
            if part in self.special_tokens:
                ids.append(self.special_tokens[part])
            else:
                ids.extend(self._encode_ordinary(part))
        return ids

    def _encode_ordinary(self, text: str) -> list[int]:
        """특수 토큰 없는 일반 텍스트의 BPE 인코딩."""
        chunks = re.findall(GPT2_PATTERN, text)
        ids: list[int] = []
        for chunk in chunks:
            tokens = list(chunk.encode("utf-8"))
            # 학습 때 정한 순서대로(낮은 rank부터) 병합 적용
            while len(tokens) >= 2:
                pairs = list(zip(tokens, tokens[1:]))
                # 학습된 merges 중 가장 먼저 학습된 쌍 찾기
                best = min(
                    pairs, key=lambda p: self.merges.get(p, float("inf"))
                )
                if best not in self.merges:
                    break  # 더 이상 병합할 게 없음
                tokens = self._merge(tokens, best, self.merges[best])
            ids.extend(tokens)
        return ids

    # ============================================================
    # Decode: 토큰 ID 리스트 → 텍스트
    # ============================================================

    def decode(self, ids: list[int]) -> str:
        """토큰 ID 시퀀스를 다시 텍스트로 복원."""
        byte_chunks = [self.vocab[i] for i in ids]
        return b"".join(byte_chunks).decode("utf-8", errors="replace")

    # ============================================================
    # 저장 / 로드
    # ============================================================

    def save(self, path: str | Path):
        """학습된 토크나이저를 JSON으로 저장."""
        path = Path(path)
        data = {
            "merges": [
                [list(pair), new_id] for pair, new_id in self.merges.items()
            ],
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """저장된 토크나이저 로드."""
        path = Path(path)
        data = json.loads(path.read_text())

        tokenizer = cls()
        tokenizer.vocab = {i: bytes([i]) for i in range(256)}

        for pair_list, new_id in data["merges"]:
            pair = (pair_list[0], pair_list[1])
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = (
                tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]
            )

        for tok, tok_id in data["special_tokens"].items():
            tokenizer.special_tokens[tok] = tok_id
            tokenizer.vocab[tok_id] = tok.encode("utf-8")

        return tokenizer

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ============================================================
# 데모: 직접 실행하면 작은 코퍼스로 학습 + 테스트
# ============================================================

if __name__ == "__main__":
    sample_text = """
    안녕하세요. 반갑습니다. 안녕하세요 여러분.
    안녕하세요는 한국어 인사말입니다.
    저는 개발자입니다. 저는 코딩을 좋아합니다.
    딥러닝과 자연어처리를 공부하고 있습니다.
    Hello world. Hello everyone. Hello hello hello.
    """ * 30

    print("=" * 50)
    print("BPE 토크나이저 학습 시작")
    print("=" * 50)

    tok = BPETokenizer()
    tok.train(sample_text, vocab_size=400, verbose=True)

    # 특수 토큰 추가
    tok.register_special_tokens(
        ["<|bos|>", "<|eos|>", "<|user|>", "<|assistant|>", "<|pad|>"]
    )

    print(f"\n최종 vocab size: {tok.vocab_size}")

    # 인코딩/디코딩 테스트
    test = "안녕하세요. 저는 개발자입니다."
    ids = tok.encode(test)
    decoded = tok.decode(ids)

    print(f"\n원본:    {test}")
    print(f"토큰 ID: {ids}")
    print(f"토큰 수: {len(ids)} (원본 글자수: {len(test)})")
    print(f"복원:    {decoded}")
    print(f"일치:    {test == decoded} ✅" if test == decoded else "❌ 불일치!")

    # 채팅 포맷 테스트
    chat = "<|user|>안녕<|assistant|>반가워요"
    ids = tok.encode(chat)
    print(f"\n채팅:    {chat}")
    print(f"토큰 ID: {ids}")
    print(f"복원:    {tok.decode(ids)}")
