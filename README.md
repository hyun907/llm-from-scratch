# LLM From Scratch

GPT-2 스타일 언어 모델을 밑바닥부터 구현하며 아키텍처를 이해하는 학습 프로젝트.

---

## 프로젝트 목적

AI 도구를 블랙박스로 쓰지 않기 위해, 언어 모델의 내부 동작을 코드 레벨에서 이해하는 것을 목표로 합니다.

- 토크나이저가 텍스트를 어떻게 숫자로 바꾸는지
- Attention이 토큰 간 관계를 어떻게 계산하는지
- 학습 루프에서 loss가 어떻게 gradient로 변환되는지

---

## 구조

```
llm-from-scratch/
├── model/
│   ├── tokenizer.py      # BPE 토크나이저
│   ├── attention.py      # Multi-Head Causal Self-Attention
│   └── transformer.py    # GPT 전체 모델
├── data/
│   └── prepare.py        # 데이터 전처리 (바이너리 변환)
├── train.py              # 학습 루프
├── api/                  # (구현 예정) 추론 API 서버
└── web/                  # (구현 예정) 인터페이스
```

---

## 핵심 개념

### 1. BPE 토크나이저 (`model/tokenizer.py`)

모델은 텍스트를 직접 읽지 못하고 숫자만 받습니다. BPE(Byte Pair Encoding)는 텍스트를 숫자 ID 시퀀스로 변환하는 방법입니다.

**학습 과정:**
1. 텍스트를 UTF-8 바이트 단위(0~255)로 쪼갠다
2. 가장 자주 인접하는 바이트 쌍을 찾는다
3. 그 쌍을 새 토큰 ID(256부터 시작)로 병합한다
4. vocab_size에 도달할 때까지 반복한다

```
"안" (3바이트) → [236, 149, 128]
  1회 병합: (236, 149) → 256  →  [256, 128]
  2회 병합: (256, 128) → 257  →  [257]
```

**결과:** 자주 쓰이는 단어는 짧은 토큰, 드문 단어는 여러 토큰으로 표현된다.

학습된 병합 규칙(`merges`)은 JSON으로 저장되고, 새 텍스트 encode 시 같은 규칙을 순서대로 적용합니다.

---

### 2. Causal Self-Attention (`model/attention.py`)

**Attention이 필요한 이유:**
"나는 오늘 학교에 갔다"에서 "갔다"의 주어가 누구인지 알려면 "나는"을 참조해야 합니다. 각 토큰이 다른 모든 토큰과의 관련도를 계산하는 게 Attention입니다.

**Q, K, V:**
- **Q (Query)**: 내가 찾는 것
- **K (Key)**: 각 토큰의 라벨
- **V (Value)**: 실제 가져올 정보

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

`√d_k`로 나누는 이유: 점수가 커질수록 softmax가 한쪽으로 쏠려 gradient가 소멸하기 때문.

**Causal Mask:**
GPT는 왼쪽에서 오른쪽으로 토큰을 생성합니다. 미래 토큰을 미리 보면 안 되므로, 각 토큰은 자신과 이전 토큰만 볼 수 있습니다.

```
하삼각 마스크:        미래 위치는 -inf → softmax 후 0
1 0 0 0              "나는"  → "나는"만 봄
1 1 0 0              "오늘"  → "나는", "오늘" 봄
1 1 1 0              "학교에" → 앞 3개 봄
1 1 1 1              "갔다"  → 전부 봄
```

**Multi-Head:**
헤드마다 다른 관점(문법, 시간, 장소 등)으로 관계를 학습하고, 마지막에 합칩니다.

---

### 3. GPT 모델 (`model/transformer.py`)

```
입력 토큰 ID
    ↓
Token Embedding + Position Embedding   ← ID를 벡터로, 위치 정보 추가
    ↓
TransformerBlock × N
│   ├── x + Attention(LayerNorm(x))    ← Pre-LN + Residual
│   └── x + FFN(LayerNorm(x))
    ↓
LayerNorm
    ↓
Linear → logits (vocab_size개 점수)
```

**Residual Connection (`x + ...`):**
Attention 결과를 원본 x에 더합니다. 레이어가 깊어질 때 gradient가 소멸하는 문제를 방지합니다.

**Pre-LayerNorm:**
Attention에 넣기 전에 정규화합니다 (원 논문은 Post-LN). 학습 초반 안정성이 높아 GPT-2 이후 표준이 되었습니다.

**FFN (FeedForward):**
`Linear(n_embd → 4×n_embd) → GELU → Linear(4×n_embd → n_embd)`
Attention이 토큰 간 관계를 보는 반면, FFN은 각 토큰을 개별 처리합니다.

**Weight Tying:**
Token Embedding과 출력 Linear(`lm_head`)가 같은 가중치를 공유합니다.
- 파라미터 수 절약 (vocab_size × n_embd)
- 입력/출력이 같은 의미 공간을 쓰므로 학습 효율 향상

---

### 4. 학습 루프 (`train.py`)

**학습의 흐름:**
```
배치 추출 → forward(loss 계산) → backward(gradient) → optimizer step
```

**Loss (Cross-Entropy):**
각 위치에서 정답 토큰에 얼마나 높은 점수를 줬는지 측정합니다.
랜덤 초기화 직후 loss는 `ln(vocab_size)` 근처입니다.

**Optimizer (AdamW):**
decay/no-decay 그룹을 분리합니다.
- `dim >= 2` (행렬 가중치) → weight decay 적용
- bias, LayerNorm → weight decay 미적용 (학습 불안정 야기)

**학습률 스케줄 (Warmup + Cosine Decay):**
```
학습률
  ↑
  |   /‾‾\
  |  /     \
  | /       \________
  +------------------→ 스텝
  warmup   cosine decay
```
- Warmup: 초기 가중치가 랜덤인 상태에서 큰 학습률로 시작하면 발산할 수 있어 서서히 올림
- Cosine decay: Linear보다 부드럽게 감소, 마지막에 fine-tuning 효과

**Gradient Clipping:**
gradient가 급격히 커지는 폭주를 방지합니다.

---

## 실행 방법

```bash
# 의존성 설치
pip install torch numpy regex

# 데모 데이터 준비
python -m data.prepare --demo

# 학습
python train.py

# 옵션
python train.py --max-iters 5000 --n-layer 8 --n-embd 512
```

---

## 모델 설정 (기본값)

| 설정 | 값 | 설명 |
|------|-----|------|
| vocab_size | 512 | 토크나이저 어휘 크기 |
| n_layer | 4 | Transformer Block 개수 |
| n_head | 4 | Attention 헤드 수 |
| n_embd | 192 | 임베딩 차원 |
| max_seq_len | 64 | 최대 시퀀스 길이 |

---

## 구현 예정

- `api/` — 학습된 모델 서빙 API (FastAPI)
- `web/` — 텍스트 생성 인터페이스
