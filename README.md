# LLM From Scratch

GPT-2 스타일 언어 모델을 밑바닥부터 구현하며 아키텍처를 이해하는 학습 프로젝트.

---

## 프로젝트 목적

AI 도구를 블랙박스로 쓰지 않기 위해, 언어 모델의 내부 동작을 코드 레벨에서 이해하는 것을 목표로 합니다.

- 토크나이저가 텍스트를 어떻게 숫자로 바꾸는지
- Attention이 토큰 간 관계를 어떻게 계산하는지
- 학습 루프에서 loss가 어떻게 gradient로 변환되는지

---

## 전체 구조

```
llm-from-scratch/
├── model/
│   ├── tokenizer.py          # BPE 토크나이저
│   ├── attention.py          # Multi-Head Causal Self-Attention
│   └── transformer.py        # GPT 전체 모델
├── data/
│   ├── prepare.py            # 데모 데이터 전처리
│   └── prepare_korean.py     # 한국어 데이터 전처리
├── train.py                  # 학습 루프
├── api/
│   ├── inference.py          # 모델 로드 + generate()
│   ├── main.py               # FastAPI 추론 서버
│   └── requirements.txt
├── web/
│   ├── app/
│   │   ├── page.tsx          # 메인 페이지
│   │   └── api/generate/
│   │       └── route.ts      # BFF — FastAPI 프록시
│   ├── components/
│   │   ├── PromptForm.tsx
│   │   └── GeneratedOutput.tsx
│   └── lib/api.ts
├── DEVLOG.md                 # AI 협업 작업 기록
└── checkpoints/              # 학습된 모델 저장 (gitignore)
```

---

## 서비스 아키텍처

```
사용자
  ↓
Next.js (web/) — 3000포트
  ├── 프론트엔드 UI (PromptForm, GeneratedOutput)
  └── API Route /api/generate  ← BFF 프록시
          ↓
      FastAPI (api/) — 8000포트
          └── 모델 로드 + 텍스트 생성
```

Next.js API Route가 BFF(Backend for Frontend)로 FastAPI를 감싸는 구조입니다.
프론트엔드는 FastAPI 주소를 직접 알지 않아도 되고, 인증/로깅은 BFF 레이어에서 추가할 수 있습니다.

---

## 실행 방법

**터미널 1 — FastAPI 추론 서버**
```bash
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

**터미널 2 — Next.js 웹**
```bash
cd web
npm install
npm run dev
```

`http://localhost:3000`에서 텍스트 생성 인터페이스를 사용할 수 있습니다.

---

## 데이터 준비 및 학습

**데모 데이터 (빠른 시작)**
```bash
python -m data.prepare --demo
python train.py
```

**한국어 데이터**
```bash
# HuggingFace 위키 다운로드 (약 2M 자, HF_TOKEN 권장)
export HF_TOKEN=your_token_here
python -m data.prepare_korean --n-samples 10000 --vocab-size 4096

# 6-layer 모델 학습 (MPS 기준 약 20분, early stopping 기준 ~4000 iter)
python train.py --max-iters 5000 --n-layer 6 --n-head 6 --n-embd 384 --block-size 128 --batch-size 32
```

> **주의:** `--block-size 256` 이상은 Apple Silicon MPS에서 데드락 발생 확인. `128` 권장.

---

## 모델 설정

| 설정 | 데모 | 한국어 v1 | 한국어 v2 (현재) | 설명 |
|------|------|----------|----------------|------|
| vocab_size | 512 | 2,053 | **4,101** | 토크나이저 어휘 크기 |
| n_layer | 4 | 6 | 6 | Transformer Block 개수 |
| n_head | 4 | 6 | 6 | Attention 헤드 수 |
| n_embd | 192 | 384 | 384 | 임베딩 차원 |
| max_seq_len | 64 | 128 | 128 | 최대 시퀀스 길이 |
| train tokens | 2,121 | 541,329 | **1,039,157** | 학습 데이터 크기 |
| params | ~1M | ~6M | **12.2M** | 모델 파라미터 수 |
| best val loss | - | 3.46 | **4.80** | (vocab 다름, 직접 비교 불가) |

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

---

### 2. Causal Self-Attention (`model/attention.py`)

**Attention이 필요한 이유:**
"나는 오늘 학교에 갔다"에서 "갔다"의 주어가 누구인지 알려면 "나는"을 참조해야 합니다.
각 토큰이 다른 모든 토큰과의 관련도를 계산하는 게 Attention입니다.

**Q, K, V:**
- **Q (Query)**: 내가 찾는 것
- **K (Key)**: 각 토큰의 라벨
- **V (Value)**: 실제 가져올 정보

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

**Causal Mask:**
GPT는 왼쪽에서 오른쪽으로 토큰을 생성합니다. 미래 토큰을 미리 볼 수 없으므로 하삼각 마스크로 차단합니다.

```
1 0 0 0    "나는"  → "나는"만 봄
1 1 0 0    "오늘"  → "나는", "오늘" 봄
1 1 1 0    "학교에" → 앞 3개 봄
1 1 1 1    "갔다"  → 전부 봄
```

**Multi-Head:** 헤드마다 다른 관점(문법, 시간, 장소 등)으로 관계를 학습하고, 마지막에 합칩니다.

---

### 3. GPT 모델 (`model/transformer.py`)

```
입력 토큰 ID
    ↓
Token Embedding + Position Embedding
    ↓
TransformerBlock × N
│   ├── x + Attention(LayerNorm(x))    ← Pre-LN + Residual
│   └── x + FFN(LayerNorm(x))
    ↓
LayerNorm → Linear → logits
```

- **Residual Connection**: gradient 소멸 방지
- **Pre-LayerNorm**: GPT-2 이후 표준, 학습 초반 안정성
- **Weight Tying**: Token Embedding과 lm_head가 같은 가중치 공유 → 파라미터 절약 + 의미 공간 통일

---

### 4. 학습 루프 (`train.py`)

```
배치 추출 → forward(loss 계산) → backward(gradient) → optimizer step
```

- **AdamW**: 행렬 가중치에만 weight decay, bias/LayerNorm은 제외
- **Warmup + Cosine Decay**: 초기 발산 방지 후 부드럽게 감소
- **Gradient Clipping**: gradient 폭주 방지

---

## 환경변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `FASTAPI_URL` | `http://localhost:8000` | Next.js → FastAPI 주소 |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | CORS 허용 origin (쉼표 구분) |
| `HF_TOKEN` | 없음 | HuggingFace 인증 토큰 (데이터 다운로드 속도 향상) |

## 알려진 이슈 / 트러블슈팅

| 증상 | 원인 | 해결 |
|------|------|------|
| 생성 텍스트에 `□` 출력 | 불완전한 UTF-8 바이트 시퀀스 | `api/inference.py`에서 후처리 제거 (이미 적용됨) |
| MPS 프리즈 (장시간 멈춤) | `--block-size 256` 이상 | `--block-size 128` 사용 |
| val loss가 개선 안 됨 | 데이터 부족으로 즉시 과적합 | 데이터 최소 500K 토큰 이상 확보 후 학습 |
| HF 다운로드 rate limit | 미인증 요청 | `HF_TOKEN` 환경변수 설정 |
