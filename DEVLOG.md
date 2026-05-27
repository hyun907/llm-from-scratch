# DEVLOG — LLM From Scratch

AI 작업 기록. 무엇을 AI에게 시켰는지, 어떤 결정을 내렸는지 추적합니다.

---

## 2026-05-26

### 아키텍처 설계 (Human)
- Next.js (web/) + FastAPI (api/) 두 서버 구조 결정
- Next.js API Route를 BFF로 두어 FastAPI를 프론트가 직접 모르게 분리
- 이유: 나중에 인증/로깅/캐싱 추가 시 프론트 코드 변경 없이 가능

### 구현 범위 (AI → Human 승인)
계획된 파일 목록을 먼저 제시하고 승인 후 구현 시작.
→ 승인됨

### AI가 구현한 것
- `api/requirements.txt`
- `api/inference.py` — 모델 로드 + generate()
- `api/main.py` — FastAPI 라우트

### Human이 결정한 것
- Next.js 선택 (백엔드 확장 가능성 고려)
- 멘토 방식 3가지 적용 결정 (plan-first, 역할 분리, 로깅)
- BFF 패턴 채택

### AI가 구현한 것 (web/)
- `web/` 전체 파일 구조 (Next.js, Tailwind 설정 포함)
- `app/api/generate/route.ts` — BFF 프록시
- `components/PromptForm.tsx`, `GeneratedOutput.tsx`
- `app/page.tsx`, `lib/api.ts`

---

### 코드 리뷰 세션

**리뷰 방식:** AI Reviewer 별도 실행 → 코드 수정 없이 코멘트만 → Human이 수정 여부 판단

**발견된 이슈:** High 3 / Medium 8 / Low 9

**Human이 수정한 것 (High 3건):**
- `model/transformer.py`: `generate()` 내부 `self.eval()` 제거
  - 이유: 학습 중 호출 시 dropout이 꺼진 채로 학습 계속되는 사이드이펙트
- `api/inference.py`: `weights_only=False` → `weights_only=True`
  - 이유: pickle 역직렬화를 통한 임의 코드 실행 취약점
- `web/app/api/generate/route.ts`: body 타입 검증 추가 + `AbortSignal.timeout(60s)`
  - 이유: 검증 없이 업스트림에 전달되는 페이로드 문제, 타임아웃 누락

**보류 중 (Medium/Low):** CORS 환경변수화, AsyncIO Lock, 프롬프트 길이 검증 등

---

### 한국어 데이터 + 재학습 (진행 중)

**AI가 구현한 것:**
- `data/prepare_korean.py` — HuggingFace 다운로드 시도 + 번들 코퍼스 fallback

**Human이 결정한 것:**
- 학습 하이퍼파라미터: `--max-iters 5000 --n-layer 6 --n-head 6 --n-embd 384 --block-size 128`
- vocab_size 2048 (한국어 멀티바이트 대응)

**상태:** 데이터 준비 중 → 완료 후 학습 예정

---

> 이후 작업은 이 파일에 계속 추가합니다.

---

## 2026-05-27

### UTF-8 깨진 글자 버그 수정 (Human 발견 → AI 수정)

**증상:** 웹 UI에서 프롬프트 입력 시 `□` 문자 다수 출력
```
"토큰 데이며할 수 있다. 종드를 조개□에서 시퀀스 언뛐가..."
```

**원인 분석:**
- 모델은 byte-level BPE로 동작 (한국어 1글자 = UTF-8 3바이트)
- 모델이 3바이트짜리 글자의 첫 1~2바이트만 생성하고 다음 글자로 넘어가면 불완전한 UTF-8 시퀀스 발생
- `tokenizer.py`의 `decode(errors="replace")`가 불완전 바이트를 `U+FFFD(□)`로 치환

**Human이 결정한 것:**
- `errors="ignore"` (글자 누락) 대신 사후 필터링 방식 채택
- 생성 결과에서 대체 문자만 제거, 나머지 유지

**AI가 구현한 것:**
- `api/inference.py` generate() 반환 직전에 후처리 1줄 추가
```python
generated_text = generated_text.replace("�", "")
```

---

### 한국어 데이터 확대 + 재학습 (3번의 시도)

#### 시도 1 — 실패: MPS 프리즈 (9시간)

**설정:** `block_size=256, max_iters=15000, vocab_size=4096`

**문제:**
- `prepare_korean.py`에 200K 자 제한이 있어 데이터가 231K 자밖에 안 받아짐 (기존 926K 대비 1/4)
- 데이터 부족 → 즉시 과적합 → val loss가 iter 0(랜덤) 이후 한 번도 개선 안 됨
- 더 심각한 문제: `block_size=256`이 Apple Silicon MPS에서 9시간 데드락

**Human이 직접 판단한 것:**
- 프로세스 상태 `UN (Uninterruptible Sleep)` 확인 후 강제 종료

#### 시도 2 — 실패: tokenizer.json 덮어씌워짐

**문제:** 시도 1로 인해 `data/tokenizer.json` (vocab 2053)이 새 vocab 4101로 덮어씌워짐
→ 기존 백업 모델(`model_5000iter_v1.pt`)과 호환 불가 → 복원 포기, 재학습으로 방향 전환

#### 시도 3 — 성공

**Human이 결정한 것:**
- 복원 vs 재학습 트레이드오프 판단: 어차피 tokenizer 재생성 필요 → 더 나은 조건으로 재학습
- `block_size=256` 원인 파악 후 `128`로 복구
- HF_TOKEN 환경변수 설정 (rate limit 해제)
- `~/.zshrc`에 저장 (영구 적용)

**AI가 수정한 것:**
- `prepare_korean.py`: `trust_remote_code=True` 제거 (HF API 변경 대응)
- `prepare_korean.py`: 다운로드 상한 `200_000 → 2_000_000` (10배 확대)

**최종 학습 설정:**
```
--max-iters 10000 --n-layer 6 --n-head 6 --n-embd 384 --block-size 128 --batch-size 32
vocab_size: 4096  |  데이터: 382개 문서, 2,004,756자  |  train 토큰: 1,039,157
```

**결과:**
```
iter  2500  val 4.82
iter  3500  val 4.80  ← best
iter  4000  val 4.80  ← checkpoint 저장 (12:12)
iter  4500  val 4.87  과적합 시작
iter 10000  val 5.10  (best는 iter 4000)
```
→ `best val loss: 4.8015` (vocab 4096 기준, 이론 최솟값 ln(4101) ≈ 8.32)

**Human이 판단한 것:**
- 과적합 시작 지점(iter ~4000) 이후 checkpoint 업데이트 없음 → 조기 종료 필요성 인식
- 다음 학습 시 `--max-iters 5000` 또는 early stopping 추가 예정

---

**보류 중 (다음 세션):**
- Early stopping 구현 (현재 best val 이후도 계속 돌아감)
- 배포: Vercel (Next.js) + HuggingFace Spaces (FastAPI)
