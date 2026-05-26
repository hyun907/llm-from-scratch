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
