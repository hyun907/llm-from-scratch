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

---

> 이후 작업은 이 파일에 계속 추가합니다.
