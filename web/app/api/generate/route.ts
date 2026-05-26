import { NextRequest, NextResponse } from 'next/server'

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000'

interface GenerateBody {
  prompt: string
  max_tokens: number
  temperature: number
  top_k: number
}

function isValidBody(v: unknown): v is GenerateBody {
  if (typeof v !== 'object' || v === null) return false
  const b = v as Record<string, unknown>
  return (
    typeof b.prompt === 'string' &&
    typeof b.max_tokens === 'number' &&
    typeof b.temperature === 'number' &&
    typeof b.top_k === 'number'
  )
}

export async function POST(req: NextRequest) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '요청 본문이 올바르지 않습니다.' }, { status: 400 })
  }

  if (!isValidBody(body)) {
    return NextResponse.json({ error: '필수 필드가 누락되었거나 타입이 올바르지 않습니다.' }, { status: 422 })
  }

  try {
    const upstream = await fetch(`${FASTAPI_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(60_000),
    })

    const data = await upstream.json()

    if (!upstream.ok) {
      return NextResponse.json(
        { error: data.detail ?? 'FastAPI 오류' },
        { status: upstream.status },
      )
    }

    return NextResponse.json(data)
  } catch {
    return NextResponse.json(
      { error: 'FastAPI 서버에 연결할 수 없습니다. api/ 서버가 실행 중인지 확인하세요.' },
      { status: 503 },
    )
  }
}
