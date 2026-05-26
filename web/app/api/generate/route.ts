import { NextRequest, NextResponse } from 'next/server'

const FASTAPI_URL = process.env.FASTAPI_URL ?? 'http://localhost:8000'

export async function POST(req: NextRequest) {
  let body: unknown
  try {
    body = await req.json()
  } catch {
    return NextResponse.json({ error: '요청 본문이 올바르지 않습니다.' }, { status: 400 })
  }

  try {
    const upstream = await fetch(`${FASTAPI_URL}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
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
