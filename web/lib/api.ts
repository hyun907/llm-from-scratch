export interface GenerateParams {
  prompt: string
  max_tokens: number
  temperature: number
  top_k: number
}

export interface GenerateResult {
  generated: string
  tokens_generated: number
}

export async function generateText(params: GenerateParams): Promise<GenerateResult> {
  const res = await fetch('/api/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(params),
  })

  if (!res.ok) {
    const { error } = await res.json().catch(() => ({ error: '알 수 없는 오류' }))
    throw new Error(error ?? `HTTP ${res.status}`)
  }

  return res.json()
}
