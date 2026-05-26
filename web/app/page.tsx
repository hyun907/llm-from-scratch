'use client'

import { useEffect, useRef, useState } from 'react'
import PromptForm from '@/components/PromptForm'
import GeneratedOutput from '@/components/GeneratedOutput'
import { generateText, type GenerateParams } from '@/lib/api'

export default function Home() {
  const [isLoading, setIsLoading] = useState(false)
  const [generated, setGenerated] = useState<string | null>(null)
  const [tokensGenerated, setTokensGenerated] = useState<number | null>(null)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  // 언마운트 시 진행 중인 요청 취소
  useEffect(() => {
    return () => { abortRef.current?.abort() }
  }, [])

  async function handleSubmit(params: GenerateParams) {
    // 이전 요청 취소 후 새 컨트롤러 생성
    abortRef.current?.abort()
    abortRef.current = new AbortController()

    setIsLoading(true)
    setError(null)
    setGenerated(null)
    setTokensGenerated(null)

    try {
      const result = await generateText(params, abortRef.current.signal)
      setGenerated(result.generated)
      setTokensGenerated(result.tokens_generated)
    } catch (e) {
      if (e instanceof DOMException && e.name === 'AbortError') return
      setError(e instanceof Error ? e.message : '알 수 없는 오류가 발생했습니다.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="space-y-8">
      <PromptForm onSubmit={handleSubmit} isLoading={isLoading} />
      <GeneratedOutput
        generated={generated}
        tokensGenerated={tokensGenerated}
        isLoading={isLoading}
        error={error}
      />
    </div>
  )
}
