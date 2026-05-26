'use client'

interface Props {
  generated: string | null
  tokensGenerated: number | null
  isLoading: boolean
  error: string | null
}

export default function GeneratedOutput({ generated, tokensGenerated, isLoading, error }: Props) {
  if (isLoading) {
    return (
      <div className="rounded-lg bg-gray-900 border border-gray-700 px-4 py-6 text-center">
        <div className="inline-block w-5 h-5 border-2 border-blue-500 border-t-transparent
                        rounded-full animate-spin" />
        <p className="mt-3 text-sm text-gray-400">모델이 텍스트를 생성하는 중...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-lg bg-red-950 border border-red-800 px-4 py-4">
        <p className="text-sm text-red-300">{error}</p>
      </div>
    )
  }

  if (!generated) return null

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-300">생성 결과</h2>
        {tokensGenerated !== null && (
          <span className="text-xs text-gray-500">{tokensGenerated} 토큰 생성됨</span>
        )}
      </div>
      <div className="rounded-lg bg-gray-900 border border-gray-700 px-4 py-4">
        <p className="text-sm text-gray-100 whitespace-pre-wrap leading-relaxed">{generated}</p>
      </div>
    </div>
  )
}
