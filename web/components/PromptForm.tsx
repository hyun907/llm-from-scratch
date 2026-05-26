'use client'

import { useState } from 'react'
import type { GenerateParams } from '@/lib/api'

interface Props {
  onSubmit: (params: GenerateParams) => void
  isLoading: boolean
}

export default function PromptForm({ onSubmit, isLoading }: Props) {
  const [prompt, setPrompt] = useState('')
  const [maxTokens, setMaxTokens] = useState(100)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(40)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!prompt.trim() || isLoading) return
    onSubmit({ prompt, max_tokens: maxTokens, temperature, top_k: topK })
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* 프롬프트 입력 */}
      <div className="space-y-2">
        <label className="text-sm font-medium text-gray-300">프롬프트</label>
        <textarea
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="텍스트를 입력하면 모델이 이어서 생성합니다..."
          rows={4}
          className="w-full rounded-lg bg-gray-900 border border-gray-700 px-4 py-3 text-sm
                     text-gray-100 placeholder-gray-500 resize-none
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      {/* 옵션 */}
      <div className="grid grid-cols-3 gap-6">
        <SliderField
          label="Max Tokens"
          value={maxTokens}
          min={10} max={500} step={10}
          onChange={setMaxTokens}
          hint="생성할 최대 토큰 수"
        />
        <SliderField
          label={`Temperature: ${temperature.toFixed(1)}`}
          value={temperature}
          min={0.1} max={2.0} step={0.1}
          onChange={setTemperature}
          hint="높을수록 다양하게, 낮을수록 보수적으로"
        />
        <SliderField
          label={`Top-K: ${topK}`}
          value={topK}
          min={1} max={100} step={1}
          onChange={setTopK}
          hint="상위 K개 토큰 중에서만 샘플링"
        />
      </div>

      {/* 제출 */}
      <button
        type="submit"
        disabled={!prompt.trim() || isLoading}
        className="w-full py-3 rounded-lg bg-blue-600 hover:bg-blue-500 disabled:opacity-40
                   disabled:cursor-not-allowed text-sm font-medium transition-colors"
      >
        {isLoading ? '생성 중...' : '텍스트 생성'}
      </button>
    </form>
  )
}

// ---- 내부 컴포넌트 ----

interface SliderFieldProps {
  label: string
  value: number
  min: number
  max: number
  step: number
  onChange: (v: number) => void
  hint: string
}

function SliderField({ label, value, min, max, step, onChange, hint }: SliderFieldProps) {
  return (
    <div className="space-y-2">
      <label className="text-sm font-medium text-gray-300">{label}</label>
      <input
        type="range"
        min={min} max={max} step={step}
        value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full accent-blue-500"
      />
      <p className="text-xs text-gray-500">{hint}</p>
    </div>
  )
}
