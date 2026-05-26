import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'LLM From Scratch',
  description: 'GPT-2 스타일 언어 모델 텍스트 생성 인터페이스',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-screen bg-gray-950 text-gray-100 antialiased">
        <header className="border-b border-gray-800 px-6 py-4">
          <h1 className="text-lg font-semibold tracking-tight">LLM From Scratch</h1>
          <p className="text-sm text-gray-400 mt-0.5">GPT-2 스타일 언어 모델 — 직접 만든 모델로 텍스트 생성</p>
        </header>
        <main className="max-w-3xl mx-auto px-6 py-10">
          {children}
        </main>
      </body>
    </html>
  )
}
