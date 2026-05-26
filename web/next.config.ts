import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  // FastAPI 서버 주소는 환경변수로 관리
  env: {
    FASTAPI_URL: process.env.FASTAPI_URL ?? 'http://localhost:8000',
  },
}

export default nextConfig
