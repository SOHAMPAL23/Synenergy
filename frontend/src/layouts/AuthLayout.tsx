import React from 'react'
import { Outlet } from 'react-router-dom'

const AuthLayout: React.FC = () => (
  <div className="min-h-screen grid-bg flex items-center justify-center relative overflow-hidden">
    {/* Ambient glow blobs */}
    <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-electric-600/10 rounded-full blur-3xl pointer-events-none" />
    <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

    <div className="relative z-10 w-full max-w-md px-4">
      {/* Logo */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-2 mb-4">
          <div className="w-10 h-10 rounded-xl bg-electric-gradient flex items-center justify-center shadow-glow-blue">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <span className="text-2xl font-display font-bold text-text-primary tracking-tight">EnerVision <span className="text-electric-400">AI</span></span>
        </div>
        <p className="text-text-secondary text-sm">Energy Intelligence Platform</p>
      </div>
      <Outlet />
    </div>
  </div>
)

export default AuthLayout
