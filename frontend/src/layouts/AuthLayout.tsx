import React from 'react'
import { Outlet } from 'react-router-dom'

const AuthLayout: React.FC = () => (
  <div className="min-h-screen grid-bg flex items-center justify-center relative overflow-hidden py-12">
    {/* Orbiting Ambient glow blobs */}
    <div className="absolute top-1/6 left-1/6 w-96 h-96 bg-electric-500/15 rounded-full blur-[100px] pointer-events-none animate-orbit" />
    <div className="absolute bottom-1/6 right-1/6 w-96 h-96 bg-violet-500/15 rounded-full blur-[100px] pointer-events-none animate-orbit" style={{ animationDelay: '-5s' }} />

    <div className="relative z-10 w-full max-w-md px-6">
      {/* Logo */}
      <div className="text-center mb-8">
        <div className="inline-flex items-center gap-3 mb-3">
          <div className="w-11 h-11 rounded-xl bg-electric-gradient flex items-center justify-center shadow-glow-blue transition-transform hover:scale-105 duration-300">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <span className="text-3xl font-display font-bold text-text-primary tracking-tight">
            EnerVision <span className="text-electric-400">AI</span>
          </span>
        </div>
        <p className="text-text-muted text-xs font-medium uppercase tracking-wider">Energy Intelligence Platform</p>
      </div>
      <div className="shadow-2xl rounded-2xl">
        <Outlet />
      </div>
    </div>
  </div>
)

export default AuthLayout
