import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Zap, AlertCircle } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { Spinner } from '../components/ui/PageHeader'

const Login: React.FC = () => {
  const { login } = useAuth()
  const navigate = useNavigate()
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await login(email, password)
      navigate('/dashboard')
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? 'Invalid email or password.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-8 border border-bg-border/60 shadow-xl"
    >
      <h2 className="text-2xl font-display font-bold text-text-primary mb-1 tracking-tight">Welcome back</h2>
      <p className="text-xs text-text-muted mb-6">Sign in to your EnerVision AI account</p>

      {error && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 bg-danger-500/10 border border-danger-500/30 rounded-xl px-3.5 py-2.5 mb-4"
        >
          <AlertCircle size={14} className="text-danger-400 flex-shrink-0" />
          <p className="text-xs text-danger-400 font-medium">{error}</p>
        </motion.div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Email address</label>
          <input
            type="email"
            required
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="w-full bg-bg-primary/60 border border-bg-border rounded-xl px-3.5 py-2.5 text-xs text-text-primary placeholder:text-text-muted/60 focus:border-electric-500/50 focus:ring-4 focus:ring-electric-500/10 transition-all outline-none"
          />
        </div>

        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Password</label>
          <div className="relative">
            <input
              type={showPw ? 'text' : 'password'}
              required
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full bg-bg-primary/60 border border-bg-border rounded-xl px-3.5 py-2.5 pr-11 text-xs text-text-primary placeholder:text-text-muted/60 focus:border-electric-500/50 focus:ring-4 focus:ring-electric-500/10 transition-all outline-none"
            />
            <button
              type="button"
              onClick={() => setShowPw(p => !p)}
              className="absolute right-3.5 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary cursor-pointer"
            >
              {showPw ? <EyeOff size={15} /> : <Eye size={15} />}
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2 mt-4 text-xs font-bold cursor-pointer"
        >
          {loading ? <Spinner size={14} className="text-white" /> : <Zap size={14} />}
          {loading ? 'Signing in…' : 'Sign In'}
        </button>
      </form>

      <p className="text-center text-xs text-text-muted mt-6">
        Don&apos;t have an account?{' '}
        <Link to="/register" className="text-electric-400 hover:text-electric-300 font-semibold transition-colors">
          Create an account
        </Link>
      </p>
    </motion.div>
  )
}

export default Login
