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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-8"
    >
      <h2 className="text-xl font-display font-semibold text-text-primary mb-1">Welcome back</h2>
      <p className="text-sm text-text-secondary mb-6">Sign in to your EnerVision AI account</p>

      {error && (
        <div className="flex items-center gap-2 bg-danger-500/10 border border-danger-500/30 rounded-lg px-3 py-2.5 mb-4">
          <AlertCircle size={14} className="text-danger-400 flex-shrink-0" />
          <p className="text-sm text-danger-400">{error}</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="label block mb-1.5">Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-electric-500/60 transition-colors"
          />
        </div>

        <div>
          <label className="label block mb-1.5">Password</label>
          <div className="relative">
            <input
              type={showPw ? 'text' : 'password'}
              required
              value={password}
              onChange={e => setPassword(e.target.value)}
              placeholder="••••••••"
              className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 pr-10 text-sm text-text-primary placeholder:text-text-muted focus:border-electric-500/60 transition-colors"
            />
            <button
              type="button"
              onClick={() => setShowPw(p => !p)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary"
            >
              {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="btn-primary w-full flex items-center justify-center gap-2 mt-2"
        >
          {loading ? <Spinner size={16} /> : <Zap size={16} />}
          {loading ? 'Signing in…' : 'Sign In'}
        </button>
      </form>

      <p className="text-center text-sm text-text-muted mt-6">
        Don&apos;t have an account?{' '}
        <Link to="/register" className="text-electric-400 hover:text-electric-300 font-medium">
          Sign up
        </Link>
      </p>
    </motion.div>
  )
}

export default Login
