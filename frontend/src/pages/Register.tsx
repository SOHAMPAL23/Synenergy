import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Eye, EyeOff, Zap, AlertCircle, CheckCircle } from 'lucide-react'
import { authService } from '../services/auth'
import { Spinner } from '../components/ui/PageHeader'

const Register: React.FC = () => {
  const navigate = useNavigate()
  const [form, setForm] = useState({ full_name: '', email: '', password: '', role: 'viewer' })
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState(false)

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
    setForm(f => ({ ...f, [e.target.name]: e.target.value }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      await authService.register(form)
      setSuccess(true)
      setTimeout(() => navigate('/login'), 2000)
    } catch (err: any) {
      setError(err?.response?.data?.detail ?? 'Registration failed. Please try again.')
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
      <h2 className="text-2xl font-display font-bold text-text-primary mb-1 tracking-tight">Create account</h2>
      <p className="text-xs text-text-muted mb-6">Join the EnerVision AI platform</p>

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

      {success && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 bg-success-500/10 border border-success-500/30 rounded-xl px-3.5 py-2.5 mb-4"
        >
          <CheckCircle size={14} className="text-success-400 flex-shrink-0" />
          <p className="text-xs text-success-400 font-medium">Account created! Redirecting to login…</p>
        </motion.div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Full name</label>
          <input
            name="full_name"
            type="text"
            required
            value={form.full_name}
            onChange={handleChange}
            placeholder="Jane Smith"
            className="w-full bg-bg-primary/60 border border-bg-border rounded-xl px-3.5 py-2.5 text-xs text-text-primary placeholder:text-text-muted/60 focus:border-electric-500/50 focus:ring-4 focus:ring-electric-500/10 transition-all outline-none"
          />
        </div>

        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Email address</label>
          <input
            name="email"
            type="email"
            required
            value={form.email}
            onChange={handleChange}
            placeholder="jane@example.com"
            className="w-full bg-bg-primary/60 border border-bg-border rounded-xl px-3.5 py-2.5 text-xs text-text-primary placeholder:text-text-muted/60 focus:border-electric-500/50 focus:ring-4 focus:ring-electric-500/10 transition-all outline-none"
          />
        </div>

        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Password</label>
          <div className="relative">
            <input
              name="password"
              type={showPw ? 'text' : 'password'}
              required
              value={form.password}
              onChange={handleChange}
              placeholder="Min. 8 characters"
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

        <div>
          <label className="label block mb-1.5 font-semibold text-text-secondary">Role</label>
          <select
            name="role"
            value={form.role}
            onChange={handleChange}
            className="w-full bg-bg-primary/60 border border-bg-border rounded-xl px-3.5 py-2.5 text-xs text-text-primary focus:border-electric-500/50 focus:ring-4 focus:ring-electric-500/10 transition-all outline-none cursor-pointer"
          >
            <option value="viewer" className="bg-bg-secondary text-text-primary text-xs">Viewer</option>
            <option value="analyst" className="bg-bg-secondary text-text-primary text-xs">Analyst</option>
            <option value="admin" className="bg-bg-secondary text-text-primary text-xs">Admin</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading || success}
          className="btn-primary w-full flex items-center justify-center gap-2 mt-4 text-xs font-bold cursor-pointer"
        >
          {loading ? <Spinner size={14} className="text-white" /> : <Zap size={14} />}
          {loading ? 'Creating account…' : 'Create Account'}
        </button>
      </form>

      <p className="text-center text-xs text-text-muted mt-6">
        Already have an account?{' '}
        <Link to="/login" className="text-electric-400 hover:text-electric-300 font-semibold transition-colors">
          Sign in
        </Link>
      </p>
    </motion.div>
  )
}

export default Register
