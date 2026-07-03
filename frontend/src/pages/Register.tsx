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
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-8"
    >
      <h2 className="text-xl font-display font-semibold text-text-primary mb-1">Create account</h2>
      <p className="text-sm text-text-secondary mb-6">Join the EnerVision AI platform</p>

      {error && (
        <div className="flex items-center gap-2 bg-danger-500/10 border border-danger-500/30 rounded-lg px-3 py-2.5 mb-4">
          <AlertCircle size={14} className="text-danger-400 flex-shrink-0" />
          <p className="text-sm text-danger-400">{error}</p>
        </div>
      )}

      {success && (
        <div className="flex items-center gap-2 bg-success-500/10 border border-success-500/30 rounded-lg px-3 py-2.5 mb-4">
          <CheckCircle size={14} className="text-success-400 flex-shrink-0" />
          <p className="text-sm text-success-400">Account created! Redirecting to login…</p>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="label block mb-1.5">Full name</label>
          <input
            name="full_name" type="text" required value={form.full_name} onChange={handleChange}
            placeholder="Jane Smith"
            className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-electric-500/60 transition-colors"
          />
        </div>

        <div>
          <label className="label block mb-1.5">Email</label>
          <input
            name="email" type="email" required value={form.email} onChange={handleChange}
            placeholder="jane@example.com"
            className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 text-sm text-text-primary placeholder:text-text-muted focus:border-electric-500/60 transition-colors"
          />
        </div>

        <div>
          <label className="label block mb-1.5">Password</label>
          <div className="relative">
            <input
              name="password" type={showPw ? 'text' : 'password'} required value={form.password} onChange={handleChange}
              placeholder="Min. 8 characters"
              className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 pr-10 text-sm text-text-primary placeholder:text-text-muted focus:border-electric-500/60 transition-colors"
            />
            <button type="button" onClick={() => setShowPw(p => !p)}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-secondary">
              {showPw ? <EyeOff size={16} /> : <Eye size={16} />}
            </button>
          </div>
        </div>

        <div>
          <label className="label block mb-1.5">Role</label>
          <select
            name="role" value={form.role} onChange={handleChange}
            className="w-full bg-bg-primary border border-bg-border rounded-lg px-3 py-2.5 text-sm text-text-primary focus:border-electric-500/60 transition-colors"
          >
            <option value="viewer">Viewer</option>
            <option value="analyst">Analyst</option>
            <option value="admin">Admin</option>
          </select>
        </div>

        <button type="submit" disabled={loading || success}
          className="btn-primary w-full flex items-center justify-center gap-2 mt-2">
          {loading ? <Spinner size={16} /> : <Zap size={16} />}
          {loading ? 'Creating account…' : 'Create Account'}
        </button>
      </form>

      <p className="text-center text-sm text-text-muted mt-6">
        Already have an account?{' '}
        <Link to="/login" className="text-electric-400 hover:text-electric-300 font-medium">Sign in</Link>
      </p>
    </motion.div>
  )
}

export default Register
