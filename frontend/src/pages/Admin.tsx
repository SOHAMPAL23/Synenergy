import React, { useCallback, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Users, Database, Activity, CheckCircle, XCircle,
  RefreshCw, ChevronDown, AlertTriangle, BarChart2, Zap,
  TrendingUp, UserCheck, UserX,
} from 'lucide-react'
import { adminService } from '../services/admin'
import type { UserItem } from '../services/admin'
import PageHeader, { ChartCard, Spinner } from '../components/ui/PageHeader'
import { useAuth } from '../contexts/AuthContext'
import { useNavigate } from 'react-router-dom'

// ── Role badge ────────────────────────────────────────────────────────────────

const RoleBadge: React.FC<{ role: string }> = ({ role }) => {
  const map: Record<string, string> = {
    admin: 'bg-violet-500/10 text-violet-400 border-violet-500/20',
    analyst: 'bg-electric-500/10 text-electric-400 border-electric-500/20',
    viewer: 'bg-bg-hover text-text-muted border-bg-border',
  }
  return (
    <span className={`px-2 py-0.5 rounded-md text-xs font-medium border capitalize ${map[role] ?? map.viewer}`}>
      {role}
    </span>
  )
}

// ── User Row ──────────────────────────────────────────────────────────────────

const UserRow: React.FC<{
  user: UserItem
  currentUserId: string
  onToggle: (id: string, active: boolean) => void
  onChangeRole: (id: string, role: string) => void
  toggling: string | null
}> = ({ user, currentUserId, onToggle, onChangeRole, toggling }) => {
  const [roleOpen, setRoleOpen] = useState(false)
  const isSelf = user.id === currentUserId

  return (
    <tr className="border-b border-bg-border/40 hover:bg-bg-hover/20 transition-colors">
      <td className="py-3 px-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-electric-gradient flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
            {user.full_name.charAt(0).toUpperCase()}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-text-primary truncate">
              {user.full_name}
              {isSelf && <span className="ml-2 text-xs text-electric-400">(you)</span>}
            </p>
            <p className="text-xs text-text-muted truncate">{user.email}</p>
          </div>
        </div>
      </td>

      <td className="py-3 px-4">
        <div className="relative">
          <button
            disabled={isSelf}
            onClick={() => setRoleOpen(o => !o)}
            className="flex items-center gap-1.5 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RoleBadge role={user.role} />
            {!isSelf && <ChevronDown size={12} className="text-text-muted" />}
          </button>
          <AnimatePresence>
            {roleOpen && !isSelf && (
              <motion.div
                initial={{ opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -4 }}
                className="absolute z-50 top-7 left-0 glass-card py-1 min-w-[120px] shadow-xl"
              >
                {(['admin', 'analyst', 'viewer'] as const).map(r => (
                  <button
                    key={r}
                    onClick={() => { onChangeRole(user.id, r); setRoleOpen(false) }}
                    className={`w-full text-left px-3 py-1.5 text-xs capitalize hover:bg-bg-hover transition-colors ${user.role === r ? 'text-electric-400 font-semibold' : 'text-text-secondary'}`}
                  >
                    {r}
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </td>

      <td className="py-3 px-4">
        <span className={`flex items-center gap-1.5 text-xs font-medium ${user.is_active ? 'text-success-500' : 'text-text-muted'}`}>
          {user.is_active
            ? <><CheckCircle size={12} /> Active</>
            : <><XCircle size={12} /> Inactive</>}
        </span>
      </td>

      <td className="py-3 px-4 text-xs text-text-muted">
        {user.created_at ? new Date(user.created_at).toLocaleDateString() : '—'}
      </td>

      <td className="py-3 px-4 text-xs text-text-muted">
        {user.last_login ? new Date(user.last_login).toLocaleDateString() : 'Never'}
      </td>

      <td className="py-3 px-4">
        {!isSelf && (
          <button
            onClick={() => onToggle(user.id, user.is_active)}
            disabled={toggling === user.id}
            className={`flex items-center gap-1.5 text-xs font-medium px-2 py-1 rounded-md transition-all ${
              user.is_active
                ? 'text-danger-500 hover:bg-danger-500/10 border border-danger-500/20'
                : 'text-success-500 hover:bg-success-500/10 border border-success-500/20'
            } disabled:opacity-50`}
          >
            {toggling === user.id
              ? <Spinner size={12} />
              : user.is_active ? <><UserX size={12} /> Deactivate</> : <><UserCheck size={12} /> Activate</>}
          </button>
        )}
      </td>
    </tr>
  )
}

// ── Stat Card ─────────────────────────────────────────────────────────────────

const StatCard: React.FC<{
  icon: React.ReactNode
  label: string
  value: string | number
  color: string
  accent: string
}> = ({ icon, label, value, color, accent }) => (
  <div className={`glass-card p-4 border ${accent}`}>
    <div className="flex items-center gap-2 mb-2">
      <span className={color}>{icon}</span>
      <p className="label text-xs">{label}</p>
    </div>
    <p className={`text-2xl font-display font-bold ${color}`}>{value}</p>
  </div>
)

// ── Main Admin Page ───────────────────────────────────────────────────────────

const Admin: React.FC = () => {
  const { user } = useAuth()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [toggling, setToggling] = useState<string | null>(null)
  const [roleFilter, setRoleFilter] = useState<string>('all')

  // Redirect non-admins
  React.useEffect(() => {
    if (user && user.role !== 'admin') {
      navigate('/dashboard')
    }
  }, [user, navigate])

  const { data: usersData, isLoading: usersLoading, refetch } = useQuery({
    queryKey: ['admin', 'users'],
    queryFn: adminService.listUsers,
    retry: false,
  })

  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['admin', 'stats'],
    queryFn: adminService.getSystemStats,
    retry: false,
  })

  const updateMutation = useMutation({
    mutationFn: ({ id, payload }: { id: string; payload: any }) =>
      adminService.updateUser(id, payload),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['admin', 'users'] }),
  })

  const handleToggle = useCallback(async (id: string, currentlyActive: boolean) => {
    setToggling(id)
    try {
      await updateMutation.mutateAsync({ id, payload: { is_active: !currentlyActive } })
    } finally {
      setToggling(null)
    }
  }, [updateMutation])

  const handleChangeRole = useCallback(async (id: string, role: string) => {
    await updateMutation.mutateAsync({ id, payload: { role } })
  }, [updateMutation])

  const filteredUsers = (usersData?.users ?? []).filter(u =>
    roleFilter === 'all' || u.role === roleFilter
  )

  return (
    <div className="page-container">
      <PageHeader
        title="Admin Panel"
        subtitle="User management and system statistics"
        badge="Admin"
        actions={
          <button
            onClick={() => refetch()}
            className="btn-secondary flex items-center gap-2 text-sm"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        }
      />

      {/* System Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-3">
        {statsLoading
          ? Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="skeleton h-20 rounded-xl" />
            ))
          : [
              { icon: <Users size={16} />, label: 'Total Users', value: stats?.total_users ?? '—', color: 'text-electric-500', accent: 'border-electric-500/20 bg-electric-500/5' },
              { icon: <Activity size={16} />, label: 'Active Users', value: stats?.active_users ?? '—', color: 'text-success-500', accent: 'border-success-500/20 bg-success-500/5' },
              { icon: <Database size={16} />, label: 'Energy Records', value: (stats?.total_energy_records ?? 0).toLocaleString(), color: 'text-cyan-500', accent: 'border-cyan-500/20 bg-cyan-500/5' },
              { icon: <TrendingUp size={16} />, label: 'Forecasts', value: stats?.total_forecasts ?? '—', color: 'text-electric-500', accent: 'border-electric-500/20 bg-electric-500/5' },
              { icon: <Zap size={16} />, label: 'Recommendations', value: stats?.total_recommendations ?? '—', color: 'text-warning-500', accent: 'border-warning-500/20 bg-warning-500/5' },
              { icon: <BarChart2 size={16} />, label: 'Models Trained', value: stats?.models_trained ?? '—', color: 'text-violet-500', accent: 'border-violet-500/20 bg-violet-500/5' },
            ].map(s => <StatCard key={s.label} {...s} />)
        }
      </div>

      {/* User Management Table */}
      <ChartCard
        title="User Management"
        subtitle={`${usersData?.total ?? 0} registered users`}
        actions={
          <div className="flex gap-1 bg-bg-primary border border-bg-border rounded-lg p-1">
            {(['all', 'admin', 'analyst', 'viewer'] as const).map(r => (
              <button
                key={r}
                onClick={() => setRoleFilter(r)}
                className={`px-2.5 py-1 rounded-md text-xs font-medium capitalize transition-all ${
                  roleFilter === r
                    ? 'bg-electric-600 text-white shadow-sm'
                    : 'text-text-muted hover:text-text-primary'
                }`}
              >
                {r}
              </button>
            ))}
          </div>
        }
      >
        {usersLoading ? (
          <div className="flex justify-center py-8">
            <Spinner size={24} />
          </div>
        ) : filteredUsers.length === 0 ? (
          <div className="text-center py-10 text-text-muted text-sm">
            <Users size={32} className="mx-auto mb-2 opacity-30" />
            No users found.
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-bg-border text-left">
                  {['User', 'Role', 'Status', 'Joined', 'Last Login', 'Actions'].map(h => (
                    <th key={h} className="label pb-3 px-4 text-xs font-medium">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map(u => (
                  <UserRow
                    key={u.id}
                    user={u}
                    currentUserId={user?.id ?? ''}
                    onToggle={handleToggle}
                    onChangeRole={handleChangeRole}
                    toggling={toggling}
                  />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </ChartCard>

      {/* Notice for non-existent backend */}
      <div className="glass-card p-4 border border-amber-500/20 bg-amber-500/5 flex items-start gap-3">
        <AlertTriangle size={16} className="text-amber-400 flex-shrink-0 mt-0.5" />
        <div className="text-xs text-text-secondary">
          <p className="font-medium text-warning-600 mb-1">Admin API Endpoints Required</p>
          <p>
            The Admin panel requires <code className="font-mono text-electric-400">GET /api/v1/admin/users</code> and{' '}
            <code className="font-mono text-electric-400">PATCH /api/v1/admin/users/:id</code> endpoints on the backend.
            These can be added to <code className="font-mono text-text-primary">backend/api/routes/admin.py</code>.
          </p>
        </div>
      </div>
    </div>
  )
}

export default Admin
