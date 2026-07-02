import React, { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, TrendingUp, AlertTriangle, Brain,
  Settings, Zap, Bell, User, ChevronDown,
  Activity, LogOut, ChevronRight, Upload, Shield,
  Sun, Moon,
} from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import { useTheme } from '../contexts/ThemeContext'
import { cn } from '../utils/format'

const navItems = [
  { to: '/dashboard',      icon: LayoutDashboard, label: 'Dashboard',      roles: null },
  { to: '/forecast',       icon: TrendingUp,      label: 'Forecast',       roles: null },
  { to: '/anomalies',      icon: AlertTriangle,   label: 'Anomalies',      roles: null },
  { to: '/explainability', icon: Brain,            label: 'Explainability', roles: null },
  { to: '/optimization',   icon: Zap,             label: 'Optimization',   roles: null },
  { to: '/upload',         icon: Upload,          label: 'Upload Data',    roles: null },
  { to: '/admin',          icon: Shield,          label: 'Admin',          roles: ['admin'] as string[] },
  { to: '/settings',       icon: Settings,        label: 'Settings',       roles: null },
]

const AppLayout: React.FC = () => {
  const { user, logout } = useAuth()
  const { theme, toggleTheme } = useTheme()
  const navigate = useNavigate()
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [profileOpen, setProfileOpen] = useState(false)

  const handleLogout = () => {
    logout()
    navigate('/login')
  }

  return (
    <div className="flex h-screen bg-transparent text-text-primary overflow-hidden grid-bg">
      {/* ── Sidebar ─────────────────────────────────────────────────────── */}
      <motion.aside
        animate={{ width: sidebarOpen ? 260 : 72 }}
        transition={{ duration: 0.25, ease: 'easeInOut' }}
        className="relative flex-shrink-0 bg-bg-secondary border-r border-bg-border flex flex-col z-20"
      >
        {/* Logo area */}
        <div className="flex items-center gap-3 px-4 h-16 border-b border-bg-border flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-electric-gradient flex items-center justify-center shadow-glow-blue flex-shrink-0">
            <Zap size={16} className="text-white" />
          </div>
          <AnimatePresence>
            {sidebarOpen && (
              <motion.span
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -10 }}
                transition={{ duration: 0.2 }}
                className="font-display font-bold text-text-primary text-lg whitespace-nowrap"
              >
                EnerVision <span className="text-electric-400">AI</span>
              </motion.span>
            )}
          </AnimatePresence>
        </div>

        {/* Nav items */}
        <nav className="flex-1 p-3 space-y-1 overflow-y-auto overflow-x-hidden">
          {navItems.filter(item => !item.roles || item.roles.includes(user?.role ?? '')).map(({ to, icon: Icon, label }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                cn('nav-item group relative', isActive && 'active')
              }
            >
              {({ isActive }) => (
                <>
                  {isActive && (
                    <motion.div
                      layoutId="activeNav"
                      className={cn(
                        "absolute inset-0 border rounded-lg",
                        theme === 'light' 
                          ? 'bg-electric-500/10 border-electric-500/20 shadow-sm' 
                          : 'bg-electric-600/20 border-electric-500/30'
                      )}
                      transition={{ duration: 0.2 }}
                    />
                  )}
                  <Icon 
                    size={18} 
                    className={cn(
                      'flex-shrink-0 relative z-10', 
                      isActive 
                        ? (theme === 'light' ? 'text-electric-600' : 'text-electric-400') 
                        : 'text-text-muted group-hover:text-text-primary'
                    )} 
                  />
                  <AnimatePresence>
                    {sidebarOpen && (
                      <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className={cn(
                          'relative z-10 whitespace-nowrap',
                          isActive 
                            ? (theme === 'light' ? 'text-electric-600 font-semibold' : 'text-text-primary font-medium') 
                            : 'text-text-secondary'
                        )}
                      >
                        {label}
                      </motion.span>
                    )}
                  </AnimatePresence>
                </>
              )}
            </NavLink>
          ))}
        </nav>

        {/* User section */}
        <div className="p-3 border-t border-bg-border flex-shrink-0">
          <button
            onClick={() => setProfileOpen(p => !p)}
            className={cn('nav-item w-full', profileOpen && 'bg-bg-hover')}
          >
            <div className="w-7 h-7 rounded-full bg-electric-gradient flex items-center justify-center flex-shrink-0 text-white text-xs font-semibold">
              {user?.full_name?.charAt(0) ?? 'U'}
            </div>
            <AnimatePresence>
              {sidebarOpen && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex-1 flex items-center justify-between min-w-0"
                >
                  <div className="min-w-0 text-left">
                    <p className="text-sm text-text-primary font-medium truncate">{user?.full_name ?? 'User'}</p>
                    <p className="text-xs text-text-muted truncate capitalize">{user?.role ?? 'viewer'}</p>
                  </div>
                  <ChevronDown size={14} className={cn('text-text-muted transition-transform', profileOpen && 'rotate-180')} />
                </motion.div>
              )}
            </AnimatePresence>
          </button>

          <AnimatePresence>
            {profileOpen && sidebarOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-1 space-y-0.5"
              >
                <button onClick={() => navigate('/settings')} className="nav-item w-full text-text-secondary">
                  <User size={14} /> Profile
                </button>
                <button onClick={handleLogout} className="nav-item w-full text-danger-400 hover:text-danger-300 hover:bg-danger-500/10">
                  <LogOut size={14} /> Sign out
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarOpen(p => !p)}
          className="absolute -right-3 top-20 w-6 h-6 bg-bg-card border border-bg-border rounded-full flex items-center justify-center text-text-muted hover:text-text-primary hover:border-electric-500/40 transition-all z-30"
        >
          {sidebarOpen ? <ChevronRight size={12} /> : <ChevronRight size={12} className="rotate-180" />}
        </button>
      </motion.aside>

      {/* ── Main area ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative z-10">
        {/* Topbar */}
        <header className="h-16 bg-bg-secondary border-b border-bg-border flex items-center justify-between px-6 flex-shrink-0 z-10">
          <div className="flex items-center gap-3">
            <Activity size={16} className="text-success-500 animate-pulse" />
            <span className="text-xs text-text-muted font-mono">API: Connected</span>
          </div>

          <div className="flex items-center gap-3">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              className="w-8 h-8 flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-bg-hover rounded-lg transition-all"
              title={theme === 'gradient' ? 'Switch to Light Mode' : 'Switch to Gradient Mode'}
            >
              {theme === 'gradient' ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <div className="h-6 w-px bg-bg-border" />
            <button className="relative w-8 h-8 flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-bg-hover rounded-lg transition-all">
              <Bell size={16} />
              <span className="absolute top-1 right-1 w-2 h-2 bg-danger-500 rounded-full" />
            </button>
            <div className="h-6 w-px bg-bg-border" />
            <div className="flex items-center gap-2">
              <div className="w-7 h-7 rounded-full bg-electric-gradient flex items-center justify-center text-white text-xs font-semibold">
                {user?.full_name?.charAt(0) ?? 'U'}
              </div>
              <span className="text-sm text-text-secondary font-medium hidden sm:block">{user?.full_name}</span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto bg-bg-primary/40">
          <div className="animate-fade-in">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default AppLayout
