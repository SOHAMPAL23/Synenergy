import React, { useState } from 'react'
import { NavLink, Outlet, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  LayoutDashboard, TrendingUp, AlertTriangle, Brain,
  Settings, Zap, Bell, User, ChevronDown,
  LogOut, ChevronRight, Upload, Shield,
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
        animate={{ width: sidebarOpen ? 260 : 76 }}
        transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
        className="relative flex-shrink-0 bg-bg-secondary/95 border-r border-bg-border flex flex-col z-20 glass"
      >
        {/* Logo area */}
        <div className="flex items-center gap-3 px-5 h-16 border-b border-bg-border flex-shrink-0">
          <div className="w-8 h-8 rounded-lg bg-electric-gradient flex items-center justify-center shadow-glow-blue flex-shrink-0 transition-transform hover:scale-105 duration-200">
            <Zap size={15} className="text-white" />
          </div>
          <AnimatePresence>
            {sidebarOpen && (
              <motion.span
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -8 }}
                transition={{ duration: 0.2 }}
                className="font-display font-bold text-text-primary text-base tracking-wide whitespace-nowrap"
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
                        "absolute inset-0 rounded-lg",
                        theme === 'light' 
                          ? 'bg-electric-500/10 border-l-2 border-electric-500 shadow-sm' 
                          : 'bg-gradient-to-r from-electric-600/20 to-transparent border-l-2 border-electric-400'
                      )}
                      transition={{ duration: 0.25, ease: 'easeOut' }}
                    />
                  )}
                  <Icon 
                    size={16} 
                    className={cn(
                      'flex-shrink-0 relative z-10 transition-transform duration-200 group-hover:scale-110', 
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
                          'relative z-10 whitespace-nowrap text-xs font-medium tracking-wide',
                          isActive 
                            ? (theme === 'light' ? 'text-electric-600 font-semibold' : 'text-text-primary font-medium') 
                            : 'text-text-secondary font-normal'
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
        <div className="p-3 border-t border-bg-border flex-shrink-0 bg-bg-secondary/40">
          <button
            onClick={() => setProfileOpen(p => !p)}
            className={cn('nav-item w-full flex items-center justify-between', profileOpen && 'bg-bg-hover')}
          >
            <div className="flex items-center gap-2 min-w-0">
              <div className="w-7 h-7 rounded-full bg-electric-gradient flex items-center justify-center flex-shrink-0 text-white text-xs font-bold shadow-sm">
                {user?.full_name?.charAt(0) ?? 'U'}
              </div>
              <AnimatePresence>
                {sidebarOpen && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="min-w-0 text-left"
                  >
                    <p className="text-xs text-text-primary font-semibold truncate leading-tight">{user?.full_name ?? 'User'}</p>
                    <p className="text-[10px] text-text-muted truncate capitalize leading-tight mt-0.5">{user?.role ?? 'viewer'}</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
            {sidebarOpen && (
              <ChevronDown size={12} className={cn('text-text-muted transition-transform duration-200 flex-shrink-0', profileOpen && 'rotate-180')} />
            )}
          </button>

          <AnimatePresence>
            {profileOpen && sidebarOpen && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-1 space-y-0.5 overflow-hidden pl-1"
              >
                <button onClick={() => { setProfileOpen(false); navigate('/settings') }} className="nav-item w-full text-text-secondary text-xs py-1.5">
                  <User size={12} /> Profile settings
                </button>
                <button onClick={handleLogout} className="nav-item w-full text-danger-400 hover:text-danger-300 hover:bg-danger-500/10 text-xs py-1.5">
                  <LogOut size={12} /> Sign out
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Collapse toggle */}
        <button
          onClick={() => setSidebarOpen(p => !p)}
          className="absolute -right-3 top-20 w-6 h-6 bg-bg-card border border-bg-border rounded-full flex items-center justify-center text-text-muted hover:text-text-primary hover:border-electric-500/40 transition-all z-30 shadow-md cursor-pointer hover:scale-105"
        >
          {sidebarOpen ? <ChevronRight size={12} className="rotate-180" /> : <ChevronRight size={12} />}
        </button>
      </motion.aside>

      {/* ── Main area ───────────────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden relative z-10">
        {/* Topbar */}
        <header className="h-16 bg-bg-secondary/80 border-b border-bg-border flex items-center justify-between px-6 flex-shrink-0 z-10 glass">
          <div className="flex items-center gap-3">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2 w-2 bg-success-500"></span>
            </span>
            <span className="text-[10px] text-text-muted font-mono tracking-wider uppercase">API Connected</span>
          </div>

          <div className="flex items-center gap-4">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              className="w-8 h-8 flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-bg-hover rounded-lg transition-all cursor-pointer"
              title={theme === 'gradient' ? 'Switch to Light Mode' : 'Switch to Gradient Mode'}
            >
              {theme === 'gradient' ? <Sun size={15} /> : <Moon size={15} />}
            </button>
            <div className="h-4 w-px bg-bg-border" />
            <button className="relative w-8 h-8 flex items-center justify-center text-text-muted hover:text-text-primary hover:bg-bg-hover rounded-lg transition-all cursor-pointer">
              <Bell size={15} />
              <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-danger-500 rounded-full" />
            </button>
            <div className="h-4 w-px bg-bg-border" />
            <div className="flex items-center gap-2.5 cursor-pointer hover:opacity-90" onClick={() => navigate('/settings')}>
              <div className="w-7 h-7 rounded-full bg-electric-gradient flex items-center justify-center text-white text-xs font-bold shadow-sm">
                {user?.full_name?.charAt(0) ?? 'U'}
              </div>
              <span className="text-xs text-text-secondary font-semibold hidden sm:block">{user?.full_name}</span>
            </div>
          </div>
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-y-auto bg-bg-primary/20">
          <div className="animate-fade-in">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}

export default AppLayout
