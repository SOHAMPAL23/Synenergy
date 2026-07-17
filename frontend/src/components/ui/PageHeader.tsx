import React from 'react'
import { cn } from '../../utils/format'

interface PageHeaderProps {
  title: string
  subtitle?: string
  actions?: React.ReactNode
  badge?: string
}

const PageHeader: React.FC<PageHeaderProps> = ({ title, subtitle, actions, badge }) => (
  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
    <div>
      <div className="flex items-center gap-3">
        <h1 className="text-2xl sm:text-3xl font-display font-extrabold text-text-primary tracking-tight">{title}</h1>
        {badge && (
          <span className="badge-info text-[9px] font-bold py-0.5 px-2 rounded-md uppercase tracking-wider bg-electric-500/10 text-electric-400 border border-electric-500/30 shadow-sm animate-pulse-slow">
            {badge}
          </span>
        )}
      </div>
      {subtitle && <p className="text-xs sm:text-sm text-text-muted mt-1.5 leading-relaxed font-medium">{subtitle}</p>}
    </div>
    {actions && <div className="flex items-center gap-2 flex-wrap sm:flex-nowrap">{actions}</div>}
  </div>
)

interface ChartCardProps {
  title: string
  subtitle?: string
  children: React.ReactNode
  className?: string
  actions?: React.ReactNode
  loading?: boolean
}

export const ChartCard: React.FC<ChartCardProps> = ({
  title, subtitle, children, className, actions, loading,
}) => (
  <div className={cn('glass-card p-6 shadow-lg border border-bg-border/60', className)}>
    <div className="flex items-center justify-between mb-5">
      <div>
        <h3 className="text-xs font-bold text-text-primary uppercase tracking-wider">{title}</h3>
        {subtitle && <p className="text-[10px] text-text-muted mt-1 leading-relaxed">{subtitle}</p>}
      </div>
      {actions}
    </div>
    {loading ? (
      <div className="space-y-3">
        <div className="skeleton h-[200px] w-full rounded-xl" />
      </div>
    ) : children}
  </div>
)

export const Spinner: React.FC<{ size?: number; className?: string }> = ({ size = 20, className }) => (
  <svg
    className={cn('animate-spin text-electric-500', className)}
    style={{ width: size, height: size }}
    fill="none"
    viewBox="0 0 24 24"
  >
    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
  </svg>
)

export const EmptyState: React.FC<{ icon?: React.ReactNode; title: string; subtitle?: string }> = ({
  icon, title, subtitle,
}) => (
  <div className="flex flex-col items-center justify-center py-16 text-center">
    {icon && <div className="mb-3 text-text-muted">{icon}</div>}
    <p className="text-text-secondary font-medium">{title}</p>
    {subtitle && <p className="text-sm text-text-muted mt-1">{subtitle}</p>}
  </div>
)

export default PageHeader
