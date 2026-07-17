import React from 'react'
import { motion } from 'framer-motion'
import { cn } from '../../utils/format'

interface MetricCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: React.ReactNode
  trend?: { value: string; up: boolean }
  accentColor?: 'blue' | 'cyan' | 'green' | 'red' | 'purple' | 'amber'
  loading?: boolean
  className?: string
}

const colorMap = {
  blue:   { glow: 'hover:shadow-glow-blue',  icon: 'bg-electric-500/10 text-electric-400 border-electric-500/20', accent: 'bg-electric-500' },
  cyan:   { glow: 'hover:shadow-glow-cyan',  icon: 'bg-cyan-500/10 text-cyan-400 border-cyan-500/20',         accent: 'bg-cyan-400'     },
  green:  { glow: 'hover:shadow-glow-green', icon: 'bg-success-500/10 text-success-400 border-success-500/20',   accent: 'bg-success-400'  },
  red:    { glow: 'hover:shadow-glow-red',   icon: 'bg-danger-500/10 text-danger-400 border-danger-500/20',     accent: 'bg-danger-400'   },
  purple: { glow: 'hover:shadow-glow-blue',  icon: 'bg-violet-500/10 text-violet-400 border-violet-500/20',     accent: 'bg-violet-400'   },
  amber:  { glow: 'hover:shadow-glow-red',   icon: 'bg-warning-500/10 text-warning-400 border-warning-500/20',   accent: 'bg-warning-400'  },
}

const MetricCard: React.FC<MetricCardProps> = ({
  title, value, subtitle, icon, trend, accentColor = 'blue', loading = false, className,
}) => {
  const colors = colorMap[accentColor]

  if (loading) {
    return (
      <div className={cn('metric-card', className)}>
        <div className="skeleton h-3.5 w-20 mb-4" />
        <div className="skeleton h-8 w-28 mb-2" />
        <div className="skeleton h-3 w-16" />
      </div>
    )
  }

  return (
    <motion.div
      whileHover={{ y: -3 }}
      transition={{ duration: 0.25, ease: 'easeOut' }}
      className={cn('metric-card cursor-default group border border-bg-border/60 hover:border-bg-border/90 shadow-md', colors.glow, className)}
    >
      <div 
        className="absolute top-0 left-0 w-full h-[2px] opacity-0 group-hover:opacity-100 transition-opacity duration-300"
        style={{ background: `linear-gradient(90deg, transparent, ${colorMap[accentColor].accent}, transparent)` }}
      />
      <div className="flex items-center justify-between mb-4">
        <div className={cn('w-9 h-9 rounded-xl flex items-center justify-center border shadow-sm', colors.icon)}>
          {icon}
        </div>
        {trend && (
          <span className={cn('text-xs font-semibold px-2 py-0.5 rounded-lg', trend.up ? 'bg-success-500/10 text-success-400' : 'bg-danger-500/10 text-danger-400')}>
            {trend.up ? '▲' : '▼'} {trend.value}
          </span>
        )}
      </div>
      <p className="label text-[10px] mb-1 font-semibold tracking-wider">{title}</p>
      <p className="stat-value tracking-tight leading-tight">{typeof value === 'number' ? value.toLocaleString() : value}</p>
      {subtitle && <p className="text-[10px] text-text-muted mt-1 font-medium">{subtitle}</p>}
    </motion.div>
  )
}

export default MetricCard
