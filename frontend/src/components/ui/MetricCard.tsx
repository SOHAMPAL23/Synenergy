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
  blue:   { glow: 'hover:shadow-glow-blue',  icon: 'bg-electric-500/20 text-electric-400', dot: 'bg-electric-400' },
  cyan:   { glow: 'hover:shadow-glow-cyan',  icon: 'bg-cyan-500/20 text-cyan-400',         dot: 'bg-cyan-400'     },
  green:  { glow: 'hover:shadow-glow-green', icon: 'bg-success-500/20 text-success-400',   dot: 'bg-success-400'  },
  red:    { glow: 'hover:shadow-glow-red',   icon: 'bg-danger-500/20 text-danger-400',     dot: 'bg-danger-400'   },
  purple: { glow: 'hover:shadow-glow-blue',  icon: 'bg-violet-500/20 text-violet-400',     dot: 'bg-violet-400'   },
  amber:  { glow: 'hover:shadow-glow-red',   icon: 'bg-warning-500/20 text-warning-400',   dot: 'bg-warning-400'  },
}

const MetricCard: React.FC<MetricCardProps> = ({
  title, value, subtitle, icon, trend, accentColor = 'blue', loading = false, className,
}) => {
  const colors = colorMap[accentColor]

  if (loading) {
    return (
      <div className={cn('metric-card', className)}>
        <div className="skeleton h-4 w-24 mb-4" />
        <div className="skeleton h-8 w-32 mb-2" />
        <div className="skeleton h-3 w-20" />
      </div>
    )
  }

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ duration: 0.2 }}
      className={cn('metric-card cursor-default group', colors.glow, className)}
    >
      <div className="flex items-start justify-between mb-4">
        <div className={cn('w-10 h-10 rounded-xl flex items-center justify-center', colors.icon)}>
          {icon}
        </div>
        {trend && (
          <span className={trend.up ? 'trend-up' : 'trend-down'}>
            {trend.up ? '▲' : '▼'} {trend.value}
          </span>
        )}
      </div>
      <p className="label mb-1">{title}</p>
      <p className="stat-value">{typeof value === 'number' ? value.toLocaleString() : value}</p>
      {subtitle && <p className="text-xs text-text-muted mt-1">{subtitle}</p>}
    </motion.div>
  )
}

export default MetricCard
