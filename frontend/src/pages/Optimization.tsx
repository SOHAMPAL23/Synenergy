import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import { BarChart, Bar, Cell, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Zap, Leaf, DollarSign, ChevronDown, ChevronUp } from 'lucide-react'
import { mlService } from '../services/ml'
import type { RecommendationItem } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'
import { getPriorityColor } from '../utils/format'

const PRIORITY_COLORS: Record<string, string> = {
  HIGH: '#ef4444', MEDIUM: '#f59e0b', LOW: '#22c55e',
}

const CARBON_FACTOR = 0.4 // kg CO2 per kWh
const BASELINE_KWH = 50_000 // rough baseline kWh/day for savings calc

const RecCard: React.FC<{
  rec: RecommendationItem
  index: number
  isImplemented: boolean
  onToggleImplement: () => void
}> = ({ rec, index, isImplemented, onToggleImplement }) => {
  const [open, setOpen] = useState(false)
  const estimatedKwh = BASELINE_KWH * (rec.estimated_saving_pct / 100)
  const co2 = (estimatedKwh * CARBON_FACTOR).toFixed(0)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className={`glass-card p-4 transition-all ${
        isImplemented 
          ? 'border-emerald-500/30 bg-emerald-500/5 shadow-sm' 
          : 'hover:border-electric-500/30'
      }`}
    >
      <div className="flex items-start gap-3">
        {/* Workspace implementation toggle */}
        <div className="flex-shrink-0 flex items-center pt-1 pr-1">
          <input 
            type="checkbox" 
            checked={isImplemented} 
            onChange={(e) => {
              e.stopPropagation()
              onToggleImplement()
            }} 
            className="w-4 h-4 rounded border-bg-border bg-bg-primary text-emerald-600 focus:ring-emerald-500 cursor-pointer"
            title={isImplemented ? "Mark as planned" : "Mark as completed"}
          />
        </div>

        <div
          className="flex-1 flex items-start gap-3 cursor-pointer min-w-0"
          onClick={() => setOpen(o => !o)}
        >
          <div
            className="w-1 self-stretch rounded-full flex-shrink-0"
            style={{ background: isImplemented ? '#10b981' : PRIORITY_COLORS[rec.priority] }}
          />
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1">
              {isImplemented ? (
                <span className="badge-low text-[10px] uppercase font-bold py-0.5 px-1.5 flex items-center gap-0.5">
                  ✓ Implemented
                </span>
              ) : (
                <span className={getPriorityColor(rec.priority)}>{rec.priority}</span>
              )}
              <span className="badge-info">{rec.category}</span>
              <span className="text-xs text-success-400 font-semibold">↓ {rec.estimated_saving_pct}%</span>
            </div>
            <h3 className={`text-sm font-semibold truncate ${isImplemented ? 'text-text-secondary line-through opacity-70' : 'text-text-primary'}`}>{rec.title}</h3>
            <p className="text-xs text-text-secondary mt-0.5 line-clamp-2">{rec.description}</p>
          </div>
          <div className="flex-shrink-0 flex items-center gap-3 text-xs text-text-muted select-none">
            <span className="text-success-400 font-medium">{rec.estimated_saving_pct}%</span>
            {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
          </div>
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 pl-8 border-l border-bg-border"
          >
            <div className="grid grid-cols-2 gap-3 mb-3">
              <div className="bg-success-500/10 border border-success-500/20 rounded-lg px-3 py-2">
                <p className="text-xs text-text-muted mb-0.5">Est. Energy Saved</p>
                <p className="text-sm font-semibold text-success-400">{estimatedKwh.toLocaleString()} kWh/day</p>
              </div>
              <div className="bg-cyan-500/10 border border-cyan-500/20 rounded-lg px-3 py-2">
                <p className="text-xs text-text-muted mb-0.5">CO₂ Reduction</p>
                <p className="text-sm font-semibold text-cyan-400">{co2} kg/day</p>
              </div>
            </div>
            <p className="text-xs text-text-muted font-medium mb-2">Action Items:</p>
            <ul className="space-y-1.5">
              {rec.action_items.map((a, i) => (
                <li key={i} className="flex items-start gap-2 text-xs text-text-secondary">
                  <span className="text-electric-400 mt-0.5 flex-shrink-0">→</span>
                  {a}
                </li>
              ))}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

const Optimization: React.FC = () => {
  const { data, isLoading } = useQuery({
    queryKey: ['recommendations'],
    queryFn: mlService.getRecommendations,
  })

  const recs = data?.recommendations ?? []

  const [implementedIds, setImplementedIds] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('implemented_recommendations')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  const toggleImplement = (id: string) => {
    setImplementedIds(prev => {
      const next = prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
      localStorage.setItem('implemented_recommendations', JSON.stringify(next))
      return next
    })
  }

  const totalSavingPct = recs.reduce((s, r) => s + r.estimated_saving_pct, 0) / Math.max(recs.length, 1)
  const totalCO2 = recs.reduce((s, r) => s + BASELINE_KWH * (r.estimated_saving_pct / 100) * CARBON_FACTOR, 0)

  const realizedSavingsKwh = recs
    .filter(r => implementedIds.includes(String(r.id ?? r.title)))
    .reduce((s, r) => s + BASELINE_KWH * (r.estimated_saving_pct / 100), 0)

  const realizedCO2 = realizedSavingsKwh * CARBON_FACTOR
  const realizedDollars = realizedSavingsKwh * 0.12

  const implementedPct = recs.length ? Math.round((implementedIds.length / recs.length) * 100) : 0

  const chartData = recs.map(r => ({
    name: r.title.length > 20 ? r.title.slice(0, 20) + '…' : r.title,
    saving: r.estimated_saving_pct,
    priority: r.priority,
  }))

  return (
    <div className="page-container">
      <PageHeader
        title="Optimization"
        subtitle="AI-powered energy optimization recommendations and estimated savings"
        badge="Active"
      />

      {/* Summary KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        {[
          { label: 'Total Recommendations', value: data?.total ?? 0, icon: <Zap size={16} />, color: 'text-electric-400', accent: 'bg-electric-500/10 border-electric-500/20' },
          { label: 'High Priority', value: data?.high_priority ?? 0, icon: <Zap size={16} />, color: 'text-danger-400', accent: 'bg-danger-500/10 border-danger-500/20' },
          { label: 'Avg. Savings', value: `${totalSavingPct.toFixed(1)}%`, icon: <DollarSign size={16} />, color: 'text-success-400', accent: 'bg-success-500/10 border-success-500/20' },
          { label: 'CO₂ Reduction', value: `${totalCO2.toFixed(0)} kg/d`, icon: <Leaf size={16} />, color: 'text-cyan-400', accent: 'bg-cyan-500/10 border-cyan-500/20' },
        ].map(kpi => (
          <div key={kpi.label} className={`glass-card p-4 border ${kpi.accent}`}>
            <div className="flex items-center gap-2 mb-2">
              <span className={kpi.color}>{kpi.icon}</span>
              <p className="label">{kpi.label}</p>
            </div>
            <p className={`text-2xl font-display font-bold ${kpi.color}`}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Realized Savings Workspace Tracker */}
      <div className="glass-card p-5 border border-emerald-500/30 bg-emerald-500/5 mb-6 flex flex-col md:flex-row items-center justify-between gap-6 animate-slide-up">
        <div className="flex items-center gap-4 flex-1">
          {/* Progress Ring */}
          <div className="relative w-18 h-18 flex items-center justify-center flex-shrink-0">
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="36" cy="36" r="30" stroke="var(--bg-border)" strokeWidth="6" fill="transparent" />
              <circle 
                cx="36" 
                cy="36" 
                r="30" 
                stroke="#10b981" 
                strokeWidth="6" 
                fill="transparent" 
                strokeDasharray={`${2 * Math.PI * 30}`}
                strokeDashoffset={`${2 * Math.PI * 30 * (1 - implementedPct / 100)}`}
                className="transition-all duration-500 ease-out"
              />
            </svg>
            <span className="absolute text-sm font-bold text-text-primary">{implementedPct}%</span>
          </div>
          <div>
            <h4 className="text-sm font-bold text-text-primary flex items-center gap-1.5">
              <Leaf size={14} className="text-emerald-400" />
              Implemented Optimization Workspace
            </h4>
            <p className="text-xs text-text-muted mt-0.5">
              Select recommendation checklists below as they are applied to track energy efficiency savings and greenhouse gas offsets.
            </p>
            <div className="flex gap-4 mt-2 flex-wrap">
              <span className="text-xs text-text-muted">
                Applied: <strong className="text-text-primary">{implementedIds.length}</strong> of {recs.length}
              </span>
              <span className="text-xs text-text-muted">
                Status: <strong className={implementedPct > 50 ? 'text-emerald-400 font-semibold' : implementedPct > 0 ? 'text-warning-400 font-semibold' : 'text-text-muted font-normal'}>
                  {implementedPct === 100 ? 'Fully Optimized' : implementedPct > 0 ? 'Active Deployment' : 'Pending Action'}
                </strong>
              </span>
            </div>
          </div>
        </div>

        <div className="flex gap-4 w-full md:w-auto flex-wrap md:flex-nowrap justify-between border-t md:border-t-0 md:border-l border-bg-border/60 pt-4 md:pt-0 md:pl-6">
          <div className="min-w-[100px]">
            <p className="text-[10px] text-text-muted mb-0.5">Realized Power Saved</p>
            <p className="text-sm font-bold text-emerald-400">{Math.round(realizedSavingsKwh).toLocaleString()} kWh/d</p>
          </div>
          <div className="min-w-[100px]">
            <p className="text-[10px] text-text-muted mb-0.5">Realized CO₂ Reduced</p>
            <p className="text-sm font-bold text-cyan-400">{Math.round(realizedCO2).toLocaleString()} kg/d</p>
          </div>
          <div className="min-w-[100px]">
            <p className="text-[10px] text-text-muted mb-0.5">Realized Cost Savings</p>
            <p className="text-sm font-bold text-text-primary">${Math.round(realizedDollars).toLocaleString()}/d</p>
          </div>
        </div>
      </div>

      {/* Savings chart + Carbon reduction */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ChartCard title="Estimated Savings by Recommendation" className="lg:col-span-2" loading={isLoading}>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" horizontal={false} />
              <XAxis type="number" tickFormatter={v => `${v}%`} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} width={130} />
              <Tooltip
                formatter={(v: any) => [`${(v as number).toFixed(1)}%`, 'Energy Saving']}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
              />
              <Bar dataKey="saving" radius={[0, 4, 4, 0]}>
                {chartData.map((d, i) => (
                  <Cell key={i} fill={PRIORITY_COLORS[d.priority]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Carbon reduction summary */}
        <ChartCard title="Carbon Impact" subtitle="CO₂ equivalent reduction potential">
          <div className="space-y-3">
            {recs.slice(0, 5).map((r, i) => {
              const co2 = ((BASELINE_KWH * r.estimated_saving_pct / 100) * CARBON_FACTOR).toFixed(0)
              return (
                <div key={i} className="flex items-center gap-2">
                  <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: PRIORITY_COLORS[r.priority] }} />
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-text-secondary truncate">{r.title}</p>
                    <div className="h-1.5 bg-bg-hover rounded-full mt-0.5 overflow-hidden">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${r.estimated_saving_pct * 5}%`,
                          background: `linear-gradient(90deg, #22c55e, #06b6d4)`,
                        }}
                      />
                    </div>
                  </div>
                  <span className="text-xs text-cyan-500 font-medium flex-shrink-0">{co2} kg</span>
                </div>
              )
            })}
            <div className="border-t border-bg-border pt-2 flex justify-between text-xs">
              <span className="text-text-muted">Total Daily</span>
              <span className="text-cyan-500 font-semibold">{totalCO2.toFixed(0)} kg CO₂</span>
            </div>
          </div>
        </ChartCard>
      </div>

      {/* Recommendations list */}
      <div>
        <h2 className="section-title mb-4">All Recommendations</h2>
        <div className="space-y-3">
          {isLoading
            ? Array.from({ length: 4 }).map((_, i) => <div key={i} className="skeleton h-16 rounded-xl" />)
            : recs.map((rec, i) => {
                const key = String(rec.id ?? rec.title)
                return (
                  <RecCard 
                    key={key} 
                    rec={rec} 
                    index={i} 
                    isImplemented={implementedIds.includes(key)}
                    onToggleImplement={() => toggleImplement(key)}
                  />
                )
              })
          }
        </div>
      </div>
    </div>
  )
}

export default Optimization
