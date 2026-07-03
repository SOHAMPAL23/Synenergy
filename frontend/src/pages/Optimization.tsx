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

const RecCard: React.FC<{ rec: RecommendationItem; index: number }> = ({ rec, index }) => {
  const [open, setOpen] = useState(false)
  const estimatedKwh = BASELINE_KWH * (rec.estimated_saving_pct / 100)
  const co2 = (estimatedKwh * CARBON_FACTOR).toFixed(0)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.06 }}
      className="glass-card p-4 hover:border-electric-500/30 transition-all"
    >
      <div
        className="flex items-start gap-3 cursor-pointer"
        onClick={() => setOpen(o => !o)}
      >
        <div
          className="w-1 self-stretch rounded-full flex-shrink-0"
          style={{ background: PRIORITY_COLORS[rec.priority] }}
        />
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1">
            <span className={getPriorityColor(rec.priority)}>{rec.priority}</span>
            <span className="badge-info">{rec.category}</span>
            <span className="text-xs text-success-400 font-semibold">↓ {rec.estimated_saving_pct}%</span>
          </div>
          <h3 className="text-sm font-semibold text-text-primary">{rec.title}</h3>
          <p className="text-xs text-text-secondary mt-0.5 line-clamp-2">{rec.description}</p>
        </div>
        <div className="flex-shrink-0 flex items-center gap-3 text-xs text-text-muted">
          <span className="text-success-400 font-medium">{rec.estimated_saving_pct}%</span>
          {open ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </div>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-3 pl-4 border-l border-bg-border"
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

  const totalSavingPct = recs.reduce((s, r) => s + r.estimated_saving_pct, 0) / Math.max(recs.length, 1)
  const totalCO2 = recs.reduce((s, r) => s + BASELINE_KWH * (r.estimated_saving_pct / 100) * CARBON_FACTOR, 0)

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
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
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
            : recs.map((rec, i) => <RecCard key={rec.id ?? i} rec={rec} index={i} />)
          }
        </div>
      </div>
    </div>
  )
}

export default Optimization
