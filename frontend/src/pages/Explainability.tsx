import React from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { Brain, Star } from 'lucide-react'
import { mlService } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'
import { cn } from '../utils/format'

const COLORS = ['#3b82f6', '#22d3ee', '#8b5cf6', '#f59e0b', '#22c55e', '#ef4444', '#06b6d4', '#a78bfa', '#fb923c', '#34d399']

const WaterfallBar = (props: any) => {
  const { x, y, width, height, fill } = props
  return <rect x={x} y={y} width={width} height={height} fill={fill} rx={3} />
}

const Explainability: React.FC = () => {
  const { data, isLoading } = useQuery({
    queryKey: ['explanations'],
    queryFn: mlService.getExplanations,
  })

  const items = data?.feature_importances ?? []

  // Waterfall: cumulative contribution visualization
  const waterfallData = items.slice(0, 8).map((item, i) => ({
    name: item.feature,
    value: parseFloat(item.mean_abs_shap.toFixed(1)),
    fill: COLORS[i % COLORS.length],
  }))

  // Radar: top 6 features normalized
  const maxShap = items[0]?.mean_abs_shap ?? 1
  const radarData = items.slice(0, 6).map(item => ({
    feature: item.feature.replace(/_/g, ' '),
    importance: parseFloat(((item.mean_abs_shap / maxShap) * 100).toFixed(1)),
  }))

  return (
    <div className="page-container">
      <PageHeader
        title="Explainability"
        subtitle={`SHAP feature importance analysis · Model: ${data?.model_name ?? '—'} · Explainer: ${data?.explainer_type ?? '—'}`}
        badge="SHAP"
        actions={
          <div className="glass-card px-3 py-1.5 flex items-center gap-2 text-xs text-text-muted">
            <Brain size={12} /> {data?.explainer_type ?? 'TreeExplainer'}
          </div>
        }
      />

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-5 animate-slide-up">
        {[
          { label: 'Features Analyzed', value: items.length, highlight: false },
          { label: 'Top Contributor', value: data?.top_features?.[0]?.replace(/_/g, ' ') ?? '—', highlight: true },
          { label: 'Model Pipeline', value: data?.model_name ?? '—', highlight: false },
          { label: 'Explainer Config', value: data?.explainer_type ?? '—', highlight: false },
        ].map(kpi => (
          <div key={kpi.label} className="glass-card p-5 border border-bg-border/60 hover:-translate-y-1 transition-all duration-300">
            <p className="label mb-1.5">{kpi.label}</p>
            <p className={cn("text-xs font-bold truncate tracking-wide text-text-primary capitalize", kpi.highlight && "text-electric-400 font-bold")}>
              {kpi.value}
            </p>
          </div>
        ))}
      </div>

      {/* Main charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 animate-slide-up">
        {/* SHAP Summary bar */}
        <ChartCard
          title="SHAP Feature Importance"
          subtitle="Mean absolute contribution magnitude |SHAP| value per feature"
          loading={isLoading}
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={[...items].slice(0, 12).reverse().map(item => ({
                feature: item.feature.replace(/_/g, ' '),
                shap: parseFloat(item.mean_abs_shap.toFixed(2)),
                rank: item.rank,
              }))}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} width={120} />
              <Tooltip
                formatter={(v: any) => [v.toFixed(3), 'Mean |SHAP|']}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ fontWeight: 600 }}
              />
              <Bar dataKey="shap" shape={<WaterfallBar />} radius={[0, 4, 4, 0]}>
                {items.slice(0, 12).map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Radar chart */}
        <ChartCard title="Feature Importance Radar" subtitle="Top 6 parameters normalized to 100% relative influence">
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--bg-border)" />
              <PolarAngleAxis dataKey="feature" tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontWeight: 500 }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 9, fill: 'var(--text-muted)' }} />
              <Radar name="Importance" dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} strokeWidth={2.5} />
              <Tooltip
                formatter={(v: any) => [`${v.toFixed(1)}%`, 'Relative influence']}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* SHAP Waterfall */}
      <ChartCard
        title="SHAP Cumulative Contributions (Top 8 Features)"
        subtitle="Individual impact contribution magnitude of metrics on AutoML decisions"
      >
        <ResponsiveContainer width="100%" height={230}>
          <BarChart data={waterfallData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-secondary)' }} />
            <YAxis tickFormatter={v => v.toFixed(0)} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
            <Tooltip
              formatter={(v: any) => [v.toFixed(3), 'Mean |SHAP|']}
              contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
            />
            <Bar dataKey="value" radius={[6, 6, 0, 0]}>
              {waterfallData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Feature rank table */}
      <ChartCard title="Complete Feature Contribution Ranking" subtitle="Ranked significance values across variables in training sets">
        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left border-collapse">
            <thead>
              <tr className="border-b border-bg-border/60 pb-2">
                {['Rank', 'Param Variable', 'Mean |SHAP|', 'Relative Weight'].map(col => (
                  <th key={col} className="label py-3 px-3">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/30">
              {items.map((item, i) => {
                const pct = (item.mean_abs_shap / maxShap) * 100
                const isTop = i < 3
                return (
                  <tr key={item.feature} className="hover:bg-bg-hover/20 transition-colors duration-150">
                    <td className="py-3 px-3 font-semibold">
                      {i === 0 && <Star size={11} className="inline text-warning-400 mr-1 pb-0.5 animate-pulse" />}
                      <span className={isTop ? 'text-electric-400 font-bold' : 'text-text-muted'}>#{item.rank}</span>
                    </td>
                    <td className="py-3 px-3 font-mono text-text-secondary">{item.feature}</td>
                    <td className="py-3 px-3 text-text-primary font-bold">{item.mean_abs_shap.toFixed(3)}</td>
                    <td className="py-3 px-3 min-w-[140px]">
                      <div className="flex items-center gap-2.5">
                        <div className="flex-1 h-2 bg-bg-border rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{ 
                              width: `${pct}%`, 
                              background: `linear-gradient(90deg, ${COLORS[i % COLORS.length]}88 0%, ${COLORS[i % COLORS.length]} 100%)` 
                            }}
                          />
                        </div>
                        <span className="text-[10px] text-text-muted font-bold w-10 text-right">{pct.toFixed(1)}%</span>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  )
}

export default Explainability
