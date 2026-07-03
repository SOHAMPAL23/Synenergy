import React from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
} from 'recharts'
import { Brain, Star } from 'lucide-react'
import { mlService } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'

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
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Features Analyzed', value: items.length },
          { label: 'Top Feature', value: data?.top_features?.[0] ?? '—' },
          { label: 'Model', value: data?.model_name ?? '—' },
          { label: 'Explainer', value: data?.explainer_type ?? '—' },
        ].map(kpi => (
          <div key={kpi.label} className="glass-card p-4">
            <p className="label mb-1">{kpi.label}</p>
            <p className="text-sm font-semibold text-text-primary truncate">{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Main charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* SHAP Summary bar */}
        <ChartCard
          title="SHAP Feature Importance"
          subtitle="Mean |SHAP| value per feature"
          loading={isLoading}
        >
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={[...items].slice(0, 12).reverse().map(item => ({
                feature: item.feature.replace(/_/g, ' '),
                shap: parseFloat(item.mean_abs_shap.toFixed(1)),
                rank: item.rank,
              }))}
              layout="vertical"
            >
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <YAxis type="category" dataKey="feature" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} width={110} />
              <Tooltip
                formatter={(v: any) => [v.toFixed(2), 'Mean |SHAP|']}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
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
        <ChartCard title="Feature Importance Radar" subtitle="Top 6 features normalized to 100%">
          <ResponsiveContainer width="100%" height={320}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="var(--bg-border)" />
              <PolarAngleAxis dataKey="feature" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
              <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <Radar name="Importance" dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.25} strokeWidth={2} />
              <Tooltip
                formatter={(v: any) => [`${v.toFixed(1)}%`, 'Relative importance']}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
              />
            </RadarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* SHAP Waterfall */}
      <ChartCard
        title="SHAP Waterfall (Top 8 Features)"
        subtitle="Contribution magnitude of each feature to model output"
      >
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={waterfallData}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
            <YAxis tickFormatter={v => v.toFixed(0)} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
            <Tooltip
              formatter={(v: any) => [v.toFixed(2), 'Mean |SHAP|']}
              contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]}>
              {waterfallData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Feature rank table */}
      <ChartCard title="Complete Feature Ranking">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-bg-border">
                {['Rank', 'Feature', 'Mean |SHAP|', 'Relative Importance'].map(col => (
                  <th key={col} className="label py-2 pr-4 text-left">{col}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/50">
              {items.map((item, i) => {
                const pct = (item.mean_abs_shap / maxShap) * 100
                return (
                  <tr key={item.feature} className="hover:bg-bg-hover/30 transition-colors">
                    <td className="py-2 pr-4">
                      {i === 0 && <Star size={12} className="inline text-warning-400 mr-1" />}
                      <span className={i < 3 ? 'text-electric-500 font-semibold' : 'text-text-muted'}>#{item.rank}</span>
                    </td>
                    <td className="py-2 pr-4 font-mono text-xs text-text-secondary">{item.feature}</td>
                    <td className="py-2 pr-4 text-text-primary font-medium">{item.mean_abs_shap.toFixed(2)}</td>
                    <td className="py-2 pr-4 min-w-[120px]">
                      <div className="flex items-center gap-2">
                        <div className="flex-1 h-1.5 bg-bg-hover rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all"
                            style={{ width: `${pct}%`, background: COLORS[i % COLORS.length] }}
                          />
                        </div>
                        <span className="text-xs text-text-muted w-10 text-right">{pct.toFixed(1)}%</span>
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
