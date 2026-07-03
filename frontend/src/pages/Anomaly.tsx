import React, { useMemo, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell, LineChart, Line, ReferenceLine, BarChart, Bar,
} from 'recharts'
import { mlService } from '../services/ml'
import type { AnomalyPoint } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'
import { formatDateTime, formatMW, getSeverityColor, downsample } from '../utils/format'

const METHODS_LABELS: Record<string, string> = {
  zscore: 'Z-Score',
  iqr: 'IQR',
  isolation_forest: 'Isolation Forest',
  lof: 'Local Outlier Factor',
  one_class_svm: 'One-Class SVM',
}

const AnomalyTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d: AnomalyPoint = payload[0].payload
  return (
    <div className="glass-card px-3 py-2 text-xs space-y-0.5">
      <p className="text-text-muted">{formatDateTime(d.timestamp)}</p>
      <p className="text-text-primary">{formatMW(d.value)}</p>
      <p className={d.is_anomaly ? 'text-danger-400' : 'text-success-400'}>
        {d.is_anomaly ? `⚠ Anomaly (score: ${d.anomaly_score})` : '✓ Normal'}
      </p>
      {d.is_anomaly && <p className="capitalize">Severity: <span style={{ color: getSeverityColor(d.severity) }}>{d.severity}</span></p>}
    </div>
  )
}

const Anomaly: React.FC = () => {
  const [filter, setFilter] = useState<'all' | 'anomalies'>('all')

  const { data, isLoading } = useQuery({
    queryKey: ['anomalies'],
    queryFn: mlService.getAnomalies,
  })

  const timelineData = useMemo(() => {
    const pts = downsample(data?.points ?? [], 500)
    return filter === 'anomalies' ? pts.filter(p => p.is_anomaly) : pts
  }, [data, filter])

  const scatterData = useMemo(() => data?.points?.filter(p => p.is_anomaly) ?? [], [data])

  // Build heatmap: day × hour grid
  const heatmap = useMemo(() => {
    const grid: number[][] = Array.from({ length: 7 }, () => Array(24).fill(0))
    ;(data?.points ?? []).forEach(p => {
      if (!p.is_anomaly) return
      const d = new Date(p.timestamp)
      grid[d.getDay()][d.getHours()]++
    })
    const max = Math.max(...grid.flat(), 1)
    return grid.map((row, day) =>
      row.map((count, hour) => ({ day, hour, count, pct: count / max }))
    )
  }, [data])

  const DAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']

  return (
    <div className="page-container">
      <PageHeader
        title="Anomaly Detection"
        subtitle="Multi-method ensemble anomaly analysis: Z-Score, IQR, Isolation Forest, LOF, One-Class SVM"
        badge="5 Methods"
        actions={
          <div className="flex gap-1 bg-bg-primary border border-bg-border rounded-lg p-1">
            {(['all', 'anomalies'] as const).map(f => (
              <button key={f} onClick={() => setFilter(f)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all capitalize ${
                  filter === f ? 'bg-electric-600 text-white' : 'text-text-secondary hover:text-text-primary'
                }`}>
                {f === 'all' ? 'All Points' : 'Anomalies Only'}
              </button>
            ))}
          </div>
        }
      />

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Total Records', value: (data?.total_records ?? 0).toLocaleString(), color: 'text-text-primary' },
          { label: 'Anomalies Detected', value: (data?.anomaly_count ?? 0).toLocaleString(), color: 'text-danger-400' },
          { label: 'Anomaly Rate', value: `${data?.anomaly_rate_pct ?? 0}%`, color: 'text-warning-400' },
          { label: 'Normal Points', value: ((data?.total_records ?? 0) - (data?.anomaly_count ?? 0)).toLocaleString(), color: 'text-success-400' },
        ].map(kpi => (
          <div key={kpi.label} className="glass-card p-4">
            <p className="label mb-1">{kpi.label}</p>
            <p className={`text-xl font-display font-bold ${kpi.color}`}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Timeline */}
      <ChartCard
        title="Outlier Timeline"
        subtitle="Energy consumption with anomaly markers"
        loading={isLoading}
        className="col-span-2"
      >
        <ResponsiveContainer width="100%" height={280}>
          <ScatterChart>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" />
            <XAxis
              dataKey="timestamp"
              tickFormatter={v => formatDateTime(v).split(',')[0]}
              tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
              name="Time"
            />
            <YAxis
              dataKey="value"
              tickFormatter={v => `${(v / 1000).toFixed(0)}K`}
              tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
              width={48}
              name="MW"
            />
            <Tooltip content={<AnomalyTooltip />} />
            <Scatter data={timelineData} name="Energy">
              {timelineData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.is_anomaly ? getSeverityColor(entry.severity) : 'var(--bg-border)'}
                  opacity={entry.is_anomaly ? 0.9 : 0.6}
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Method breakdown + Severity distribution */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <ChartCard title="Method Breakdown" subtitle="Anomalies detected per algorithm">
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={(data?.method_breakdown ?? []).map(m => ({
              name: METHODS_LABELS[m.method] ?? m.method,
              count: m.count,
            }))}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" vertical={false} />
              <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
                labelStyle={{ color: 'var(--text-muted)' }}
              />
              <Bar dataKey="count" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                {(data?.method_breakdown ?? []).map((_, i) => (
                  <Cell key={i} fill={['#3b82f6', '#22d3ee', '#8b5cf6', '#f59e0b', '#ef4444'][i % 5]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        {/* Severity score distribution */}
        <ChartCard title="Severity Score Distribution" subtitle="Anomaly score concentration">
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={
              scatterData
                .slice(0, 100)
                .map((p, i) => ({ i, score: p.anomaly_score, severity: p.severity }))
            }>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" />
              <XAxis dataKey="i" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
              <Tooltip
                formatter={(v: any) => v.toFixed(4)}
                contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
              />
              <ReferenceLine y={0.8} stroke="#ef4444" strokeDasharray="4 2" label={{ value: 'High', fill: '#ef4444', fontSize: 9 }} />
              <ReferenceLine y={0.5} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'Medium', fill: '#f59e0b', fontSize: 9 }} />
              <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} dot={(p: any) => (
                <circle key={p.key} cx={p.cx} cy={p.cy} r={3}
                  fill={getSeverityColor(p.payload.severity)} opacity={0.9} />
              )} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Heatmap */}
      <ChartCard title="Anomaly Heatmap" subtitle="Day × hour distribution of detected anomalies">
        <div className="overflow-x-auto">
          <div className="min-w-[640px]">
            <div className="flex gap-1 mb-1 ml-10">
              {Array.from({ length: 24 }, (_, h) => (
                <div key={h} className="w-6 text-center text-xs text-text-muted">{h % 6 === 0 ? `${h}h` : ''}</div>
              ))}
            </div>
            {heatmap.map((row, day) => (
              <div key={day} className="flex items-center gap-1 mb-0.5">
                <span className="w-8 text-xs text-text-muted text-right pr-2">{DAYS[day]}</span>
                {row.map(cell => (
                  <div
                    key={cell.hour}
                    title={`${DAYS[cell.day]} ${cell.hour}:00 — ${cell.count} anomalies`}
                    className="w-6 h-6 rounded-sm transition-all cursor-pointer hover:ring-1 hover:ring-electric-400"
                    style={{
                      background: cell.count === 0
                        ? 'var(--bg-secondary)'
                        : `rgba(239,68,68,${0.15 + cell.pct * 0.85})`,
                    }}
                  />
                ))}
              </div>
            ))}
            <div className="flex items-center gap-2 mt-3 ml-10">
              <span className="text-xs text-text-muted">Low</span>
              {[0.1, 0.3, 0.5, 0.7, 0.9, 1.0].map(p => (
                <div key={p} className="w-5 h-5 rounded-sm" style={{ background: `rgba(239,68,68,${p})` }} />
              ))}
              <span className="text-xs text-text-muted">High</span>
            </div>
          </div>
        </div>
      </ChartCard>
    </div>
  )
}

export default Anomaly
