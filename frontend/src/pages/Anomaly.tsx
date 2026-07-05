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
  const [severityFilter, setSeverityFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all')
  const [showAcknowledged, setShowAcknowledged] = useState(false)
  const [acknowledgedKeys, setAcknowledgedKeys] = useState<string[]>(() => {
    try {
      const saved = localStorage.getItem('acknowledged_anomalies')
      return saved ? JSON.parse(saved) : []
    } catch {
      return []
    }
  })

  const toggleAcknowledge = (timestamp: string) => {
    setAcknowledgedKeys(prev => {
      const next = prev.includes(timestamp) ? prev.filter(t => t !== timestamp) : [...prev, timestamp]
      localStorage.setItem('acknowledged_anomalies', JSON.stringify(next))
      return next
    })
  }

  const { data, isLoading } = useQuery({
    queryKey: ['anomalies'],
    queryFn: mlService.getAnomalies,
  })

  const allAnomalies = useMemo(() => {
    return (data?.points ?? []).filter(p => p.is_anomaly)
  }, [data])

  const filteredAnomalies = useMemo(() => {
    return allAnomalies.filter(p => {
      const matchSeverity = severityFilter === 'all' || p.severity === severityFilter
      const isAck = acknowledgedKeys.includes(p.timestamp)
      const matchAck = showAcknowledged || !isAck
      return matchSeverity && matchAck
    })
  }, [allAnomalies, severityFilter, acknowledgedKeys, showAcknowledged])

  const handleExportReport = () => {
    if (filteredAnomalies.length === 0) return
    const timestampStr = new Date().toISOString().split('T')[0]
    const header = `ENERVISION AI - ANOMALY INCIDENT REPORT (${timestampStr})\n` +
                   `============================================================\n\n` +
                   `Active alerts in triage: ${filteredAnomalies.length}\n` +
                   `Filters applied: Severity=${severityFilter.toUpperCase()}, Show Resolved=${showAcknowledged ? 'YES' : 'NO'}\n\n` +
                   `Incidents List:\n` +
                   `------------------------------------------------------------\n`
    const rows = filteredAnomalies.map((p, idx) => {
      const status = acknowledgedKeys.includes(p.timestamp) ? 'RESOLVED' : 'OPEN'
      return `${idx + 1}. Time: ${formatDateTime(p.timestamp)} | Value: ${formatMW(p.value)} | Severity: ${p.severity.toUpperCase()} | Score: ${(p.anomaly_score * 100).toFixed(0)}% | Status: ${status}`
    }).join('\n')
    
    const blob = new Blob([header + rows], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `enervision_incident_report_${timestampStr}.txt`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

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

      {/* Triage Alert Queue Section */}
      <ChartCard 
        title="Anomaly Alert Queue" 
        subtitle={`${filteredAnomalies.length} active alerts matching filters`}
        actions={
          <div className="flex items-center gap-3 flex-wrap sm:flex-nowrap">
            {/* Severity filter */}
            <div className="flex gap-1 bg-bg-primary border border-bg-border rounded-lg p-0.5">
              {(['all', 'high', 'medium', 'low'] as const).map(sev => (
                <button 
                  key={sev} 
                  onClick={() => setSeverityFilter(sev)}
                  className={`px-2.5 py-1 rounded-md text-[10px] uppercase font-bold transition-all cursor-pointer ${
                    severityFilter === sev 
                      ? 'bg-electric-600 text-white' 
                      : 'text-text-muted hover:text-text-primary'
                  }`}
                >
                  {sev}
                </button>
              ))}
            </div>

            {/* Show Acknowledged Switch */}
            <label className="flex items-center gap-1.5 text-xs text-text-muted cursor-pointer hover:text-text-primary select-none whitespace-nowrap">
              <input 
                type="checkbox" 
                checked={showAcknowledged} 
                onChange={(e) => setShowAcknowledged(e.target.checked)}
                className="w-3.5 h-3.5 rounded border-bg-border bg-bg-primary text-electric-600 focus:ring-electric-500 cursor-pointer"
              />
              Show Resolved
            </label>

            {/* Export Button */}
            <button 
              onClick={handleExportReport} 
              className="btn-secondary text-[11px] px-2.5 py-1 flex items-center gap-1 cursor-pointer whitespace-nowrap"
              disabled={filteredAnomalies.length === 0}
            >
              Export Report
            </button>
          </div>
        }
      >
        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-sm text-left border-collapse">
            <thead>
              <tr className="border-b border-bg-border text-xs">
                <th className="label py-2.5 px-3">Triage</th>
                <th className="label py-2.5 px-3">Time Detected</th>
                <th className="label py-2.5 px-3">Measured Load</th>
                <th className="label py-2.5 px-3">Confidence Score</th>
                <th className="label py-2.5 px-3">Severity</th>
                <th className="label py-2.5 px-3">Resolution</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-bg-border/40">
              {filteredAnomalies.slice(0, 100).map((p) => {
                const isAck = acknowledgedKeys.includes(p.timestamp)
                return (
                  <tr key={p.timestamp} className={`hover:bg-bg-hover/20 transition-colors ${isAck ? 'opacity-50' : ''}`}>
                    <td className="py-2.5 px-3">
                      <input 
                        type="checkbox" 
                        checked={isAck} 
                        onChange={() => toggleAcknowledge(p.timestamp)}
                        className="w-4 h-4 rounded border-bg-border bg-bg-primary text-emerald-600 focus:ring-emerald-500 cursor-pointer"
                        title={isAck ? "Re-open incident" : "Acknowledge and resolve incident"}
                      />
                    </td>
                    <td className="py-2.5 px-3 font-medium text-text-primary text-xs">
                      {formatDateTime(p.timestamp)}
                    </td>
                    <td className="py-2.5 px-3 text-text-secondary text-xs font-mono">
                      {formatMW(p.value)}
                    </td>
                    <td className="py-2.5 px-3 text-text-muted text-xs">
                      {(p.anomaly_score * 100).toFixed(0)}% ensemble
                    </td>
                    <td className="py-2.5 px-3">
                      <span 
                        className="px-2 py-0.5 rounded text-[10px] font-bold uppercase"
                        style={{ 
                          color: getSeverityColor(p.severity),
                          background: `${getSeverityColor(p.severity)}15`,
                          border: `1px solid ${getSeverityColor(p.severity)}30` 
                        }}
                      >
                        {p.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-xs">
                      {isAck ? (
                        <span className="text-emerald-500 font-semibold flex items-center gap-0.5">✓ Resolved</span>
                      ) : (
                        <span className="text-warning-500 font-semibold flex items-center gap-0.5">⚠ Open Alert</span>
                      )}
                    </td>
                  </tr>
                )
              })}
              {filteredAnomalies.length === 0 && (
                <tr>
                  <td colSpan={6} className="py-12 text-center text-text-muted">
                    No active anomalies found matching filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {filteredAnomalies.length > 100 && (
          <p className="text-[10px] text-text-muted mt-2">Showing first 100 anomalies in queue.</p>
        )}
      </ChartCard>
    </div>
  )
}

export default Anomaly
