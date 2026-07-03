import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion } from 'framer-motion'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend,
} from 'recharts'
import { Zap, TrendingUp, AlertTriangle, Lightbulb, Upload, RefreshCw, ArrowRight } from 'lucide-react'
import { dashboardService } from '../services/dashboard'
import { mlService } from '../services/ml'
import MetricCard from '../components/ui/MetricCard'
import PageHeader, { ChartCard, Spinner, EmptyState } from '../components/ui/PageHeader'
import { formatMW, formatDateTime, getPriorityColor, downsample } from '../utils/format'
import { useNavigate } from 'react-router-dom'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-xs">
      <p className="text-text-muted mb-1">{formatDateTime(label)}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>{p.name}: <span className="font-semibold">{formatMW(p.value)}</span></p>
      ))}
    </div>
  )
}

const Dashboard: React.FC = () => {
  const navigate = useNavigate()
  const [training, setTraining] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [uploadFile, setUploadFile] = useState<File | null>(null)
  const [uploadMsg, setUploadMsg] = useState('')

  const { data: dashboard, isLoading, refetch } = useQuery({
    queryKey: ['dashboard'],
    queryFn: dashboardService.getDashboard,
    retry: false,
  })

  const { data: forecasts } = useQuery({
    queryKey: ['forecasts'],
    queryFn: mlService.getForecasts,
    retry: false,
  })

  const chartData = downsample(
    (forecasts?.forecasts?.['24h']?.points ?? []).map(p => ({
      ts: p.timestamp,
      Forecast: p.forecast,
      'Upper Bound': p.upper_bound,
      'Lower Bound': p.lower_bound,
    })),
    48
  )

  const stats = dashboard?.stats

  const handleTrain = async () => {
    setTraining(true)
    try { await mlService.train(); refetch() }
    catch (e: any) { alert(e?.response?.data?.detail ?? 'Training failed') }
    finally { setTraining(false) }
  }

  const handleUpload = async () => {
    if (!uploadFile) return
    setUploading(true)
    setUploadMsg('')
    try {
      const res = await mlService.uploadCSV(uploadFile)
      setUploadMsg(`✓ Uploaded ${res.rows_valid.toLocaleString()} records.`)
      setUploadFile(null)
    } catch (e: any) {
      setUploadMsg(`✗ ${e?.response?.data?.detail ?? 'Upload failed'}`)
    } finally {
      setUploading(false)
    }
  }

  const cards = [
    {
      title: 'Avg. Consumption',
      value: stats ? formatMW(stats.avg_consumption_mw) : '—',
      subtitle: 'Historical average',
      icon: <Zap size={18} />, accentColor: 'cyan' as const,
    },
    {
      title: 'Peak Consumption',
      value: stats ? formatMW(stats.max_consumption_mw) : '—',
      subtitle: 'All-time maximum',
      icon: <TrendingUp size={18} />, accentColor: 'blue' as const,
    },
    {
      title: 'Recommendations',
      value: stats?.recommendations_count ?? '—',
      subtitle: `${stats?.high_priority_recommendations ?? 0} high priority`,
      icon: <Lightbulb size={18} />, accentColor: 'amber' as const,
    },
    {
      title: 'Best Model',
      value: stats?.best_model ?? 'Not trained',
      subtitle: `${stats?.forecast_horizons_available?.join(', ') || 'No forecasts'}`,
      icon: <AlertTriangle size={18} />, accentColor: 'purple' as const,
    },
  ]

  return (
    <div className="page-container">
      <PageHeader
        title="Dashboard"
        subtitle="Energy consumption overview and ML analytics"
        badge="Live"
        actions={
          <div className="flex items-center gap-2">
            {/* Upload CSV */}
            <label className="btn-secondary flex items-center gap-2 cursor-pointer text-sm">
              <Upload size={14} />
              {uploading ? 'Uploading…' : 'Upload CSV'}
              <input type="file" accept=".csv" className="hidden" onChange={e => setUploadFile(e.target.files?.[0] ?? null)} />
            </label>
            {uploadFile && (
              <button onClick={handleUpload} disabled={uploading} className="btn-primary flex items-center gap-2 text-sm">
                {uploading ? <Spinner size={14} /> : null} Upload
              </button>
            )}
            {uploadMsg && <span className="text-xs text-slate-400">{uploadMsg}</span>}

            <button onClick={handleTrain} disabled={training}
              className="btn-primary flex items-center gap-2 text-sm">
              {training ? <Spinner size={14} /> : <RefreshCw size={14} />}
              {training ? 'Training…' : 'Run Training'}
            </button>
          </div>
        }
      />

      {/* Metric cards */}
      <motion.div
        initial="hidden"
        animate="show"
        variants={{ hidden: {}, show: { transition: { staggerChildren: 0.1 } } }}
        className="grid grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {cards.map((card, i) => (
          <motion.div key={i} variants={{ hidden: { opacity: 0, y: 16 }, show: { opacity: 1, y: 0 } }}>
            <MetricCard {...card} loading={isLoading} />
          </motion.div>
        ))}
      </motion.div>

      {/* Charts row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Main forecast chart */}
        <ChartCard
          title="24-Hour Energy Forecast"
          subtitle={`Model: ${forecasts?.best_model ?? '—'}`}
          className="lg:col-span-2"
        >
          {chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="forecastGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="upperGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" />
                <XAxis dataKey="ts" tickFormatter={v => formatDateTime(v).split(',')[0]} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <YAxis tickFormatter={v => `${(v/1000).toFixed(0)}K`} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Area type="monotone" dataKey="Upper Bound" stroke="#22d3ee" strokeWidth={1} fill="url(#upperGrad)" strokeDasharray="4 2" />
                <Area type="monotone" dataKey="Forecast" stroke="#3b82f6" strokeWidth={2} fill="url(#forecastGrad)" />
                <Area type="monotone" dataKey="Lower Bound" stroke="#22d3ee" strokeWidth={1} fill="none" strokeDasharray="4 2" />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <EmptyState title="No forecast data" subtitle="Upload a CSV and run training to generate forecasts." />
          )}
        </ChartCard>

        {/* Top Recommendations */}
        <ChartCard title="Top Recommendations" subtitle={`${dashboard?.top_recommendations?.length ?? 0} active`}>
          <div className="space-y-2 max-h-64 overflow-y-auto">
            {(dashboard?.top_recommendations ?? []).length > 0 ? (
              dashboard?.top_recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-2 p-2.5 rounded-lg bg-bg-primary/60">
                  <span className={getPriorityColor(rec.priority)}>{rec.priority}</span>
                  <div className="min-w-0">
                    <p className="text-xs text-text-primary font-medium truncate">{rec.title}</p>
                    <p className="text-xs text-text-muted truncate">{rec.category} · {rec.estimated_saving_pct}% savings</p>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState title="No recommendations" subtitle="Run training to generate optimization advice." />
            )}
          </div>
          {(dashboard?.top_recommendations?.length ?? 0) > 0 && (
            <button onClick={() => navigate('/optimization')} className="mt-3 flex items-center gap-1 text-xs text-electric-400 hover:text-electric-300 font-medium">
              View all <ArrowRight size={12} />
            </button>
          )}
        </ChartCard>
      </div>

      {/* Stats footer */}
      {stats && (
        <div className="glass-card p-4 flex flex-wrap gap-6 text-xs">
          <div><span className="text-text-muted">Total Records: </span><span className="text-text-primary font-medium">{stats.total_records.toLocaleString()}</span></div>
          {stats.date_range_start && <div><span className="text-text-muted">From: </span><span className="text-text-primary font-medium">{stats.date_range_start?.split('T')[0]}</span></div>}
          {stats.date_range_end && <div><span className="text-text-muted">To: </span><span className="text-text-primary font-medium">{stats.date_range_end?.split('T')[0]}</span></div>}
          <div><span className="text-text-muted">Min: </span><span className="text-text-primary font-medium">{formatMW(stats.min_consumption_mw)}</span></div>
          <div><span className="text-text-muted">Max: </span><span className="text-text-primary font-medium">{formatMW(stats.max_consumption_mw)}</span></div>
        </div>
      )}
    </div>
  )
}

export default Dashboard
