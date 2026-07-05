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

  const hasData = stats && stats.total_records > 0
  const hasModel = stats && stats.best_model && stats.best_model !== 'Not trained'
  const isHealthy = hasData && hasModel

  const renderWizard = () => {
    if (isLoading) return null
    return (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-5 border border-bg-border mb-6 overflow-hidden relative"
      >
        {isHealthy ? (
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <span className="relative flex h-3.5 w-3.5">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3.5 w-3.5 bg-success-500"></span>
              </span>
              <div>
                <h4 className="text-sm font-semibold text-text-primary">System Health Status: Fully Active</h4>
                <p className="text-xs text-text-muted">
                  Database contains active records. Best model: <strong className="text-electric-400 font-semibold">{stats.best_model}</strong>. Predictions are online.
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => navigate('/forecast')} className="btn-secondary text-xs px-3 py-1.5 cursor-pointer">View Forecasts</button>
              <button onClick={() => navigate('/anomalies')} className="btn-secondary text-xs px-3 py-1.5 cursor-pointer">Check Outliers</button>
            </div>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-4 border-b border-bg-border/40 pb-2">
              <div>
                <h4 className="text-sm font-bold text-text-primary flex items-center gap-2">
                  <RefreshCw size={14} className="text-electric-400 animate-spin" style={{ animationDuration: '3s' }} />
                  System Setup & Onboarding Wizard
                </h4>
                <p className="text-xs text-text-muted">Follow these steps to initialize the machine learning pipelines and start optimizing.</p>
              </div>
              <span className="badge-medium text-xs font-semibold px-2 py-0.5">Setup Pending</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Step 1 */}
              <div className={`p-3 rounded-xl border transition-all ${
                hasData 
                  ? 'bg-success-500/5 border-success-500/25' 
                  : 'bg-electric-500/5 border-electric-500/25'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                    hasData ? 'bg-success-500' : 'bg-electric-500'
                  }`}>
                    1
                  </span>
                  <span className="font-semibold text-xs text-text-primary">Ingest Energy CSV</span>
                </div>
                <p className="text-xs text-text-muted mb-2">Upload a time-series CSV file to build the historical model baseline.</p>
                {hasData ? (
                  <span className="text-xs font-semibold text-success-500 flex items-center gap-1">✓ Complete ({stats.total_records.toLocaleString()} rows)</span>
                ) : (
                  <button onClick={() => navigate('/upload')} className="text-xs font-bold text-electric-400 hover:text-electric-300 flex items-center gap-1 cursor-pointer">
                    Go to Upload &rarr;
                  </button>
                )}
              </div>

              {/* Step 2 */}
              <div className={`p-3 rounded-xl border transition-all ${
                hasData
                  ? hasModel
                    ? 'bg-success-500/5 border-success-500/25'
                    : 'bg-warning-500/5 border-warning-500/25 animate-pulse'
                  : 'opacity-50 border-bg-border'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                    hasModel ? 'bg-success-500' : 'bg-bg-border text-text-muted'
                  }`}>
                    2
                  </span>
                  <span className="font-semibold text-xs text-text-primary">Run AutoML Pipeline</span>
                </div>
                <p className="text-xs text-text-muted mb-2">Train 6 model variations, automatically selecting the best configuration by RMSE.</p>
                {hasModel ? (
                  <span className="text-xs font-semibold text-success-500">✓ Best Model: {stats.best_model}</span>
                ) : hasData ? (
                  <button 
                    onClick={handleTrain} 
                    disabled={training} 
                    className="text-xs font-bold text-warning-500 hover:text-warning-400 flex items-center gap-1 cursor-pointer"
                  >
                    {training ? 'Training...' : 'Trigger Training \u2192'}
                  </button>
                ) : (
                  <span className="text-xs text-text-muted">Awaiting Ingestion</span>
                )}
              </div>

              {/* Step 3 */}
              <div className={`p-3 rounded-xl border transition-all ${
                isHealthy 
                  ? 'bg-success-500/5 border-success-500/25' 
                  : 'opacity-50 border-bg-border'
              }`}>
                <div className="flex items-center gap-2 mb-1">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold text-white ${
                    isHealthy ? 'bg-success-500' : 'bg-bg-border text-text-muted'
                  }`}>
                    3
                  </span>
                  <span className="font-semibold text-xs text-text-primary">Access Insights</span>
                </div>
                <p className="text-xs text-text-muted mb-2">Forecast consumption peaks, flag anomaly points, and generate optimal action items.</p>
                {isHealthy ? (
                  <span className="text-xs font-semibold text-success-500">✓ Pipelines Active</span>
                ) : (
                  <span className="text-xs text-text-muted">Awaiting Setup Steps</span>
                )}
              </div>
            </div>
          </div>
        )}
      </motion.div>
    )
  }

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

      {renderWizard()}

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
