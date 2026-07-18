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
import { formatMW, formatDateTime, getPriorityColor, downsample, cn } from '../utils/format'
import { useNavigate } from 'react-router-dom'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3.5 py-2.5 text-xs border border-bg-border/60 shadow-lg select-none">
      <p className="text-text-muted font-medium mb-1.5">{formatDateTime(label)}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }} className="font-medium">
          {p.name}: <span className="font-bold text-text-primary ml-1">{formatMW(p.value)}</span>
        </p>
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
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 mb-6 overflow-hidden relative border border-bg-border/60"
      >
        {isHealthy ? (
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <span className="relative flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-success-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-success-500 shadow-glow-green"></span>
              </span>
              <div>
                <h4 className="text-sm font-bold text-text-primary tracking-tight">System Status: Fully Operational</h4>
                <p className="text-xs text-text-muted mt-0.5">
                  Database contains active records. Selected ML Baseline: <strong className="text-electric-400 font-semibold">{stats.best_model}</strong>. Real-time predictions online.
                </p>
              </div>
            </div>
            <div className="flex gap-2">
              <button onClick={() => navigate('/forecast')} className="btn-secondary text-xs px-3.5 py-1.5 cursor-pointer">View Forecasts</button>
              <button onClick={() => navigate('/anomalies')} className="btn-secondary text-xs px-3.5 py-1.5 cursor-pointer">Check Outliers</button>
            </div>
          </div>
        ) : (
          <div>
            <div className="flex items-center justify-between mb-5 border-b border-bg-border/40 pb-3">
              <div>
                <h4 className="text-sm font-bold text-text-primary flex items-center gap-2 tracking-tight">
                  <RefreshCw size={14} className="text-electric-400 animate-spin" style={{ animationDuration: '4s' }} />
                  System Setup & ML Ingestion Pipeline
                </h4>
                <p className="text-xs text-text-muted mt-0.5">Follow these setup milestones to baseline your energy grids and start prediction models.</p>
              </div>
              <span className="badge-medium text-[10px] font-bold px-2 py-0.5">Initial Setup Pending</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
              {/* Step 1 */}
              <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                hasData 
                  ? 'bg-success-500/5 border-success-500/20 shadow-sm' 
                  : 'bg-electric-500/5 border-electric-500/20 hover:border-electric-500/35'
              }`}>
                <div className="flex items-center gap-2.5 mb-2">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white shadow-sm ${
                    hasData ? 'bg-success-500' : 'bg-electric-500'
                  }`}>
                    1
                  </span>
                  <span className="font-semibold text-xs text-text-primary tracking-wide">Ingest Energy CSV</span>
                </div>
                <p className="text-xs text-text-muted leading-relaxed mb-3">Upload a historical time-series CSV file to construct the model energy consumption baselines.</p>
                {hasData ? (
                  <span className="text-[11px] font-bold text-success-400 flex items-center gap-1">✓ Complete ({stats.total_records.toLocaleString()} rows)</span>
                ) : (
                  <button onClick={() => navigate('/upload')} className="text-xs font-bold text-electric-400 hover:text-electric-300 flex items-center gap-1 cursor-pointer transition-transform hover:translate-x-0.5">
                    Go to Upload &rarr;
                  </button>
                )}
              </div>

              {/* Step 2 */}
              <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                hasData
                  ? hasModel
                    ? 'bg-success-500/5 border-success-500/20 shadow-sm'
                    : 'bg-warning-500/5 border-warning-500/20 animate-pulse-slow'
                  : 'opacity-50 border-bg-border'
              }`}>
                <div className="flex items-center gap-2.5 mb-2">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white ${
                    hasModel ? 'bg-success-500 shadow-sm' : 'bg-bg-border text-text-muted'
                  }`}>
                    2
                  </span>
                  <span className="font-semibold text-xs text-text-primary tracking-wide">Run AutoML Training</span>
                </div>
                <p className="text-xs text-text-muted leading-relaxed mb-3">Train 6 distinct model variations, automatically ranking and selecting the configuration with the lowest RMSE.</p>
                {hasModel ? (
                  <span className="text-[11px] font-bold text-success-400">✓ Best Model Selected: {stats.best_model}</span>
                ) : hasData ? (
                  <button 
                    onClick={handleTrain} 
                    disabled={training} 
                    className="text-xs font-bold text-warning-500 hover:text-warning-400 flex items-center gap-1 cursor-pointer transition-transform hover:translate-x-0.5"
                  >
                    {training ? 'Training Pipeline...' : 'Trigger Training \u2192'}
                  </button>
                ) : (
                  <span className="text-xs text-text-muted">Awaiting Ingestion</span>
                )}
              </div>

              {/* Step 3 */}
              <div className={`p-4 rounded-2xl border transition-all duration-300 ${
                isHealthy 
                  ? 'bg-success-500/5 border-success-500/20 shadow-sm' 
                  : 'opacity-50 border-bg-border'
              }`}>
                <div className="flex items-center gap-2.5 mb-2">
                  <span className={`w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold text-white ${
                    isHealthy ? 'bg-success-500 shadow-sm' : 'bg-bg-border text-text-muted'
                  }`}>
                    3
                  </span>
                  <span className="font-semibold text-xs text-text-primary tracking-wide">Generate Analytics</span>
                </div>
                <p className="text-xs text-text-muted leading-relaxed mb-3">Examine hourly load forecasts, flag out-of-bounds anomaly points, and read optimization checklist cards.</p>
                {isHealthy ? (
                  <span className="text-[11px] font-bold text-success-400">✓ Analytics Operational</span>
                ) : (
                  <span className="text-xs text-text-muted">Awaiting Pipeline Activation</span>
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
          <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
            {(dashboard?.top_recommendations ?? []).length > 0 ? (
              dashboard?.top_recommendations.map((rec, i) => (
                <div key={i} className="flex items-start gap-2.5 p-3 rounded-xl bg-bg-primary/45 border border-bg-border/30 hover:border-bg-border/60 hover:bg-bg-primary/70 transition-all duration-200 cursor-default">
                  <span className={cn('text-[9px] py-0.5 px-1.5 font-bold uppercase tracking-wide rounded-md', getPriorityColor(rec.priority))}>{rec.priority}</span>
                  <div className="min-w-0">
                    <p className="text-xs text-text-primary font-semibold truncate">{rec.title}</p>
                    <p className="text-[10px] text-text-muted mt-0.5">{rec.category} · <span className="text-success-400 font-bold">{rec.estimated_saving_pct}% savings</span></p>
                  </div>
                </div>
              ))
            ) : (
              <EmptyState title="No recommendations" subtitle="Run training to generate optimization advice." />
            )}
          </div>
          {(dashboard?.top_recommendations?.length ?? 0) > 0 && (
            <button onClick={() => navigate('/optimization')} className="mt-4 flex items-center gap-1.5 text-xs text-electric-400 hover:text-electric-300 font-semibold cursor-pointer group">
              View all checklists <ArrowRight size={12} className="transition-transform group-hover:translate-x-0.5" />
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
