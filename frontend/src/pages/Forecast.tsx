import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend, ReferenceLine,
} from 'recharts'
import { Calendar, Zap } from 'lucide-react'
import { mlService } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'
import { formatDateTime, formatMW, downsample, cn } from '../utils/format'

type Horizon = '24h' | '7d' | '30d'

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div className="glass-card px-3 py-2 text-xs space-y-1">
      <p className="text-text-muted">{formatDateTime(label)}</p>
      {payload.map((p: any) => (
        <p key={p.name} style={{ color: p.color }}>
          {p.name}: <span className="font-semibold">{formatMW(p.value)}</span>
        </p>
      ))}
    </div>
  )
}

const Forecast: React.FC = () => {
  const [horizon, setHorizon] = useState<Horizon>('24h')
  const [peakShaving, setPeakShaving] = useState(0)
  const [solarShift, setSolarShift] = useState(0)
  const [gridOpt, setGridOpt] = useState(0)

  const { data, isLoading } = useQuery({
    queryKey: ['forecasts'],
    queryFn: mlService.getForecasts,
  })

  const maxPoints: Record<Horizon, number> = { '24h': 24, '7d': 84, '30d': 120 }
  const raw = data?.forecasts?.[horizon]?.points ?? []
  const chartData = downsample(raw, maxPoints[horizon]).map(p => ({
    ts: p.timestamp,
    Forecast: Math.round(p.forecast),
    Upper: Math.round(p.upper_bound),
    Lower: Math.round(p.lower_bound),
  }))

  const avgForecast = chartData.length
    ? Math.round(chartData.reduce((s, p) => s + p.Forecast, 0) / chartData.length)
    : 0

  const peakForecast = chartData.length
    ? Math.max(...chartData.map(p => p.Forecast))
    : 0

  const simulatedChartData = chartData.map(p => {
    const hour = new Date(p.ts).getHours()
    const isDaylight = hour >= 8 && hour <= 18
    const solarReduction = isDaylight ? p.Forecast * (solarShift / 100) : 0
    const optReduction = p.Forecast * (gridOpt / 100)
    const shavedAmount = p.Forecast > avgForecast ? (p.Forecast - avgForecast) * (peakShaving / 100) : 0
    const simulated = Math.max(0, Math.round(p.Forecast - shavedAmount - solarReduction - optReduction))
    return {
      ...p,
      'Simulated Load': simulated,
    }
  })

  const totalOffsetMW = simulatedChartData.reduce((acc, p) => acc + (p.Forecast - p['Simulated Load']), 0)
  const simulatedPeak = simulatedChartData.length ? Math.max(...simulatedChartData.map(p => p['Simulated Load'])) : 0
  const peakShavedVal = Math.max(0, peakForecast - simulatedPeak)
  const estDailySavings = totalOffsetMW * 85
  const estCarbonAbated = totalOffsetMW * 420

  return (
    <div className="page-container">
      <PageHeader
        title="Forecast Analytics"
        subtitle={`Model: ${data?.best_model ?? '—'} · Multi-horizon energy consumption forecasting`}
        badge={data?.best_model ?? 'AI'}
        actions={
          <div className="flex gap-1 bg-bg-primary border border-bg-border rounded-lg p-1">
            {(['24h', '7d', '30d'] as Horizon[]).map(h => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-all ${
                  horizon === h
                    ? 'bg-electric-600 text-white shadow-glow-blue'
                    : 'text-text-secondary hover:text-text-primary'
                }`}
              >
                {h}
              </button>
            ))}
          </div>
        }
      />

      {/* KPI row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Horizon', value: horizon, color: 'text-electric-500' },
          { label: 'Avg Forecast', value: formatMW(avgForecast), color: 'text-cyan-500' },
          { label: 'Peak Forecast', value: formatMW(peakForecast), color: 'text-warning-500' },
          { label: 'Points', value: chartData.length.toString(), color: 'text-success-500' },
        ].map(kpi => (
          <div key={kpi.label} className="glass-card p-4">
            <p className="label mb-1">{kpi.label}</p>
            <p className={`text-xl font-display font-bold ${kpi.color}`}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Main forecast chart */}
      <ChartCard
        title={`Energy Forecast — ${horizon} horizon`}
        subtitle="Forecast with 95% confidence interval"
        loading={isLoading}
        actions={<Calendar size={14} className="text-slate-500" />}
      >
        <ResponsiveContainer width="100%" height={320}>
          <AreaChart data={simulatedChartData}>
            <defs>
              <linearGradient id="fGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.12} />
                <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--bg-border)" />
            <XAxis
              dataKey="ts"
              tickFormatter={v => {
                const d = new Date(v)
                return horizon === '24h' ? `${d.getHours()}:00` : `${d.getMonth() + 1}/${d.getDate()}`
              }}
              tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            />
            <YAxis tickFormatter={v => `${(v / 1000).toFixed(0)}K`} tick={{ fontSize: 11, fill: 'var(--text-muted)' }} width={48} />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ fontSize: 12, color: 'var(--text-secondary)' }} />
            <Area type="monotone" dataKey="Upper" stroke="#22d3ee" strokeWidth={1.5} fill="url(#ciGrad)" strokeDasharray="5 3" name="Upper Bound" />
            <Area type="monotone" dataKey="Forecast" stroke="#3b82f6" strokeWidth={2.5} fill="url(#fGrad)" />
            <Area type="monotone" dataKey="Lower" stroke="#22d3ee" strokeWidth={1.5} fill="none" strokeDasharray="5 3" name="Lower Bound" />
            <Line type="monotone" dataKey="Simulated Load" stroke="#10b981" strokeWidth={2.5} strokeDasharray="4 4" dot={false} name="Simulated Load" />
            {avgForecast > 0 && (
              <ReferenceLine y={avgForecast} stroke="#f59e0b" strokeDasharray="4 2" label={{ value: 'Avg', fill: '#f59e0b', fontSize: 10 }} />
            )}
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* What-If Simulator Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 mb-2 animate-slide-up">
        {/* Sliders Card */}
        <div className="glass-card p-6 border border-bg-border/60 lg:col-span-2 shadow-lg">
          <div className="flex items-center gap-3 mb-6">
            <span className="p-2 rounded-xl bg-electric-500/10 text-electric-400 shadow-sm">
              <Zap size={16} />
            </span>
            <div>
              <h4 className="text-sm font-bold text-text-primary tracking-tight">What-If Forecast Simulator</h4>
              <p className="text-xs text-text-muted mt-0.5">Simulate operational peak shavings and clean wind/solar offset integrations on the chart horizon.</p>
            </div>
          </div>
          
          <div className="space-y-5">
            {/* Slider 1 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-text-secondary">Peak Shaving Intensity</span>
                <span className="text-electric-400 font-bold">{peakShaving}% load shave</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="20" 
                value={peakShaving} 
                onChange={e => setPeakShaving(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-electric-500 focus:ring-2 focus:ring-electric-500/20"
              />
              <p className="text-[10px] text-text-muted mt-1.5 leading-relaxed">Shaves loads that exceed the median forecast baseline dynamically.</p>
            </div>

            {/* Slider 2 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-text-secondary">Solar & Wind Offset Shift</span>
                <span className="text-emerald-400 font-bold">{solarShift}% capacity</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="30" 
                value={solarShift} 
                onChange={e => setSolarShift(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-emerald-500 focus:ring-2 focus:ring-emerald-500/20"
              />
              <p className="text-[10px] text-text-muted mt-1.5 leading-relaxed">Offsets consumption during peak daylight generation windows (08:00 - 18:00).</p>
            </div>

            {/* Slider 3 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5 font-medium">
                <span className="text-text-secondary">Grid Efficiency Factors</span>
                <span className="text-cyan-400 font-bold">{gridOpt}% continuous gain</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="10" 
                value={gridOpt} 
                onChange={e => setGridOpt(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-cyan-500 focus:ring-2 focus:ring-cyan-500/20"
              />
              <p className="text-[10px] text-text-muted mt-1.5 leading-relaxed">Applies standard continuous machine adjustments and distribution fixes across all hours.</p>
            </div>
          </div>
        </div>

        {/* Dynamic ROI Metrics Card */}
        <div className="glass-card p-6 border border-success-500/25 bg-success-500/5 flex flex-col justify-between shadow-glow-green">
          <div>
            <h4 className="text-sm font-bold text-success-400 tracking-tight">Simulation ROI Estimates</h4>
            <p className="text-xs text-text-muted mt-0.5 mb-5">Calculated offsets based on current simulator values.</p>
            
            <div className="space-y-3.5">
              <div className="flex justify-between items-center border-b border-bg-border/45 pb-2.5">
                <span className="text-xs text-text-secondary">Peak Shaved</span>
                <span className="text-sm font-bold text-text-primary">{formatMW(peakShavedVal)}</span>
              </div>
              
              <div className="flex justify-between items-center border-b border-bg-border/45 pb-2.5">
                <span className="text-xs text-text-secondary">Est. Cost Savings</span>
                <span className="text-sm font-bold text-success-400">+${Math.round(estDailySavings).toLocaleString()}/day</span>
              </div>

              <div className="flex justify-between items-center pb-1">
                <span className="text-xs text-text-secondary">CO₂ Offset</span>
                <span className="text-sm font-bold text-cyan-400">{Math.round(estCarbonAbated).toLocaleString()} kg/day</span>
              </div>
            </div>
          </div>

          <div className="text-[10px] text-text-muted mt-5 border-t border-bg-border/40 pt-3">
            *Simulation calculations are generated locally on the selected forecast interval.
          </div>
        </div>
      </div>

      {/* Horizon comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {(['24h', '7d', '30d'] as Horizon[]).map(h => {
          const pts = downsample(data?.forecasts?.[h]?.points ?? [], 30)
          const hData = pts.map(p => ({ ts: p.timestamp, v: p.forecast }))
          const isCurrent = h === horizon
          return (
            <ChartCard 
              key={h} 
              title={`${h} Outlook`} 
              subtitle={data?.forecasts?.[h]?.model_name ?? '—'}
              className={cn('hover:-translate-y-1 transition-all duration-300', isCurrent && 'border-electric-500/35 bg-electric-500/5')}
            >
              <ResponsiveContainer width="100%" height={120}>
                <AreaChart data={hData}>
                  <defs>
                    <linearGradient id={`grad-${h}`} x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor={isCurrent ? '#3b82f6' : '#64748b'} stopOpacity={0.2} />
                      <stop offset="95%" stopColor={isCurrent ? '#3b82f6' : '#64748b'} stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="ts" hide />
                  <YAxis hide />
                  <Tooltip
                    formatter={(v: any) => formatMW(v as number)}
                    labelFormatter={(label: any) => formatDateTime(String(label))}
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 10 }}
                    labelStyle={{ color: 'var(--text-muted)' }}
                  />
                  <Area type="monotone" dataKey="v" stroke={isCurrent ? '#3b82f6' : 'var(--text-muted)'} strokeWidth={2} fill={`url(#grad-${h})`} name="Forecast" />
                </AreaChart>
              </ResponsiveContainer>
              <div className="flex justify-between items-center text-xs mt-3 pt-2.5 border-t border-bg-border/30">
                <span className="text-text-muted">{pts.length} interval points</span>
                <span className="text-electric-400 hover:text-electric-300 font-semibold cursor-pointer flex items-center gap-0.5" onClick={() => setHorizon(h)}>
                  Focus {h} &rarr;
                </span>
              </div>
            </ChartCard>
          )
        })}
      </div>

      {/* Model metrics table */}
      {data && (
        <ChartCard title="Model Performance Metrics" subtitle="AutoML accuracy rankings across standard test sets">
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-bg-border/55 pb-2">
                  {['Horizon', 'Model Baseline', 'Data Points', 'Avg Consumption', 'Peak Load'].map(col => (
                    <th key={col} className="label py-3 pr-4 text-left font-bold">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-bg-border/30">
                {Object.entries(data.forecasts).map(([h, fc]) => {
                  const avg = fc.points.reduce((s, p) => s + p.forecast, 0) / Math.max(fc.points.length, 1)
                  const peak = Math.max(...fc.points.map(p => p.upper_bound))
                  return (
                    <tr key={h} className="text-text-secondary hover:bg-bg-hover/20 transition-colors">
                      <td className="py-3 pr-4 font-bold text-electric-400 capitalize">{h} Horizon</td>
                      <td className="py-3 pr-4 font-mono">{fc.model_name}</td>
                      <td className="py-3 pr-4 text-text-muted">{fc.points.length.toLocaleString()} records</td>
                      <td className="py-3 pr-4 font-semibold text-text-primary">{formatMW(avg)}</td>
                      <td className="py-3 pr-4 font-semibold text-text-primary">{formatMW(peak)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </ChartCard>
      )}
    </div>
  )
}

export default Forecast
