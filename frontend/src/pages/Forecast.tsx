import React, { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend, ReferenceLine,
} from 'recharts'
import { Calendar } from 'lucide-react'
import { mlService } from '../services/ml'
import PageHeader, { ChartCard } from '../components/ui/PageHeader'
import { formatDateTime, formatMW, downsample } from '../utils/format'

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
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4 animate-slide-up">
        {/* Sliders Card */}
        <div className="glass-card p-5 border border-bg-border lg:col-span-2">
          <div className="flex items-center gap-2 mb-4">
            <span className="p-1.5 rounded-lg bg-electric-500/10 text-electric-400">
              <Zap size={16} />
            </span>
            <div>
              <h4 className="text-sm font-bold text-text-primary">What-If Forecast Simulator</h4>
              <p className="text-xs text-text-muted">Simulate operational offsets and clean grid integrations on the forecast horizon.</p>
            </div>
          </div>
          
          <div className="space-y-4">
            {/* Slider 1 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-text-secondary font-medium">Peak Shaving Intensity</span>
                <span className="text-electric-400 font-semibold">{peakShaving}% reduction</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="20" 
                value={peakShaving} 
                onChange={e => setPeakShaving(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-electric-500"
              />
              <p className="text-[10px] text-text-muted mt-1">Shaves loads that exceed the median forecast baseline.</p>
            </div>

            {/* Slider 2 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-text-secondary font-medium">Solar & Wind Offset Shift</span>
                <span className="text-emerald-400 font-semibold">{solarShift}% capacity</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="30" 
                value={solarShift} 
                onChange={e => setSolarShift(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-emerald-500"
              />
              <p className="text-[10px] text-text-muted mt-1">Offsets consumption during daylight generation windows (08:00 - 18:00).</p>
            </div>

            {/* Slider 3 */}
            <div>
              <div className="flex justify-between text-xs mb-1.5">
                <span className="text-text-secondary font-medium">Grid Efficiency Factors</span>
                <span className="text-cyan-400 font-semibold">{gridOpt}% gain</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="10" 
                value={gridOpt} 
                onChange={e => setGridOpt(Number(e.target.value))}
                className="w-full h-1.5 bg-bg-primary rounded-lg appearance-none cursor-pointer accent-cyan-500"
              />
              <p className="text-[10px] text-text-muted mt-1">Applies standard continuous grid efficiency adjustments across all hours.</p>
            </div>
          </div>
        </div>

        {/* Dynamic ROI Metrics Card */}
        <div className="glass-card p-5 border border-emerald-500/25 bg-emerald-500/5 flex flex-col justify-between">
          <div>
            <h4 className="text-sm font-bold text-emerald-400 mb-0.5">Estimated Simulation ROI</h4>
            <p className="text-xs text-text-muted mb-4">Calculated offsets based on current slider configurations.</p>
            
            <div className="space-y-3">
              <div className="flex justify-between items-center border-b border-bg-border/40 pb-2">
                <span className="text-xs text-text-secondary">Peak Shaved</span>
                <span className="text-sm font-bold text-text-primary">{formatMW(peakShavedVal)}</span>
              </div>
              
              <div className="flex justify-between items-center border-b border-bg-border/40 pb-2">
                <span className="text-xs text-text-secondary">Est. Cost Savings</span>
                <span className="text-sm font-bold text-emerald-400">+${Math.round(estDailySavings).toLocaleString()}/day</span>
              </div>

              <div className="flex justify-between items-center pb-2">
                <span className="text-xs text-text-secondary">CO₂ Offset</span>
                <span className="text-sm font-bold text-cyan-400">{Math.round(estCarbonAbated).toLocaleString()} kg/day</span>
              </div>
            </div>
          </div>

          <div className="text-[10px] text-text-muted mt-4 border-t border-bg-border/40 pt-2.5">
            *Simulation calculations are generated locally on the selected forecast interval.
          </div>
        </div>
      </div>

      {/* Horizon comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {(['24h', '7d', '30d'] as Horizon[]).map(h => {
          const pts = downsample(data?.forecasts?.[h]?.points ?? [], 30)
          const hData = pts.map(p => ({ ts: p.timestamp, v: p.forecast }))
          return (
            <ChartCard key={h} title={`${h} Outlook`} subtitle={data?.forecasts?.[h]?.model_name ?? '—'}>
              <ResponsiveContainer width="100%" height={140}>
                <LineChart data={hData}>
                  <XAxis dataKey="ts" hide />
                  <YAxis hide />
                  <Tooltip
                    formatter={(v: any) => formatMW(v as number)}
                    labelFormatter={(label: any) => formatDateTime(String(label))}
                    contentStyle={{ background: 'var(--bg-card)', border: '1px solid var(--bg-border)', borderRadius: 8, fontSize: 11 }}
                    labelStyle={{ color: 'var(--text-muted)' }}
                  />
                  <Line type="monotone" dataKey="v" stroke={h === horizon ? '#3b82f6' : 'var(--text-muted)'} strokeWidth={2} dot={false} name="Forecast" />
                </LineChart>
              </ResponsiveContainer>
              <div className="flex justify-between text-xs mt-2">
                <span className="text-text-muted">{pts.length} points</span>
                <span className="text-electric-500 font-medium cursor-pointer" onClick={() => setHorizon(h)}>View →</span>
              </div>
            </ChartCard>
          )
        })}
      </div>

      {/* Model metrics table */}
      {data && (
        <ChartCard title="Model Performance Metrics">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-bg-border">
                  {['Horizon', 'Model', 'Points', 'Avg (MW)', 'Peak (MW)'].map(col => (
                    <th key={col} className="label py-2 pr-4 text-left">{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-bg-border/50">
                {Object.entries(data.forecasts).map(([h, fc]) => {
                  const avg = fc.points.reduce((s, p) => s + p.forecast, 0) / Math.max(fc.points.length, 1)
                  const peak = Math.max(...fc.points.map(p => p.upper_bound))
                  return (
                    <tr key={h} className="text-text-secondary hover:bg-bg-hover/30 transition-colors">
                      <td className="py-2 pr-4 font-semibold text-electric-500">{h}</td>
                      <td className="py-2 pr-4">{fc.model_name}</td>
                      <td className="py-2 pr-4">{fc.points.length.toLocaleString()}</td>
                      <td className="py-2 pr-4">{formatMW(avg)}</td>
                      <td className="py-2 pr-4">{formatMW(peak)}</td>
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
