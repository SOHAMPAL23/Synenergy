import api from './api'
import axios from 'axios'

// ── Types ───────────────────────────────────────────────────────────────────

export interface ForecastPoint {
  timestamp: string
  forecast: number
  lower_bound: number
  upper_bound: number
}

export interface ForecastHorizon {
  horizon: string
  model_name: string
  points: ForecastPoint[]
  generated_at: string
}

export interface ForecastsResponse {
  forecasts: Record<string, ForecastHorizon>
  best_model: string
}

export interface AnomalyPoint {
  timestamp: string
  value: number
  is_anomaly: boolean
  anomaly_score: number
  severity: string
}

export interface MethodBreakdown {
  method: string
  count: number
}

export interface AnomaliesResponse {
  total_records: number
  anomaly_count: number
  anomaly_rate_pct: number
  points: AnomalyPoint[]
  method_breakdown: MethodBreakdown[]
  generated_at: string
}

export interface FeatureImportanceItem {
  feature: string
  mean_abs_shap: number
  rank: number
}

export interface ExplanationResponse {
  model_name: string
  explainer_type: string
  feature_importances: FeatureImportanceItem[]
  top_features: string[]
  generated_at: string
}

export interface ModelMetrics {
  rmse: number
  mae: number
  mape: number
}

export interface TrainResponse {
  status: string
  best_model: string
  metrics: Record<string, ModelMetrics>
  training_time_seconds: number
  message: string
}

export interface RecommendationItem {
  id?: string
  category: string
  priority: 'HIGH' | 'MEDIUM' | 'LOW'
  title: string
  description: string
  estimated_saving_pct: number
  action_items: string[]
}

export interface RecommendationsResponse {
  total: number
  high_priority: number
  medium_priority: number
  low_priority: number
  recommendations: RecommendationItem[]
}

export interface UploadResponse {
  upload_id: string
  filename: string
  rows_loaded: number
  rows_valid: number
  rows_rejected: number
  columns: string[]
  time_range: { start: string; end: string }
  warnings: string[]
  message: string
}

// ── Mock data generators ─────────────────────────────────────────────────────

function generateMockForecastPoints(n: number, baseValue = 48000, trending = true): ForecastPoint[] {
  const pts: ForecastPoint[] = []
  const start = new Date()
  for (let i = 0; i < n; i++) {
    const ts = new Date(start.getTime() + i * 3600000)
    const noise = (Math.random() - 0.5) * 3000
    const trend = trending ? i * 20 : 0
    const val = baseValue + noise + trend + Math.sin(i / 4) * 2000
    pts.push({
      timestamp: ts.toISOString(),
      forecast: Math.round(val),
      lower_bound: Math.round(val * 0.94),
      upper_bound: Math.round(val * 1.06),
    })
  }
  return pts
}

function generateMockAnomalyPoints(n: number): AnomalyPoint[] {
  const pts: AnomalyPoint[] = []
  const start = new Date(Date.now() - n * 3600000)
  for (let i = 0; i < n; i++) {
    const ts = new Date(start.getTime() + i * 3600000)
    const isAnom = Math.random() < 0.06
    const score = isAnom ? 0.5 + Math.random() * 0.5 : Math.random() * 0.4
    const severity = isAnom ? (score > 0.8 ? 'high' : score > 0.6 ? 'medium' : 'low') : 'none'
    pts.push({
      timestamp: ts.toISOString(),
      value: 45000 + (Math.random() - 0.5) * 10000 + (isAnom ? 20000 * Math.sign(Math.random() - 0.5) : 0),
      is_anomaly: isAnom,
      anomaly_score: parseFloat(score.toFixed(4)),
      severity,
    })
  }
  return pts
}

// ── Mock responses ────────────────────────────────────────────────────────────

const MOCK_FORECASTS: ForecastsResponse = {
  best_model: 'XGBoost',
  forecasts: {
    '24h': {
      horizon: '24h',
      model_name: 'XGBoost',
      points: generateMockForecastPoints(24),
      generated_at: new Date().toISOString(),
    },
    '7d': {
      horizon: '7d',
      model_name: 'XGBoost',
      points: generateMockForecastPoints(7 * 24),
      generated_at: new Date().toISOString(),
    },
    '30d': {
      horizon: '30d',
      model_name: 'XGBoost',
      points: generateMockForecastPoints(30 * 24),
      generated_at: new Date().toISOString(),
    },
  },
}

const MOCK_ANOMALY_PTS = generateMockAnomalyPoints(720)
const MOCK_ANOMALIES: AnomaliesResponse = {
  total_records: 720,
  anomaly_count: MOCK_ANOMALY_PTS.filter(p => p.is_anomaly).length,
  anomaly_rate_pct: parseFloat(
    ((MOCK_ANOMALY_PTS.filter(p => p.is_anomaly).length / 720) * 100).toFixed(2)
  ),
  points: MOCK_ANOMALY_PTS,
  method_breakdown: [
    { method: 'zscore', count: 28 },
    { method: 'iqr', count: 22 },
    { method: 'isolation_forest', count: 31 },
    { method: 'lof', count: 19 },
    { method: 'one_class_svm', count: 25 },
  ],
  generated_at: new Date().toISOString(),
}

const MOCK_EXPLANATIONS: ExplanationResponse = {
  model_name: 'XGBoost',
  explainer_type: 'TreeExplainer',
  feature_importances: [
    { feature: 'hour_of_day', mean_abs_shap: 4821.3, rank: 1 },
    { feature: 'day_of_week', mean_abs_shap: 3102.7, rank: 2 },
    { feature: 'lag_1', mean_abs_shap: 2893.1, rank: 3 },
    { feature: 'rolling_mean_24h', mean_abs_shap: 2211.4, rank: 4 },
    { feature: 'month', mean_abs_shap: 1876.2, rank: 5 },
    { feature: 'lag_24', mean_abs_shap: 1523.8, rank: 6 },
    { feature: 'is_weekend', mean_abs_shap: 1102.5, rank: 7 },
    { feature: 'rolling_std_24h', mean_abs_shap: 892.3, rank: 8 },
    { feature: 'rolling_mean_7d', mean_abs_shap: 743.1, rank: 9 },
    { feature: 'quarter', mean_abs_shap: 521.6, rank: 10 },
  ],
  top_features: ['hour_of_day', 'day_of_week', 'lag_1', 'rolling_mean_24h', 'month'],
  generated_at: new Date().toISOString(),
}

const MOCK_RECOMMENDATIONS: RecommendationsResponse = {
  total: 6,
  high_priority: 2,
  medium_priority: 3,
  low_priority: 1,
  recommendations: [
    {
      category: 'Peak Shifting',
      priority: 'HIGH',
      title: 'Shift Heavy Loads Off-Peak',
      description: 'Forecasted consumption exceeds 58,000 MW during 08:00–10:00. Rescheduling energy-intensive processes to 22:00–06:00 can reduce peak demand charges.',
      estimated_saving_pct: 18.5,
      action_items: ['Reschedule HVAC pre-cooling to overnight hours', 'Delay batch processing jobs to off-peak windows', 'Install smart load controllers on industrial equipment'],
    },
    {
      category: 'Anomaly Response',
      priority: 'HIGH',
      title: 'Investigate Consumption Spike at 14:23',
      description: 'An anomaly with severity score 0.92 was detected. Unexplained 23% surge above baseline requires immediate investigation.',
      estimated_saving_pct: 12.3,
      action_items: ['Audit sub-meter readings', 'Check for equipment malfunction', 'Review HVAC scheduling'],
    },
    {
      category: 'Energy Efficiency',
      priority: 'MEDIUM',
      title: 'Optimize HVAC Scheduling',
      description: 'Pattern analysis shows HVAC accounts for 34% of consumption. Predictive pre-cooling can reduce runtime during expensive peak hours.',
      estimated_saving_pct: 9.1,
      action_items: ['Install predictive thermostat controller', 'Align HVAC schedule with weather forecast API'],
    },
    {
      category: 'Demand Response',
      priority: 'MEDIUM',
      title: 'Enroll in Demand Response Program',
      description: 'Grid operator demand response events occur 12–18 times annually. Automated curtailment can yield significant bill credits.',
      estimated_saving_pct: 6.8,
      action_items: ['Contact utility demand response program', 'Install automated curtailment relay switches'],
    },
    {
      category: 'Monitoring',
      priority: 'MEDIUM',
      title: 'Add Sub-Metering to High-Consumption Zones',
      description: 'Granular metering reveals 3 zones with unexplained consumption spikes. Better instrumentation will enable targeted optimization.',
      estimated_saving_pct: 4.2,
      action_items: ['Deploy smart sub-meters on zones A, C, and F'],
    },
    {
      category: 'Renewable Integration',
      priority: 'LOW',
      title: 'Evaluate On-Site Solar PV',
      description: 'Solar irradiance data suggests a 200 kWp rooftop system could offset 15% of daytime consumption at the current location.',
      estimated_saving_pct: 3.5,
      action_items: ['Commission solar feasibility study', 'Explore net metering agreements with utility'],
    },
  ],
}

// ── Service ──────────────────────────────────────────────────────────────────

const isMockError = (e: unknown) => {
  if (axios.isAxiosError(e)) {
    const status = e.response?.status
    return !status || status === 404 || status === 422 || status === 400
  }
  return true
}

export const mlService = {
  async uploadCSV(file: File): Promise<UploadResponse> {
    const form = new FormData()
    form.append('file', file)
    const { data } = await api.post<UploadResponse>('/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },

  async train(): Promise<TrainResponse> {
    const { data } = await api.post<TrainResponse>('/train', {})
    return data
  },

  async getForecasts(): Promise<ForecastsResponse> {
    try {
      const { data } = await api.get<ForecastsResponse>('/forecast')
      return data
    } catch (e) {
      if (isMockError(e)) return MOCK_FORECASTS
      throw e
    }
  },

  async getAnomalies(): Promise<AnomaliesResponse> {
    try {
      const { data } = await api.get<AnomaliesResponse>('/anomalies')
      return data
    } catch (e) {
      if (isMockError(e)) return MOCK_ANOMALIES
      throw e
    }
  },

  async getExplanations(): Promise<ExplanationResponse> {
    try {
      const { data } = await api.get<ExplanationResponse>('/explanations')
      return data
    } catch (e) {
      if (isMockError(e)) return MOCK_EXPLANATIONS
      throw e
    }
  },

  async getRecommendations(): Promise<RecommendationsResponse> {
    try {
      const { data } = await api.get<RecommendationsResponse>('/recommendations')
      return data
    } catch (e) {
      if (isMockError(e)) return MOCK_RECOMMENDATIONS
      throw e
    }
  },
}
