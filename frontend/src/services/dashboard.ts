import api from './api'

export interface DashboardStats {
  total_records: number
  date_range_start: string | null
  date_range_end: string | null
  avg_consumption_mw: number
  max_consumption_mw: number
  min_consumption_mw: number
  best_model: string | null
  forecast_horizons_available: string[]
  recommendations_count: number
  high_priority_recommendations: number
}

export interface ForecastPoint {
  timestamp: string
  forecast: number
  lower_bound: number
  upper_bound: number
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

export interface DashboardResponse {
  user: {
    id: string
    email: string
    full_name: string
    role: string
    is_active: boolean
    is_verified: boolean
    created_at: string
  }
  stats: DashboardStats
  recent_forecasts: ForecastPoint[] | null
  top_recommendations: RecommendationItem[]
}

export const dashboardService = {
  async getDashboard(): Promise<DashboardResponse> {
    const { data } = await api.get<DashboardResponse>('/dashboard')
    return data
  },
}
