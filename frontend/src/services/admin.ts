import api from './api'

// ── Types ─────────────────────────────────────────────────────────────────────

export interface UserItem {
  id: string
  email: string
  full_name: string
  role: 'admin' | 'analyst' | 'viewer'
  is_active: boolean
  is_verified: boolean
  created_at: string
  last_login: string | null
}

export interface UsersListResponse {
  total: number
  users: UserItem[]
}

export interface SystemStats {
  total_users: number
  active_users: number
  total_energy_records: number
  total_forecasts: number
  total_recommendations: number
  models_trained: number
}

export interface UpdateUserPayload {
  role?: 'admin' | 'analyst' | 'viewer'
  is_active?: boolean
}

// ── Admin Service ─────────────────────────────────────────────────────────────

export const adminService = {
  /** List all users (admin only) */
  async listUsers(): Promise<UsersListResponse> {
    const { data } = await api.get<UsersListResponse>('/admin/users')
    return data
  },

  /** Update a user's role or active status (admin only) */
  async updateUser(userId: string, payload: UpdateUserPayload): Promise<UserItem> {
    const { data } = await api.patch<UserItem>(`/admin/users/${userId}`, payload)
    return data
  },

  /** Deactivate a user account (admin only) */
  async deactivateUser(userId: string): Promise<void> {
    await api.patch(`/admin/users/${userId}`, { is_active: false })
  },

  /** Reactivate a user account (admin only) */
  async activateUser(userId: string): Promise<void> {
    await api.patch(`/admin/users/${userId}`, { is_active: true })
  },

  /** Get platform-level system statistics (admin only) */
  async getSystemStats(): Promise<SystemStats> {
    const { data } = await api.get<SystemStats>('/admin/stats')
    return data
  },
}
