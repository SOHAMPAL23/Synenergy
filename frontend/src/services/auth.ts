import api from './api'

export interface LoginPayload { email: string; password: string }
export interface RegisterPayload { email: string; password: string; full_name: string; role?: string }

export interface TokenResponse {
  access_token: string
  refresh_token: string
  token_type: string
  expires_in: number
}

export interface UserProfile {
  id: string
  email: string
  full_name: string
  role: string
  is_active: boolean
  is_verified: boolean
  created_at: string
}

export const authService = {
  async login(payload: LoginPayload): Promise<TokenResponse> {
    const { data } = await api.post<TokenResponse>('/auth/login', payload)
    localStorage.setItem('access_token', data.access_token)
    localStorage.setItem('refresh_token', data.refresh_token)
    return data
  },

  async register(payload: RegisterPayload): Promise<UserProfile> {
    const { data } = await api.post<UserProfile>('/auth/register', payload)
    return data
  },

  async me(): Promise<UserProfile> {
    const { data } = await api.get<UserProfile>('/auth/me')
    return data
  },

  logout() {
    localStorage.removeItem('access_token')
    localStorage.removeItem('refresh_token')
  },

  isAuthenticated(): boolean {
    return !!localStorage.getItem('access_token')
  },
}
