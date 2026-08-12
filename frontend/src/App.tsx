import React, { Suspense, lazy } from 'react'
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { AuthProvider } from './contexts/AuthContext'
import { ThemeProvider } from './contexts/ThemeContext'
import { ProtectedRoute, PublicRoute } from './routes'
import AuthLayout from './layouts/AuthLayout'
import AppLayout from './layouts/AppLayout'

// Lazy load pages
const Login = lazy(() => import('./pages/Login'))
const Register = lazy(() => import('./pages/Register'))
const Dashboard = lazy(() => import('./pages/Dashboard'))
const Forecast = lazy(() => import('./pages/Forecast'))
const Anomaly = lazy(() => import('./pages/Anomaly'))
const Explainability = lazy(() => import('./pages/Explainability'))
const Optimization = lazy(() => import('./pages/Optimization'))
const Settings = lazy(() => import('./pages/Settings'))
const Upload = lazy(() => import('./pages/Upload'))
const Admin = lazy(() => import('./pages/Admin'))

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

// Simple Loading Fallback
const PageLoader = () => (
  <div className="flex items-center justify-center min-h-[50vh]">
    <div className="w-10 h-10 border-4 border-electric-500/20 border-t-electric-500 rounded-full animate-spin"></div>
  </div>
)

const App: React.FC = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <AuthProvider>
        <BrowserRouter>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              {/* Public routes */}
              <Route element={<PublicRoute />}>
                <Route element={<AuthLayout />}>
                  <Route path="/login"    element={<Login />} />
                  <Route path="/register" element={<Register />} />
                </Route>
              </Route>

              {/* Protected routes */}
              <Route element={<ProtectedRoute />}>
                <Route element={<AppLayout />}>
                  <Route path="/dashboard"      element={<Dashboard />} />
                  <Route path="/forecast"       element={<Forecast />} />
                  <Route path="/anomalies"      element={<Anomaly />} />
                  <Route path="/explainability" element={<Explainability />} />
                  <Route path="/optimization"   element={<Optimization />} />
                  <Route path="/upload"         element={<Upload />} />
                  <Route path="/admin"          element={<Admin />} />
                  <Route path="/settings"       element={<Settings />} />
                </Route>
              </Route>

              {/* Redirects */}
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </ThemeProvider>
  </QueryClientProvider>
)

export default App
