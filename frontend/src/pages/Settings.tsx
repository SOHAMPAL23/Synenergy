import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { useAuth } from '../contexts/AuthContext'
import { useTheme } from '../contexts/ThemeContext'
import PageHeader from '../components/ui/PageHeader'
import { User, Bell, Shield, Database, Palette, ChevronRight, CheckCircle } from 'lucide-react'

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
  <div className="glass-card p-6">
    <h3 className="text-sm font-semibold text-text-primary mb-4">{title}</h3>
    {children}
  </div>
)

const SettingRow: React.FC<{
  label: string;
  description?: string;
  children: React.ReactNode;
}> = ({ label, description, children }) => (
  <div className="flex items-center justify-between py-3 border-b border-bg-border/50 last:border-0">
    <div>
      <p className="text-sm text-text-primary font-medium">{label}</p>
      {description && <p className="text-xs text-text-muted mt-0.5">{description}</p>}
    </div>
    {children}
  </div>
)

const Settings: React.FC = () => {
  const { user } = useAuth()
  const { theme, setTheme } = useTheme()
  const [saved, setSaved] = useState(false)
  const [notifs, setNotifs] = useState({
    anomalyAlerts: true,
    weeklyReport: true,
    modelTraining: false,
    peakWarnings: true,
  })

  const tabs = [
    { id: 'profile', label: 'Profile', icon: User },
    { id: 'notifications', label: 'Notifications', icon: Bell },
    { id: 'security', label: 'Security', icon: Shield },
    { id: 'data', label: 'Data', icon: Database },
    { id: 'appearance', label: 'Appearance', icon: Palette },
  ]

  const [activeTab, setActiveTab] = useState('profile')

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  return (
    <div className="page-container">
      <PageHeader
        title="Settings"
        subtitle="Manage your account, notifications, and preferences"
        actions={
          <button onClick={handleSave} className="btn-primary flex items-center gap-2 text-sm">
            {saved ? <><CheckCircle size={14} /> Saved!</> : 'Save Changes'}
          </button>
        }
      />

      <div className="flex flex-col md:flex-row gap-6">
        {/* Sidebar tabs */}
        <div className="w-full md:w-48 flex-shrink-0">
          <nav className="space-y-0.5">
            {tabs.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`nav-item w-full text-left ${activeTab === id ? 'active' : ''}`}
              >
                <Icon size={16} />
                {label}
                {activeTab === id && <ChevronRight size={12} className="ml-auto text-electric-500" />}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex-1 space-y-4"
        >
          {activeTab === 'profile' && (
            <>
              <Section title="Account Information">
                <div className="space-y-4">
                  <div className="flex items-center gap-4 mb-5">
                    <div className="w-14 h-14 rounded-2xl bg-electric-gradient flex items-center justify-center text-white text-2xl font-display font-bold shadow-glow-blue">
                      {user?.full_name?.charAt(0) ?? 'U'}
                    </div>
                    <div>
                      <p className="text-text-primary font-semibold">{user?.full_name}</p>
                      <p className="text-sm text-text-secondary">{user?.email}</p>
                      <span className="badge-info capitalize mt-1 inline-block">{user?.role}</span>
                    </div>
                  </div>
                  {[
                    { label: 'Full Name', value: user?.full_name },
                    { label: 'Email', value: user?.email },
                    { label: 'Role', value: user?.role },
                    { label: 'Account Status', value: user?.is_active ? 'Active' : 'Inactive' },
                    { label: 'Member Since', value: user?.created_at?.split('T')[0] },
                  ].map(field => (
                    <SettingRow key={field.label} label={field.label}>
                      <span className="text-sm text-text-secondary capitalize">{field.value}</span>
                    </SettingRow>
                  ))}
                </div>
              </Section>

              <Section title="API Access">
                <SettingRow label="API Base URL" description="Backend endpoint for all ML requests">
                  <span className="font-mono text-xs text-electric-500 bg-bg-hover px-2 py-1 rounded">
                    {window.location.origin}/api/v1
                  </span>
                </SettingRow>
                <SettingRow label="Auth Type" description="Token-based JWT authentication">
                  <span className="badge-info">Bearer JWT</span>
                </SettingRow>
              </Section>
            </>
          )}

          {activeTab === 'notifications' && (
            <Section title="Notification Preferences">
              {[
                { key: 'anomalyAlerts', label: 'Anomaly Alerts', description: 'Get notified when anomalies are detected in energy data' },
                { key: 'weeklyReport', label: 'Weekly Report', description: 'Receive a weekly energy consumption summary' },
                { key: 'modelTraining', label: 'Training Complete', description: 'Alert when ML model training finishes' },
                { key: 'peakWarnings', label: 'Peak Demand Warnings', description: 'Early warning when peak consumption is forecasted' },
              ].map(({ key, label, description }) => (
                <SettingRow key={key} label={label} description={description}>
                  <button
                    onClick={() => setNotifs(n => ({ ...n, [key]: !n[key as keyof typeof n] }))}
                    className={`relative w-11 h-6 rounded-full transition-all ${
                      notifs[key as keyof typeof notifs] ? 'bg-electric-500' : 'bg-bg-border'
                    }`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow transition-transform ${
                      notifs[key as keyof typeof notifs] ? 'translate-x-5' : 'translate-x-0'
                    }`} />
                  </button>
                </SettingRow>
              ))}
            </Section>
          )}

          {activeTab === 'security' && (
            <Section title="Security">
              <SettingRow label="Password" description="Last changed: unknown">
                <button className="btn-secondary text-xs">Change Password</button>
              </SettingRow>
              <SettingRow label="Two-Factor Authentication" description="Add an extra layer of security">
                <span className="badge-medium">Not configured</span>
              </SettingRow>
              <SettingRow label="Active Sessions" description="Devices currently signed in">
                <span className="text-sm text-text-secondary">1 session</span>
              </SettingRow>
            </Section>
          )}

          {activeTab === 'data' && (
            <Section title="Data Management">
              <SettingRow label="Upload Limit" description="Maximum CSV file size">
                <span className="text-sm text-text-secondary">50 MB</span>
              </SettingRow>
              <SettingRow label="Supported Format" description="Required columns">
                <span className="font-mono text-xs text-electric-500 bg-bg-hover px-2 py-1 rounded">
                  DE_load_actual_entsoe_transparency
                </span>
              </SettingRow>
              <SettingRow label="Data Retention" description="How long energy records are stored">
                <span className="text-sm text-text-secondary">Indefinite</span>
              </SettingRow>
              <SettingRow label="ML Model Storage" description="Trained model artifacts location">
                <span className="font-mono text-xs text-text-muted">ml/outputs/models/</span>
              </SettingRow>
            </Section>
          )}

          {activeTab === 'appearance' && (
            <Section title="Appearance">
              <SettingRow label="Theme" description="Interface color scheme">
                <div className="flex gap-2">
                  {[
                    { id: 'gradient', label: 'Gradient Mode' },
                    { id: 'light', label: 'Light Mode' },
                  ].map(t => (
                    <button
                      key={t.id}
                      onClick={() => setTheme(t.id as any)}
                      className={`btn-secondary text-xs ${theme === t.id ? 'border-electric-500/40 text-text-primary bg-bg-hover font-semibold' : ''}`}
                    >
                      {t.label}
                    </button>
                  ))}
                </div>
              </SettingRow>
              <SettingRow label="Chart Color Palette" description="Visualization color scheme">
                <div className="flex gap-1.5">
                  {['#3b82f6', '#22d3ee', '#8b5cf6', '#22c55e', '#f59e0b'].map(c => (
                    <div key={c} className="w-5 h-5 rounded-full border border-bg-border" style={{ background: c }} />
                  ))}
                </div>
              </SettingRow>
              <SettingRow label="Sidebar" description="Navigation panel default state">
                <span className="text-sm text-text-secondary">Expanded</span>
              </SettingRow>
            </Section>
          )}
        </motion.div>
      </div>
    </div>
  )
}

export default Settings
