export function formatNumber(n: number, decimals = 1): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(decimals)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(decimals)}K`
  return n.toFixed(decimals)
}

export function formatMW(mw: number): string {
  return `${formatNumber(mw)} MW`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

export function formatDateTime(iso: string): string {
  return new Date(iso).toLocaleString('en-US', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
}

export function cn(...classes: (string | undefined | false | null)[]): string {
  return classes.filter(Boolean).join(' ')
}

export function getPriorityColor(priority: string): string {
  switch (priority) {
    case 'HIGH':   return 'badge-high'
    case 'MEDIUM': return 'badge-medium'
    case 'LOW':    return 'badge-low'
    default:       return 'badge-info'
  }
}

export function getSeverityColor(severity: string): string {
  switch (severity) {
    case 'high':   return '#ef4444'
    case 'medium': return '#f59e0b'
    case 'low':    return '#22c55e'
    default:       return '#3b82f6'
  }
}

export function downsample<T>(arr: T[], maxPoints: number): T[] {
  if (arr.length <= maxPoints) return arr
  const step = Math.ceil(arr.length / maxPoints)
  return arr.filter((_, i) => i % step === 0)
}
