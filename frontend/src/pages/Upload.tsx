import React, { useCallback, useRef, useState } from 'react'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Upload, FileText, CheckCircle, XCircle, AlertTriangle,
  Clock, Layers, Trash2, CloudUpload,
} from 'lucide-react'
import { mlService } from '../services/ml'
import type { UploadResponse } from '../services/ml'
import PageHeader, { ChartCard, Spinner } from '../components/ui/PageHeader'
import { formatDateTime } from '../utils/format'

// ── Upload History (session-only, not persisted) ──────────────────────────────

interface UploadRecord {
  id: string
  filename: string
  rows_valid: number
  rows_rejected: number
  time_range: { start: string; end: string }
  columns: string[]
  warnings: string[]
  timestamp: string
}

// ── Drag & Drop Zone ──────────────────────────────────────────────────────────

interface DropZoneProps {
  onFile: (file: File) => void
  uploading: boolean
}

const DropZone: React.FC<DropZoneProps> = ({ onFile, uploading }) => {
  const [dragging, setDragging] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) onFile(file)
  }, [onFile])

  const handleChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) onFile(file)
    e.target.value = ''
  }, [onFile])

  return (
    <motion.div
      animate={{ scale: dragging ? 1.01 : 1 }}
      onDragOver={e => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`relative border-2 border-dashed rounded-2xl p-12 text-center cursor-pointer transition-all ${
        dragging
          ? 'border-electric-400 bg-electric-500/10 shadow-glow-blue'
          : 'border-bg-border hover:border-electric-500/50 hover:bg-electric-500/5'
      }`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={handleChange}
        disabled={uploading}
      />

      <AnimatePresence mode="wait">
        {uploading ? (
          <motion.div
            key="uploading"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9 }}
            className="flex flex-col items-center gap-3"
          >
            <Spinner size={40} />
            <p className="text-text-primary font-semibold">Uploading and processing…</p>
            <p className="text-sm text-text-muted">Validating schema and inserting records</p>
          </motion.div>
        ) : (
          <motion.div
            key="idle"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="flex flex-col items-center gap-3"
          >
            <div className="w-16 h-16 rounded-2xl bg-electric-gradient flex items-center justify-center shadow-glow-blue">
              <CloudUpload size={28} className="text-white" />
            </div>
            <div>
              <p className="text-text-primary font-semibold text-lg">
                {dragging ? 'Drop your CSV here' : 'Drag & drop your CSV file'}
              </p>
              <p className="text-sm text-text-muted mt-1">
                or <span className="text-electric-400 font-medium">click to browse</span> · Max 50MB
              </p>
            </div>
            <div className="flex items-center gap-2 mt-2 text-xs text-text-muted">
              <FileText size={14} />
              Required column:
              <code className="font-mono text-electric-400 bg-bg-primary px-1.5 py-0.5 rounded">
                DE_load_actual_entsoe_transparency
              </code>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  )
}

// ── Upload Result Card ────────────────────────────────────────────────────────

const UploadResult: React.FC<{ result: UploadResponse }> = ({ result }) => {
  const success = result.rows_valid > 0
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={`glass-card p-5 border ${success ? 'border-success-500/30 bg-success-500/5' : 'border-danger-500/30 bg-danger-500/5'}`}
    >
      <div className="flex items-start gap-3">
        {success
          ? <CheckCircle size={20} className="text-success-400 flex-shrink-0 mt-0.5" />
          : <XCircle size={20} className="text-danger-400 flex-shrink-0 mt-0.5" />}
        <div className="flex-1 min-w-0">
          <p className={`font-semibold ${success ? 'text-success-600' : 'text-danger-600'}`}>
            {result.message}
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-3">
            {[
              { label: 'Valid Rows', value: result.rows_valid.toLocaleString(), color: 'text-success-500' },
              { label: 'Rejected Rows', value: result.rows_rejected.toLocaleString(), color: result.rows_rejected > 0 ? 'text-warning-500' : 'text-text-muted' },
              { label: 'Date Start', value: result.time_range.start?.split('T')[0] ?? '—', color: 'text-text-secondary' },
              { label: 'Date End', value: result.time_range.end?.split('T')[0] ?? '—', color: 'text-text-secondary' },
            ].map(stat => (
              <div key={stat.label} className="bg-bg-primary/60 rounded-lg px-3 py-2">
                <p className="text-xs text-text-muted mb-0.5">{stat.label}</p>
                <p className={`text-sm font-semibold ${stat.color}`}>{stat.value}</p>
              </div>
            ))}
          </div>

          {/* Columns */}
          <div className="mt-3">
            <p className="text-xs text-text-muted mb-1.5">Detected Columns ({result.columns.length})</p>
            <div className="flex flex-wrap gap-1.5">
              {result.columns.slice(0, 12).map(col => (
                <span key={col} className="font-mono text-xs bg-bg-primary border border-bg-border text-text-secondary px-2 py-0.5 rounded">
                  {col}
                </span>
              ))}
              {result.columns.length > 12 && (
                <span className="text-xs text-text-muted">+{result.columns.length - 12} more</span>
              )}
            </div>
          </div>

          {/* Warnings */}
          {result.warnings.length > 0 && (
            <div className="mt-3">
              {result.warnings.map((w, i) => (
                <div key={i} className="flex items-center gap-2 text-xs text-warning-500 mt-1">
                  <AlertTriangle size={12} />
                  {w}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}

// ── Main Upload Page ──────────────────────────────────────────────────────────

const UploadPage: React.FC = () => {
  const queryClient = useQueryClient()
  const [lastResult, setLastResult] = useState<UploadResponse | null>(null)
  const [uploadHistory, setUploadHistory] = useState<UploadRecord[]>([])
  const [error, setError] = useState<string | null>(null)

  const uploadMutation = useMutation({
    mutationFn: (file: File) => mlService.uploadCSV(file),
    onSuccess: (data) => {
      setLastResult(data)
      setError(null)
      setUploadHistory(prev => [{
        id: data.upload_id,
        filename: data.filename,
        rows_valid: data.rows_valid,
        rows_rejected: data.rows_rejected,
        time_range: data.time_range,
        columns: data.columns,
        warnings: data.warnings,
        timestamp: new Date().toISOString(),
      }, ...prev.slice(0, 9)])
      // Invalidate dashboard and energy queries
      queryClient.invalidateQueries({ queryKey: ['dashboard'] })
    },
    onError: (e: any) => {
      setError(e?.response?.data?.detail ?? 'Upload failed. Please check your CSV format.')
      setLastResult(null)
    },
  })

  const handleFile = useCallback((file: File) => {
    if (!file.name.endsWith('.csv')) {
      setError('Only CSV files are supported.')
      return
    }
    setError(null)
    uploadMutation.mutate(file)
  }, [uploadMutation])

  return (
    <div className="page-container">
      <PageHeader
        title="Upload Data"
        subtitle="Import energy consumption time-series data from CSV files"
        badge="CSV"
        actions={
          <div className="glass-card px-3 py-1.5 flex items-center gap-2 text-xs text-text-muted">
            <Layers size={12} />
            Supported: utc_timestamp + DE_load_actual_entsoe_transparency
          </div>
        }
      />

      {/* Upload KPIs */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[
          { label: 'Max File Size', value: '50 MB', icon: <Upload size={16} />, color: 'text-electric-500' },
          { label: 'Format', value: 'CSV', icon: <FileText size={16} />, color: 'text-cyan-500' },
          { label: 'Uploads This Session', value: uploadHistory.length.toString(), icon: <Clock size={16} />, color: 'text-success-500' },
          { label: 'Total Valid Rows', value: uploadHistory.reduce((s, u) => s + u.rows_valid, 0).toLocaleString(), icon: <CheckCircle size={16} />, color: 'text-warning-500' },
        ].map(kpi => (
          <div key={kpi.label} className="glass-card p-4">
            <div className="flex items-center gap-2 mb-1">
              <span className={kpi.color}>{kpi.icon}</span>
              <p className="label text-xs">{kpi.label}</p>
            </div>
            <p className={`text-xl font-display font-bold ${kpi.color}`}>{kpi.value}</p>
          </div>
        ))}
      </div>

      {/* Drop Zone */}
      <DropZone onFile={handleFile} uploading={uploadMutation.isPending} />

      {/* Error */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 bg-danger-500/10 border border-danger-500/30 rounded-xl p-4 text-sm text-danger-400"
        >
          <XCircle size={16} />
          {error}
        </motion.div>
      )}

      {/* Last upload result */}
      {lastResult && <UploadResult result={lastResult} />}

      {/* Upload history */}
      {uploadHistory.length > 0 && (
        <ChartCard
          title="Upload History"
          subtitle="This session only (not persisted on refresh)"
          actions={
            <button
              onClick={() => setUploadHistory([])}
              className="flex items-center gap-1.5 text-xs text-text-muted hover:text-danger-500 transition-colors"
            >
              <Trash2 size={12} /> Clear
            </button>
          }
        >
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-bg-border">
                  {['Filename', 'Valid Rows', 'Rejected', 'Date Range', 'Uploaded At'].map(h => (
                    <th key={h} className="label pb-3 pr-4 text-left text-xs">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-bg-border/50">
                {uploadHistory.map(u => (
                  <tr key={u.id} className="hover:bg-bg-hover/20 transition-colors">
                    <td className="py-2.5 pr-4">
                      <div className="flex items-center gap-2">
                        <FileText size={14} className="text-text-muted" />
                        <span className="text-text-primary font-medium text-xs">{u.filename}</span>
                      </div>
                    </td>
                    <td className="py-2.5 pr-4 text-success-500 font-medium text-xs">
                      {u.rows_valid.toLocaleString()}
                    </td>
                    <td className={`py-2.5 pr-4 text-xs font-medium ${u.rows_rejected > 0 ? 'text-warning-500' : 'text-text-muted'}`}>
                      {u.rows_rejected.toLocaleString()}
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-text-secondary">
                      {u.time_range.start?.split('T')[0]} → {u.time_range.end?.split('T')[0]}
                    </td>
                    <td className="py-2.5 pr-4 text-xs text-text-muted">
                      {formatDateTime(u.timestamp)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>
      )}

      {/* CSV Format guide */}
      <ChartCard title="CSV Format Guide" subtitle="Required schema for energy data imports">
        <div className="space-y-4">
          <div className="bg-bg-primary rounded-xl p-4 font-mono text-xs text-text-secondary overflow-x-auto">
            <p className="text-text-muted mb-2"># Example CSV structure:</p>
            <p className="text-success-500">utc_timestamp,DE_load_actual_entsoe_transparency</p>
            <p>2023-01-01 00:00:00+00:00,42543.75</p>
            <p>2023-01-01 01:00:00+00:00,41234.50</p>
            <p>2023-01-01 02:00:00+00:00,39876.25</p>
            <p className="text-text-muted">...</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <p className="text-xs font-semibold text-text-primary mb-2">Required Columns</p>
              <div className="space-y-1.5">
                {[
                  { name: 'utc_timestamp', desc: 'ISO 8601 datetime with UTC timezone' },
                  { name: 'DE_load_actual_entsoe_transparency', desc: 'Actual load in MW (numeric)' },
                ].map(col => (
                  <div key={col.name} className="flex gap-2">
                    <code className="text-xs font-mono text-electric-400 bg-bg-primary px-1.5 py-0.5 rounded whitespace-nowrap">
                      {col.name}
                    </code>
                    <span className="text-xs text-text-muted">{col.desc}</span>
                  </div>
                ))}
              </div>
            </div>
            <div>
              <p className="text-xs font-semibold text-text-primary mb-2">Accepted Timestamp Columns</p>
              <div className="flex flex-wrap gap-1.5">
                {['utc_timestamp', 'timestamp', 'datetime', 'date', 'time'].map(c => (
                  <code key={c} className="text-xs font-mono text-text-secondary bg-bg-primary border border-bg-border px-1.5 py-0.5 rounded">
                    {c}
                  </code>
                ))}
              </div>
            </div>
          </div>
        </div>
      </ChartCard>
    </div>
  )
}

export default UploadPage
