import { useSimStore } from '../store'

export function ControlPanel() {
  const status = useSimStore((s) => s.status)
  const setStatus = useSimStore((s) => s.setStatus)
  const isRunning = status === 'running'

  async function handlePlay() {
    const res = await fetch('/api/control/start', { method: 'POST' })
    if (res.ok) {
      const data = (await res.json()) as { ok: boolean; state: string }
      if (data.ok) setStatus(data.state === 'running' ? 'running' : 'idle')
    }
  }

  async function handleStop() {
    const res = await fetch('/api/control/pause', { method: 'POST' })
    if (res.ok) {
      const data = (await res.json()) as { ok: boolean; state: string }
      if (data.ok) setStatus('stopped')
    }
  }

  const playLabel = status === 'stopped' ? 'Resume' : 'Play'

  return (
    <div className="shrink-0 px-3 py-3" style={{ background: '#131315' }}>
      {/* Sim controls */}
      <div className="flex gap-2">
        <button
          onClick={handlePlay}
          disabled={isRunning}
          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-full text-xs font-semibold transition-all"
          style={
            isRunning
              ? { background: 'rgba(255,255,255,0.04)', color: 'rgba(235,235,245,0.18)', cursor: 'not-allowed' }
              : { background: '#0a84ff', color: '#fff', boxShadow: '0 2px 8px rgba(10,132,255,0.3)' }
          }
        >
          <svg width="9" height="10" viewBox="0 0 9 10" fill="currentColor">
            <polygon points="0.5,0.5 8.5,5 0.5,9.5" />
          </svg>
          {playLabel}
        </button>
        <button
          onClick={handleStop}
          disabled={!isRunning}
          className="flex-1 flex items-center justify-center gap-1.5 py-2 rounded-full text-xs font-semibold transition-all"
          style={
            !isRunning
              ? { background: 'rgba(255,255,255,0.04)', color: 'rgba(235,235,245,0.18)', cursor: 'not-allowed' }
              : { background: 'rgba(255,69,58,0.12)', color: '#ff453a', border: '1px solid rgba(255,69,58,0.22)' }
          }
        >
          <svg width="8" height="8" viewBox="0 0 8 8" fill="currentColor">
            <rect width="8" height="8" rx="1.5" />
          </svg>
          Stop
        </button>
      </div>

    </div>
  )
}
