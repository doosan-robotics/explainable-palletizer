import { useSimStore } from '../store'

const STATUS_COLOR: Record<string, string> = {
  idle: '#636366',
  running: '#30d158',
  stopped: '#ff453a',
}

export function Header() {
  const modelName = useSimStore((s) => s.modelName)
  const status = useSimStore((s) => s.status)
  const isRunning = status === 'running'

  return (
    <header
      className="flex items-center justify-between px-4 h-11 shrink-0"
      style={{
        background: 'rgba(17,17,19,0.92)',
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(255,255,255,0.07)',
      }}
    >
      <div className="flex items-center gap-3">
        <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
          <polygon
            points="8,1.2 14.2,4.6 14.2,11.4 8,14.8 1.8,11.4 1.8,4.6"
            stroke="#0a84ff"
            strokeWidth="1.4"
          />
          <circle cx="8" cy="8" r="1.5" fill="#0a84ff" fillOpacity="0.6" />
        </svg>
        <div className="flex items-center gap-2">
          <span
            className="text-sm font-semibold"
            style={{ color: 'rgba(235,235,245,0.9)', letterSpacing: '-0.01em' }}
          >
            DR AI Palletizer
          </span>
          {!isRunning && (
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: STATUS_COLOR[status] ?? '#636366' }}
            />
          )}
        </div>
        {isRunning && (
          <div
            className="flex items-center gap-1.5 px-2 py-0.5 rounded-full"
            style={{
              background: 'rgba(48,209,88,0.1)',
              border: '1px solid rgba(48,209,88,0.25)',
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: '#30d158',
                animation: 'pulse-dot 1.4s ease-in-out infinite',
              }}
            />
            <span
              className="text-xs font-semibold"
              style={{ color: '#30d158', letterSpacing: '0.02em' }}
            >
              Running
            </span>
          </div>
        )}
      </div>
      {modelName && (
        <span
          className="text-xs font-mono px-2 py-0.5 rounded-md"
          style={{
            background: 'rgba(255,255,255,0.05)',
            color: 'rgba(235,235,245,0.38)',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          {modelName}
        </span>
      )}
      <style>{`
        @keyframes pulse-dot {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
      `}</style>
    </header>
  )
}
