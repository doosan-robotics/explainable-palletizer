import { useSimStore } from '../store'
import { useSimStream } from '../hooks/useSimStream'

const VIEW_HINT: Record<string, string> = {
  persp: 'Perspective viewport camera',
  top: 'Top-down orthographic view',
  front: 'Front orthographic view',
  viewport: 'Isaac Sim livestream viewport',
}

export function SimViewer() {
  const activeView = useSimStore((s) => s.activeView)
  const status = useSimStore((s) => s.status)
  const blobUrl = useSimStream(activeView)

  const isConnecting = status === 'running' && !blobUrl

  return (
    <div
      style={{
        width: '100%',
        aspectRatio: '16 / 9',
        background: '#0a0a0c',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}
    >
      {blobUrl ? (
        <img
          src={blobUrl}
          alt={VIEW_HINT[activeView] ?? 'Sim stream'}
          style={{ width: '100%', height: '100%', display: 'block' }}
        />
      ) : (
        <div className="flex flex-col items-center gap-4 select-none">
          <div
            className="w-12 h-12 rounded-2xl flex items-center justify-center"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.07)',
            }}
          >
            {isConnecting ? (
              <div
                className="w-5 h-5 rounded-full border-2 animate-spin"
                style={{
                  borderColor: 'rgba(255,255,255,0.06)',
                  borderTopColor: 'rgba(235,235,245,0.3)',
                }}
              />
            ) : (
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="rgba(235,235,245,0.18)"
                strokeWidth={1.5}
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M15 10l4.553-2.277A1 1 0 0121 8.649v6.702a1 1 0 01-1.447.894L15 14" />
                <rect x="3" y="8" width="12" height="8" rx="2" />
              </svg>
            )}
          </div>
          <div className="flex flex-col items-center gap-1">
            <p className="text-xs font-medium" style={{ color: 'rgba(235,235,245,0.22)' }}>
              {isConnecting ? 'Connecting to stream' : 'No signal'}
            </p>
            {!isConnecting && (
              <p
                className="text-xs text-center"
                style={{ color: 'rgba(235,235,245,0.13)', maxWidth: 180 }}
              >
                Press Play to start the simulation
              </p>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
