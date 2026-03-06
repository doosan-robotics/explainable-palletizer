import { useEffect, useRef } from 'react'
import { useSimStore } from '../store'

export function ActionPanel() {
  const actions = useSimStore((s) => s.actions)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [actions])

  return (
    <div
      className="flex flex-col min-h-0"
      style={{ flex: '3 3 0%', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <div
        className="flex items-center px-4 py-2.5 shrink-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
      >
        <span
          className="text-xs font-semibold uppercase"
          style={{ color: 'rgba(235,235,245,0.28)', letterSpacing: '0.07em' }}
        >
          Actions
        </span>
        {actions.length > 0 && (
          <span
            className="ml-auto text-xs font-mono px-1.5 py-px rounded"
            style={{
              background: 'rgba(255,255,255,0.05)',
              color: 'rgba(235,235,245,0.22)',
            }}
          >
            {actions.length}
          </span>
        )}
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2">
        {actions.length === 0 ? (
          <p className="text-xs" style={{ color: 'rgba(235,235,245,0.18)' }}>
            Waiting for actions...
          </p>
        ) : (
          actions.map((action, i) => (
            <div key={i} className="flex items-start gap-2">
              <span
                className="text-xs font-mono shrink-0 mt-px w-5 text-right tabular-nums"
                style={{ color: 'rgba(235,235,245,0.18)' }}
              >
                {i + 1}
              </span>
              <span
                className="text-xs font-mono leading-relaxed"
                style={{ color: 'rgba(235,235,245,0.72)' }}
              >
                {action}
              </span>
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
