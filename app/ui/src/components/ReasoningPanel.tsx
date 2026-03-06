import { useEffect, useRef } from 'react'
import { useSimStore } from '../store'

export function ReasoningPanel() {
  const reasoning = useSimStore((s) => s.reasoning)
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [reasoning])

  return (
    <div
      className="flex flex-col min-h-0"
      style={{ flex: '2 2 0%', borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <div
        className="flex items-center px-4 py-2.5 shrink-0"
        style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}
      >
        <span
          className="text-xs font-semibold uppercase"
          style={{ color: 'rgba(235,235,245,0.28)', letterSpacing: '0.07em' }}
        >
          Reasoning
        </span>
      </div>
      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2.5">
        {reasoning.length === 0 ? (
          <p className="text-xs" style={{ color: 'rgba(235,235,245,0.18)' }}>
            No reasoning yet.
          </p>
        ) : (
          reasoning.map((entry, i) => (
            <div
              key={i}
              className="pl-2.5 text-xs leading-relaxed"
              style={{
                color: 'rgba(235,235,245,0.6)',
                borderLeft: '1.5px solid rgba(255,255,255,0.1)',
              }}
            >
              {entry}
            </div>
          ))
        )}
        <div ref={bottomRef} />
      </div>
    </div>
  )
}
