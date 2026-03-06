import { useEffect, useRef, useState } from 'react'

const RECONNECT_DELAY_MS = 2000

interface Frame {
  view: string
  url: string
}

export function useSimStream(view: string): string | null {
  const [frame, setFrame] = useState<Frame | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    let cancelled = false

    function connect() {
      if (cancelled) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/sim/stream/${view}?fps=10`)
      wsRef.current = ws
      ws.binaryType = 'blob'

      ws.onmessage = (event: MessageEvent<Blob>) => {
        if (cancelled) return
        const url = URL.createObjectURL(event.data)
        setFrame((prev) => {
          if (prev) {
            const old = prev.url
            // Delay revoke until the browser has rendered the new frame
            setTimeout(() => URL.revokeObjectURL(old), 500)
          }
          return { view, url }
        })
      }

      ws.onclose = () => {
        if (cancelled) return
        setFrame(null)
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS)
      }

      ws.onerror = () => ws.close()
    }

    connect()

    return () => {
      cancelled = true
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [view])

  // view가 바뀌면 effect 실행 전이라도 렌더 시점에 즉시 null 반환
  return frame?.view === view ? frame.url : null
}
