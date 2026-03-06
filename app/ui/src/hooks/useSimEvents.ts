import { useEffect, useRef } from 'react'
import { useSimStore } from '../store'

const RECONNECT_DELAY_MS = 2000

export function useSimEvents(): void {
  const appendAction = useSimStore((s) => s.appendAction)
  const appendReasoning = useSimStore((s) => s.appendReasoning)
  const setStatus = useSimStore((s) => s.setStatus)
  const setModelName = useSimStore((s) => s.setModelName)
  const setBoxImages = useSimStore((s) => s.setBoxImages)

  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true

    function connect() {
      if (!mountedRef.current) return

      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const ws = new WebSocket(`${protocol}//${window.location.host}/api/events`)
      wsRef.current = ws

      ws.onmessage = (event: MessageEvent<string>) => {
        if (!mountedRef.current) return
        try {
          const msg = JSON.parse(event.data) as {
            type: string
            content?: string
            state?: string
            model?: string
            images?: { data: string; label: string }[]
          }
          if (msg.type === 'action' && msg.content) {
            appendAction(msg.content)
          } else if (msg.type === 'reasoning' && msg.content) {
            appendReasoning(msg.content)
          } else if (msg.type === 'status' && msg.state) {
            // Map control-loop states to UI status values
            const s = msg.state
            if (s === 'idle') setStatus('idle')
            else if (s === 'paused') setStatus('stopped')
            else setStatus('running') // running, initializing, thinking, waiting_for_boxes
            if (msg.model !== undefined) setModelName(msg.model)
          } else if (msg.type === 'box_images' && Array.isArray(msg.images)) {
            setBoxImages(msg.images)
          }
        } catch {
          // ignore malformed messages
        }
      }

      ws.onclose = () => {
        if (!mountedRef.current) return
        reconnectTimer.current = setTimeout(connect, RECONNECT_DELAY_MS)
      }

      ws.onerror = () => {
        ws.close()
      }
    }

    connect()

    return () => {
      mountedRef.current = false
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      if (wsRef.current) wsRef.current.close()
    }
  }, [appendAction, appendReasoning, setStatus, setModelName, setBoxImages])
}
