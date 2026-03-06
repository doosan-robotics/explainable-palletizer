import { useEffect } from 'react'
import { Header } from './components/Header'
import { ViewTabs } from './components/ViewTabs'
import { SimViewer } from './components/SimViewer'
import { BoxImagesPanel } from './components/BoxImagesPanel'
import { ActionPanel } from './components/ActionPanel'
import { ReasoningPanel } from './components/ReasoningPanel'
import { ControlPanel } from './components/ControlPanel'
import { useSimEvents } from './hooks/useSimEvents'
import { useSimStore } from './store'

export default function App() {
  useSimEvents()

  const setStatus = useSimStore((s) => s.setStatus)
  const setModelName = useSimStore((s) => s.setModelName)

  useEffect(() => {
    fetch('/api/status')
      .then((r) => r.json())
      .then((data: { state?: string; status?: string }) => {
        if (data.state === 'running' || data.state === 'stopped') {
          setStatus(data.state)
        }
      })
      .catch(() => {})
  }, [setStatus, setModelName])

  return (
    <div className="h-screen flex flex-col" style={{ background: '#111113' }}>
      <Header />
      <main
        className="flex-1 min-h-0"
        style={{ display: 'grid', gridTemplateColumns: '1fr 640px' }}
      >
        <div
          className="flex flex-col min-h-0"
          style={{ borderRight: '1px solid rgba(255,255,255,0.06)' }}
        >
          <ViewTabs />
          <SimViewer />
        </div>
        <div className="flex flex-col min-h-0" style={{ background: '#161618' }}>
          <BoxImagesPanel />
          <ActionPanel />
          <ReasoningPanel />
          <ControlPanel />
        </div>
      </main>
    </div>
  )
}
