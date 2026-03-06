import { useSimStore, type ViewName } from '../store'

const VIEWS: { id: ViewName; label: string }[] = [
  { id: 'persp', label: 'Perspective' },
  { id: 'top', label: 'Top' },
  { id: 'front', label: 'Front' },
]

export function ViewTabs() {
  const activeView = useSimStore((s) => s.activeView)
  const setActiveView = useSimStore((s) => s.setActiveView)

  return (
    <div
      className="shrink-0 px-3 py-2"
      style={{ background: '#1a1a1c', borderBottom: '1px solid rgba(255,255,255,0.06)' }}
    >
      <div
        className="flex p-0.5 rounded-lg"
        style={{ background: 'rgba(255,255,255,0.05)' }}
      >
        {VIEWS.map((v) => {
          const isActive = activeView === v.id
          return (
            <button
              key={v.id}
              onClick={() => setActiveView(v.id)}
              className="flex-1 py-1.5 text-xs font-medium rounded-[7px] transition-all"
              style={{
                background: isActive ? 'rgba(255,255,255,0.1)' : 'transparent',
                color: isActive ? 'rgba(235,235,245,0.92)' : 'rgba(235,235,245,0.38)',
                boxShadow: isActive
                  ? '0 1px 3px rgba(0,0,0,0.5), inset 0 0.5px 0 rgba(255,255,255,0.07)'
                  : 'none',
              }}
            >
              {v.label}
            </button>
          )
        })}
      </div>
    </div>
  )
}
