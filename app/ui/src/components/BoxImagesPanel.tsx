import { useSimStore } from '../store'

export function BoxImagesPanel() {
  const boxImages = useSimStore((s) => s.boxImages)

  return (
    <div
      className="shrink-0 px-3 py-3"
      style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}
    >
      <div
        className="flex items-center mb-2.5"
      >
        <span
          className="text-xs font-semibold uppercase"
          style={{ color: 'rgba(235,235,245,0.28)', letterSpacing: '0.07em' }}
        >
          Box Images
        </span>
        {boxImages.length > 0 && (
          <span
            className="ml-auto text-xs font-mono px-1.5 py-px rounded"
            style={{
              background: 'rgba(255,255,255,0.05)',
              color: 'rgba(235,235,245,0.22)',
            }}
          >
            {boxImages.length}
          </span>
        )}
      </div>
      {boxImages.length === 0 ? (
        <div
          className="flex items-center justify-center rounded"
          style={{
            height: 60,
            background: 'rgba(255,255,255,0.03)',
            border: '1px solid rgba(255,255,255,0.07)',
          }}
        >
          <span className="text-xs" style={{ color: 'rgba(235,235,245,0.18)' }}>
            Waiting for boxes...
          </span>
        </div>
      ) : (
        <div
          className="grid gap-1.5"
          style={{
            gridTemplateColumns: `repeat(${Math.min(boxImages.length, 3)}, 1fr)`,
          }}
        >
          {boxImages.map((img, i) => (
            <div key={`${img.label}-${i}`} className="flex flex-col gap-1">
              <div
                className="w-full overflow-hidden rounded"
                style={{
                  aspectRatio: '4 / 3',
                  background: 'rgba(255,255,255,0.03)',
                  border: '1px solid rgba(255,255,255,0.07)',
                }}
              >
                <img
                  src={img.data}
                  alt={img.label}
                  className="w-full h-full object-cover"
                />
              </div>
              <span
                className="text-center text-xs"
                style={{ color: 'rgba(235,235,245,0.22)', fontSize: '0.65rem' }}
              >
                {img.label}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
