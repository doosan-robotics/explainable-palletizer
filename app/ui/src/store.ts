import { create } from 'zustand'

type Status = 'idle' | 'running' | 'stopped'
export type ViewName = 'persp' | 'top' | 'front'

export interface BoxImage {
  data: string
  label: string
}

interface SimStore {
  status: Status
  activeView: ViewName
  modelName: string
  actions: string[]
  reasoning: string[]
  boxImages: BoxImage[]
  setStatus: (s: Status) => void
  setActiveView: (v: ViewName) => void
  setModelName: (name: string) => void
  appendAction: (s: string) => void
  appendReasoning: (s: string) => void
  setBoxImages: (images: BoxImage[]) => void
}

export const useSimStore = create<SimStore>((set) => ({
  status: 'idle',
  activeView: 'persp',
  modelName: '',
  actions: [],
  reasoning: [],
  boxImages: [],
  setStatus: (status) => set({ status }),
  setActiveView: (activeView) => set({ activeView }),
  setModelName: (modelName) => set({ modelName }),
  appendAction: (s) => set((state) => ({ actions: [...state.actions, s] })),
  appendReasoning: (s) => set((state) => ({ reasoning: [...state.reasoning, s] })),
  setBoxImages: (boxImages) => set({ boxImages }),
}))
