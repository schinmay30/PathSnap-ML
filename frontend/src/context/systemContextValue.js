import { createContext, useContext } from 'react'

export const SystemContext = createContext(null)

export function useSystem() {
  const ctx = useContext(SystemContext)
  if (!ctx) {
    throw new Error('useSystem must be used inside SystemProvider')
  }
  return ctx
}
