import axios from 'axios'

const STORAGE_KEY = 'pathsnap_api_base_url'
const DEFAULT_API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export function getStoredApiBaseUrl() {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY)
    return saved && saved.trim() ? saved.trim() : DEFAULT_API_BASE_URL
  } catch {
    return DEFAULT_API_BASE_URL
  }
}

export function setStoredApiBaseUrl(url) {
  const normalized = (url || '').trim().replace(/\/+$/, '') || DEFAULT_API_BASE_URL
  try {
    window.localStorage.setItem(STORAGE_KEY, normalized)
  } catch {
    // Ignore storage quota / privacy mode restrictions
  }
  apiClient.defaults.baseURL = normalized
  return normalized
}

export function resetApiBaseUrl() {
  try {
    window.localStorage.removeItem(STORAGE_KEY)
  } catch {
    // Ignore storage errors
  }
  apiClient.defaults.baseURL = DEFAULT_API_BASE_URL
  return DEFAULT_API_BASE_URL
}

const apiClient = axios.create({
  baseURL: getStoredApiBaseUrl(),
  timeout: 15000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
})

export default apiClient
