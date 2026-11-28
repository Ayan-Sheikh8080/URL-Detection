
import axios from 'axios'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || '/api',
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
})

export const analyzeUrl = async ({
  url,
  modelConfig = 'combined',
  useLlm = true,
  browserEnabled = true,
}) => {
  const response = await api.post('/analyze', {
    url,
    model_config: modelConfig,
    use_llm: useLlm,
    browser_enabled: browserEnabled,
  })

  return response.data
}

export const getMonitoringStats = async () => {
  const response = await api.get('/monitoring/stats')
  return response.data
}

export default api
