
import { useState } from 'react'
import {
  ScanLine,
  Globe,
  ShieldCheck,
  ShieldAlert,
  LoaderCircle,
  BrainCircuit,
  Monitor,
} from 'lucide-react'
import { analyzeUrl } from '../services/api'

function formatPercent(value) {
  if (typeof value !== 'number') return 'N/A'
  return `${(value * 100).toFixed(1)}%`
}

export default function URLScanner({ onScanComplete }) {
  const [url, setUrl] = useState('')
  const [modelConfig, setModelConfig] = useState('combined')
  const [useLlm, setUseLlm] = useState(true)
  const [browserEnabled, setBrowserEnabled] = useState(true)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')

  const handleScan = async (event) => {
    event.preventDefault()

    if (!url.trim()) {
      setError('Please enter a URL.')
      return
    }

    setLoading(true)
    setError('')
    setResult(null)

    try {
      const data = await analyzeUrl({
        url: url.trim(),
        modelConfig,
        useLlm,
        browserEnabled,
      })

      setResult(data)
      onScanComplete?.(data)
    } catch (err) {
      const detail = err.response?.data?.detail
      setError(
        typeof detail === 'string'
          ? detail
          : err.message || 'Unable to analyze this URL.'
      )
    } finally {
      setLoading(false)
    }
  }

  const verdict = result?.verdict?.toLowerCase()
  const isSafe = verdict === 'safe' || verdict === 'legitimate'
  const isPhishing = verdict === 'phishing'

  const verdictColor = isSafe
    ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10'
    : isPhishing
      ? 'text-red-400 border-red-500/30 bg-red-500/10'
      : 'text-amber-400 border-amber-500/30 bg-amber-500/10'

  return (
    <section className="space-y-5">
      <div className="rounded-2xl border border-emerald-500/20 bg-gradient-to-br from-[#102b2b] to-[#101b2c] p-6 sm:p-8">
        <div className="flex items-center gap-2 text-emerald-400">
          <ScanLine size={21} />
          <span className="text-sm font-semibold">
            URL SECURITY SCANNER
          </span>
        </div>

        <h2 className="mt-4 text-2xl font-bold text-white">
          Check a website for threats
        </h2>

        <p className="mt-2 text-sm text-slate-400">
          Analyze a URL using your machine learning models, browser evidence,
          RAG pipeline and AI reasoning.
        </p>

        <form onSubmit={handleScan} className="mt-6 space-y-4">
          <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-[#080d18]/80 px-4">
            <Globe className="shrink-0 text-slate-500" size={20} />
            <input
              type="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://example.com"
              required
              className="w-full bg-transparent py-4 text-sm text-white outline-none placeholder:text-slate-600"
            />
          </div>

          <div id="scanner-settings" className="grid scroll-mt-6 gap-4 sm:grid-cols-3">
            <label className="text-xs text-slate-400">
              Model configuration
              <select
                value={modelConfig}
                onChange={(event) => setModelConfig(event.target.value)}
                className="mt-2 w-full rounded-lg border border-slate-700 bg-[#080d18] p-3 text-sm text-white outline-none"
              >
                <option value="combined">Combined</option>
                <option value="url_only">URL only</option>
                <option value="content_only">Content only</option>
              </select>
            </label>

            <label className="flex items-center gap-3 rounded-lg border border-slate-700 bg-[#080d18]/60 p-3 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={useLlm}
                onChange={(event) => setUseLlm(event.target.checked)}
                className="accent-emerald-500"
              />
              Enable Bedrock AI
            </label>

            <label className="flex items-center gap-3 rounded-lg border border-slate-700 bg-[#080d18]/60 p-3 text-sm text-slate-300">
              <input
                type="checkbox"
                checked={browserEnabled}
                onChange={(event) => setBrowserEnabled(event.target.checked)}
                className="accent-emerald-500"
              />
              Browser analysis
            </label>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="flex w-full items-center justify-center gap-2 rounded-xl bg-emerald-500 px-7 py-4 text-sm font-semibold text-[#06130e] transition hover:bg-emerald-400 disabled:cursor-not-allowed disabled:opacity-60 sm:w-auto"
          >
            {loading ? (
              <>
                <LoaderCircle className="animate-spin" size={18} />
                Analyzing URL...
              </>
            ) : (
              <>
                <ScanLine size={18} />
                Analyze URL
              </>
            )}
          </button>
        </form>

        {loading && (
          <p className="mt-4 text-sm text-emerald-400">
            Your backend is analyzing the URL. This may take a little while.
          </p>
        )}

        {error && (
          <div className="mt-5 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
            {error}
          </div>
        )}
      </div>

      {result && (
        <div className="space-y-5">
          <div className={`rounded-2xl border p-6 ${verdictColor}`}>
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                {isSafe ? (
                  <ShieldCheck size={32} />
                ) : (
                  <ShieldAlert size={32} />
                )}
                <div>
                  <p className="text-xs uppercase tracking-widest opacity-70">
                    Analysis verdict
                  </p>
                  <h2 className="mt-1 text-2xl font-bold capitalize">
                    {result.verdict}
                  </h2>
                </div>
              </div>

              <div>
                <p className="text-xs opacity-70">Confidence</p>
                <p className="text-3xl font-bold">
                  {formatPercent(result.confidence)}
                </p>
              </div>
            </div>

            <p className="mt-4 break-all text-sm opacity-80">
              {result.url}
            </p>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
            {[
              ['Model score', formatPercent(result.model_score)],
              ['Combined score', formatPercent(result.combined_score)],
              ['Model', result.model_family || 'N/A'],
              ['Analysis time', `${result.load_ms ?? 'N/A'} ms`],
            ].map(([label, value]) => (
              <div
                key={label}
                className="rounded-xl border border-slate-800 bg-[#101b2c] p-4"
              >
                <p className="text-xs text-slate-500">{label}</p>
                <p className="mt-2 break-words text-lg font-semibold text-white">
                  {value}
                </p>
              </div>
            ))}
          </div>

          <div className="grid gap-5 lg:grid-cols-2">
            <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
              <h3 className="flex items-center gap-2 font-semibold text-white">
                <ShieldAlert size={18} className="text-emerald-400" />
                Detection reasons
              </h3>
              <ul className="mt-4 space-y-3">
                {(result.reasons || []).length ? (
                  result.reasons.map((reason, index) => (
                    <li
                      key={index}
                      className="text-sm leading-relaxed text-slate-300"
                    >
                      • {typeof reason === 'string' ? reason : JSON.stringify(reason)}
                    </li>
                  ))
                ) : (
                  <li className="text-sm text-slate-500">
                    No reasons returned.
                  </li>
                )}
              </ul>
            </div>

            <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
              <h3 className="flex items-center gap-2 font-semibold text-white">
                <BrainCircuit size={18} className="text-emerald-400" />
                AI and RAG information
              </h3>
              <div className="mt-4 space-y-3 text-sm text-slate-300">
                <p>
                  <span className="text-slate-500">Reasoner:</span>{' '}
                  {result.reasoner || 'N/A'}
                </p>
                <p>
                  <span className="text-slate-500">Model configuration:</span>{' '}
                  {result.model_config || 'N/A'}
                </p>
                <p>
                  <span className="text-slate-500">Retrieved sources:</span>{' '}
                  {result.retrieved_sources?.length ?? 0}
                </p>
                <p>
                  <span className="text-slate-500">Agent rounds:</span>{' '}
                  {result.agent_rounds ?? 0}
                </p>
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
            <h3 className="flex items-center gap-2 font-semibold text-white">
              <Monitor size={18} className="text-emerald-400" />
              Browser evidence
            </h3>
            <div className="mt-4 grid gap-3 text-sm sm:grid-cols-2">
              <p className="text-slate-300">
                Browser status:{' '}
                <span className={result.browser_ok ? 'text-emerald-400' : 'text-amber-400'}>
                  {result.browser_ok ? 'Successful' : 'Unavailable'}
                </span>
              </p>
              <p className="text-slate-300">
                Page title:{' '}
                {result.evidence_summary?.title || 'Not available'}
              </p>
              <p className="break-all text-slate-300">
                Final URL:{' '}
                {result.evidence_summary?.final_url || 'Not available'}
              </p>
              <p className="text-slate-300">
                Load time: {result.load_ms ?? 'N/A'} ms
              </p>
            </div>

            {result.browser_error && (
              <p className="mt-3 text-sm text-amber-400">
                {result.browser_error}
              </p>
            )}

            {result.screenshot_url && (
              <img
                src={result.screenshot_url}
                alt="Analyzed webpage screenshot"
                className="mt-5 max-h-96 w-full rounded-xl border border-slate-700 object-contain"
              />
            )}
          </div>

          <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
            <h3 className="font-semibold text-white">RAG retrieved evidence</h3>
            <div className="mt-4 space-y-3">
              {(result.rag_retrieval || []).map((item, index) => (
                <div
                  key={index}
                  className="rounded-xl border border-slate-800 bg-[#0c1422] p-4"
                >
                  <p className="text-xs text-emerald-400">
                    {item.source} · Similarity: {item.similarity}
                  </p>
                  <p className="mt-2 text-sm text-slate-300">{item.text}</p>
                  <p className="mt-2 text-xs text-slate-500">
                    Label: {item.label}
                  </p>
                </div>
              ))}
              {!result.rag_retrieval?.length && (
                <p className="text-sm text-slate-500">
                  No RAG evidence returned.
                </p>
              )}
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
            <h3 className="font-semibold text-white">Agent execution trace</h3>
            <pre className="mt-4 overflow-x-auto whitespace-pre-wrap break-words rounded-xl bg-[#080d18] p-4 text-xs text-slate-300">
              {JSON.stringify(result.agent_trace || [], null, 2)}
            </pre>
          </div>
        </div>
      )}
    </section>
  )
}
