import { useCallback, useEffect, useMemo, useState } from 'react'
import URLScanner from './components/URLScanner'
import {
  ShieldCheck,
  LayoutDashboard,
  ScanLine,
  History,
  Activity,
  Settings,
  Bell,
  Search,
  ShieldAlert,
  Clock3,
  ArrowUpRight,
  Globe,
  Zap,
  RefreshCw,
  X,
} from 'lucide-react'
import { getMonitoringStats } from './services/api'

function Sidebar() {
  return (
    <aside className="flex w-64 shrink-0 flex-col border-r border-slate-800 bg-[#0c1422] px-5 py-6 max-md:hidden">
      <a href="#dashboard" className="mb-12 flex items-center gap-3">
        <div className="rounded-xl bg-emerald-500/15 p-2.5">
          <ShieldCheck className="h-7 w-7 text-emerald-400" />
        </div>
        <div>
          <span className="block text-lg font-bold text-white">PhishGuard</span>
          <span className="block text-xs text-slate-500">URL Intelligence</span>
        </div>
      </a>

      <p className="mb-4 px-3 text-xs font-semibold uppercase tracking-widest text-slate-500">
        Workspace
      </p>

      <nav aria-label="Workspace" className="space-y-2">
        <a href="#dashboard" className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-slate-400 transition hover:bg-slate-800 hover:text-white">
          <LayoutDashboard size={19} />
          Dashboard
        </a>
        <a href="#url-scanner" className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-slate-400 transition hover:bg-slate-800 hover:text-white">
          <ScanLine size={19} />
          URL Scanner
        </a>
        <a href="#scan-history" className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-slate-400 transition hover:bg-slate-800 hover:text-white">
          <History size={19} />
          Scan History
        </a>
        <a href="#monitoring" className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-slate-400 transition hover:bg-slate-800 hover:text-white">
          <Activity size={19} />
          Monitoring
        </a>
      </nav>

      <div className="mt-auto border-t border-slate-800 pt-5">
        <a href="#scanner-settings" className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm text-slate-400 hover:bg-slate-800 hover:text-white">
          <Settings size={19} />
          Scanner Settings
        </a>
        <div className="mt-5 rounded-xl border border-emerald-500/20 bg-emerald-500/5 p-4">
          <div className="mb-2 flex items-center gap-2 text-sm font-medium text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            API Monitoring
          </div>
          <p className="text-xs text-slate-400">
            Status and scan metrics are available in Monitoring.
          </p>
        </div>
      </div>
    </aside>
  )
}

function StatCard({ title, value, detail, icon: Icon, color }) {
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm text-slate-400">{title}</p>
        <div className={`rounded-xl p-2 ${color}`}>
          <Icon size={20} />
        </div>
      </div>
      <h3 className="mt-5 text-3xl font-bold text-white">{value}</h3>
      <p className="mt-2 text-xs text-slate-500">{detail}</p>
    </div>
  )
}

function App() {
  const [scans, setScans] = useState([])
  const [monitoring, setMonitoring] = useState(null)
  const [monitoringError, setMonitoringError] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [notificationsOpen, setNotificationsOpen] = useState(false)

  const refreshMonitoring = useCallback(async () => {
    try {
      const stats = await getMonitoringStats()
      setMonitoring(stats)
      setMonitoringError('')
    } catch (error) {
      const detail = error.response?.data?.detail
      setMonitoringError(
        typeof detail === 'string'
          ? detail
          : error.message || 'Unable to load monitoring data.'
      )
    }
  }, [])

  useEffect(() => {
    let active = true
    getMonitoringStats()
      .then((stats) => {
        if (active) {
          setMonitoring(stats)
        }
      })
      .catch((error) => {
        if (active) {
          const detail = error.response?.data?.detail
          setMonitoringError(
            typeof detail === 'string'
              ? detail
              : error.message || 'Unable to load monitoring data.'
          )
        }
      })

    return () => {
      active = false
    }
  }, [])

  const addScan = useCallback((result) => {
    setScans((current) => [
      {
        url: result.url,
        verdict: result.verdict,
        confidence: typeof result.confidence === 'number'
          ? `${(result.confidence * 100).toFixed(1)}%`
          : 'N/A',
        time: new Date().toLocaleTimeString(),
      },
      ...current,
    ])
    refreshMonitoring()
  }, [refreshMonitoring])

  const filteredScans = useMemo(() => {
    const query = searchQuery.trim().toLowerCase()
    if (!query) return scans
    return scans.filter((scan) =>
      `${scan.url} ${scan.verdict}`.toLowerCase().includes(query)
    )
  }, [scans, searchQuery])

  const phishingAlerts = scans.filter(
    (scan) => scan.verdict?.toLowerCase() === 'phishing'
  )
  const distribution = monitoring?.verdict_distribution || {}

  return (
    <div className="flex min-h-screen bg-[#080d18] text-slate-200">
      <Sidebar />

      <main className="min-w-0 flex-1">
        <header className="relative flex items-center justify-between border-b border-slate-800 px-5 py-5 sm:px-8">
          <div>
            <p className="text-sm text-slate-500">Security Workspace</p>
            <h2 className="mt-1 text-xl font-semibold text-white">Dashboard</h2>
          </div>

          <div className="flex items-center gap-3">
            <button
              type="button"
              aria-label={searchOpen ? 'Close search' : 'Search scan history'}
              aria-expanded={searchOpen}
              onClick={() => {
                setSearchOpen((open) => !open)
                setNotificationsOpen(false)
                if (!searchOpen) {
                  document.getElementById('scan-history')?.scrollIntoView({ behavior: 'smooth' })
                } else {
                  setSearchQuery('')
                }
              }}
              className="rounded-xl border border-slate-800 p-2.5 text-slate-400 hover:text-white"
            >
              {searchOpen ? <X size={19} /> : <Search size={19} />}
            </button>
            <button
              type="button"
              aria-label="Notifications"
              aria-expanded={notificationsOpen}
              onClick={() => {
                setNotificationsOpen((open) => !open)
                setSearchOpen(false)
              }}
              className="relative rounded-xl border border-slate-800 p-2.5 text-slate-400 hover:text-white"
            >
              <Bell size={19} />
              {phishingAlerts.length > 0 && (
                <span className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-red-500 px-1 text-[10px] text-white">
                  {phishingAlerts.length}
                </span>
              )}
            </button>
            <div className="hidden h-9 w-9 items-center justify-center rounded-full bg-emerald-500/20 text-sm font-bold text-emerald-400 sm:flex">
              A
            </div>

            {notificationsOpen && (
              <div className="absolute right-5 top-16 z-10 w-80 rounded-xl border border-slate-700 bg-[#101b2c] p-4 shadow-xl sm:right-8">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-white">Notifications</h3>
                  <button
                    type="button"
                    aria-label="Close notifications"
                    onClick={() => setNotificationsOpen(false)}
                    className="text-slate-400 hover:text-white"
                  >
                    <X size={18} />
                  </button>
                </div>
                {phishingAlerts.length ? (
                  <ul className="mt-3 space-y-3">
                    {phishingAlerts.slice(0, 5).map((scan) => (
                      <li key={`${scan.url}-${scan.time}`} className="flex gap-3 text-sm">
                        <ShieldAlert className="mt-0.5 shrink-0 text-red-400" size={18} />
                        <span className="break-all text-slate-300">
                          Phishing detected: {scan.url}
                        </span>
                      </li>
                    ))}
                  </ul>
                ) : (
                  <p className="mt-3 text-sm text-slate-400">
                    No phishing alerts in this session.
                  </p>
                )}
              </div>
            )}
          </div>
        </header>

        <div className="mx-auto max-w-7xl space-y-8 p-5 sm:p-8">
          <section id="dashboard" className="scroll-mt-6">
            <p className="text-sm font-medium text-emerald-400">
              THREAT INTELLIGENCE PLATFORM
            </p>
            <h1 className="mt-2 text-3xl font-bold text-white sm:text-4xl">
              Welcome to PhishGuard
            </h1>
            <p className="mt-2 text-sm text-slate-400">
              Analyze URLs and identify potential phishing threats using machine learning and AI.
            </p>
          </section>

          <section id="url-scanner" className="scroll-mt-6">
            <URLScanner onScanComplete={addScan} />
          </section>

          <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4" aria-label="Scan totals">
            <StatCard
              title="Total Scans"
              value={monitoring ? monitoring.total_analyses.toLocaleString() : '—'}
              detail="All-time API analyses"
              icon={Globe}
              color="bg-blue-500/10 text-blue-400"
            />
            <StatCard
              title="Safe URLs"
              value={(distribution.safe || 0).toLocaleString()}
              detail="Recorded safe verdicts"
              icon={ShieldCheck}
              color="bg-emerald-500/10 text-emerald-400"
            />
            <StatCard
              title="Threats Detected"
              value={((distribution.phishing || 0) + (distribution.suspicious || 0)).toLocaleString()}
              detail="Phishing and suspicious verdicts"
              icon={ShieldAlert}
              color="bg-red-500/10 text-red-400"
            />
            <StatCard
              title="API Status"
              value={monitoringError ? 'Unavailable' : monitoring ? 'Online' : 'Loading'}
              detail={monitoringError ? 'Monitoring endpoint unavailable' : 'Monitoring endpoint'}
              icon={Zap}
              color="bg-amber-500/10 text-amber-400"
            />
          </section>

          <section id="scan-history" className="scroll-mt-6 overflow-hidden rounded-2xl border border-slate-800 bg-[#101b2c]">
            <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 p-5">
              <div>
                <h2 className="font-semibold text-white">Recent Scans</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Scans completed in this browser session
                </p>
              </div>
              <a href="#scan-history" className="flex items-center gap-1 text-sm text-emerald-400 hover:text-emerald-300">
                View all <ArrowUpRight size={16} />
              </a>
            </div>

            {searchOpen && (
              <div className="border-b border-slate-800 p-4">
                <label htmlFor="scan-search" className="sr-only">Search scanned URLs or verdicts</label>
                <div className="flex items-center gap-3 rounded-xl border border-slate-700 bg-[#080d18] px-3">
                  <Search className="text-slate-500" size={18} />
                  <input
                    id="scan-search"
                    type="search"
                    value={searchQuery}
                    onChange={(event) => setSearchQuery(event.target.value)}
                    placeholder="Filter scans by URL or verdict"
                    className="w-full bg-transparent py-3 text-sm text-white outline-none placeholder:text-slate-500"
                  />
                </div>
              </div>
            )}

            <div className="overflow-x-auto">
              <table className="w-full min-w-[600px] text-left text-sm">
                <thead className="bg-[#0c1422] text-xs uppercase text-slate-500">
                  <tr>
                    <th className="px-5 py-4 font-medium">URL</th>
                    <th className="px-5 py-4 font-medium">Verdict</th>
                    <th className="px-5 py-4 font-medium">Confidence</th>
                    <th className="px-5 py-4 font-medium">Time</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredScans.map((scan, index) => (
                    <tr key={`${scan.url}-${scan.time}-${index}`} className="border-t border-slate-800/80">
                      <td className="break-all px-5 py-4 text-slate-300">{scan.url}</td>
                      <td className="px-5 py-4">
                        <span className={`rounded-full px-3 py-1 text-xs font-medium ${
                          ['safe', 'legitimate'].includes(scan.verdict?.toLowerCase())
                            ? 'bg-emerald-500/10 text-emerald-400'
                            : 'bg-red-500/10 text-red-400'
                        }`}>
                          {scan.verdict}
                        </span>
                      </td>
                      <td className="px-5 py-4 text-slate-300">{scan.confidence}</td>
                      <td className="px-5 py-4 text-slate-500">
                        <span className="flex items-center gap-2">
                          <Clock3 size={14} />
                          {scan.time}
                        </span>
                      </td>
                    </tr>
                  ))}
                  {filteredScans.length === 0 && (
                    <tr>
                      <td colSpan="4" className="px-5 py-8 text-center text-sm text-slate-500">
                        {scans.length
                          ? 'No scans match your search.'
                          : 'No scans in this session yet. Analyze a URL to see it here.'}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </section>

          <section id="monitoring" className="scroll-mt-6 rounded-2xl border border-slate-800 bg-[#101b2c] p-5">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <h2 className="flex items-center gap-2 font-semibold text-white">
                  <Activity size={19} className="text-emerald-400" />
                  Monitoring
                </h2>
                <p className="mt-1 text-sm text-slate-400">
                  Live summary from the analysis API.
                </p>
              </div>
              <button
                type="button"
                onClick={refreshMonitoring}
                className="flex items-center gap-2 rounded-lg border border-slate-700 px-3 py-2 text-sm text-slate-300 hover:bg-slate-800"
              >
                <RefreshCw size={16} />
                Refresh
              </button>
            </div>
            {monitoringError ? (
              <p role="alert" className="mt-4 text-sm text-red-300">{monitoringError}</p>
            ) : (
              <div className="mt-5 grid gap-4 sm:grid-cols-3">
                <p className="text-sm text-slate-400">
                  Average latency: <span className="text-white">{monitoring ? `${monitoring.average_latency_ms} ms` : 'Loading'}</span>
                </p>
                <p className="text-sm text-slate-400">
                  Browser failure rate: <span className="text-white">{monitoring ? `${monitoring.browser_failure_rate}%` : 'Loading'}</span>
                </p>
                <p className="text-sm text-slate-400">
                  API uptime: <span className="text-white">{monitoring ? `${Math.floor(monitoring.uptime_seconds / 60)} min` : 'Loading'}</span>
                </p>
              </div>
            )}
          </section>

          <footer className="pb-4 text-center text-xs text-slate-600">
            PhishGuard · AI-powered phishing URL detection
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
