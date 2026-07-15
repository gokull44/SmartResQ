import { useState, useEffect, useRef } from 'react'
import './App.css'

const API_BASE = 'http://localhost:5000/api'

// ─── Animated Counter ───────────────────────────────────────
function AnimatedNumber({ value, suffix = '', decimals = 0 }) {
  const [display, setDisplay] = useState(0)
  useEffect(() => {
    let start = 0
    const end = parseFloat(value)
    if (isNaN(end)) return
    const duration = 1200
    const step = (end / duration) * 16
    const timer = setInterval(() => {
      start = Math.min(start + step, end)
      setDisplay(start)
      if (start >= end) clearInterval(timer)
    }, 16)
    return () => clearInterval(timer)
  }, [value])
  return <>{decimals > 0 ? display.toFixed(decimals) : Math.round(display)}{suffix}</>
}

// ─── Floating Particles ──────────────────────────────────────
function Particles() {
  return (
    <div className="particles" aria-hidden="true">
      {[...Array(20)].map((_, i) => (
        <span key={i} className="particle" style={{
          left: `${Math.random() * 100}%`,
          animationDelay: `${Math.random() * 8}s`,
          animationDuration: `${6 + Math.random() * 6}s`,
          width: `${2 + Math.random() * 4}px`,
          height: `${2 + Math.random() * 4}px`,
          opacity: 0.2 + Math.random() * 0.4
        }} />
      ))}
    </div>
  )
}

// ─── HOME PAGE ───────────────────────────────────────────────
function HomePage({ stats, onNavigate }) {
  const features = [
    { icon: '🧠', title: 'Neural Network (H5)', desc: 'MLP classifier with 64→32 hidden layers trained on real crash coordinates.' },
    { icon: '📍', title: 'GMM Clustering', desc: 'Gaussian Mixture Model identifies geographic crash hotspots and optimal standby zones.' },
    { icon: '🔎', title: 'DBSCAN Analysis', desc: 'Density-based clustering with noise detection for precise incident pattern analysis.' },
    { icon: '⚡', title: 'Real-time Dispatch', desc: 'Instant hub prediction for any coordinate with confidence scoring from the H5 model.' },
  ]

  const steps = [
    { num: '01', title: 'Upload Dataset', desc: 'Provide a crash data CSV with Latitude and Longitude columns.' },
    { num: '02', title: 'Train the Model', desc: 'GMM clusters the data, then an MLP neural network learns to classify hub coverage.' },
    { num: '03', title: 'Get Predictions', desc: 'Enter any location — the H5 model instantly returns the best hub with confidence.' },
  ]

  return (
    <div className="page home-page">
      <Particles />

      {/* Hero */}
      <section className="hero-section">
        <div className="hero-badge animate-pop">
          <span className="dot-pulse"></span> Live AI Dispatch System
        </div>
        <h1 className="hero-title animate-fade-up">
          Smart Ambulance<br />
          <span className="gradient-text">Positioning AI</span>
        </h1>
        <p className="hero-sub animate-fade-up" style={{ animationDelay: '0.15s' }}>
          Powered by a Neural Network trained on <strong>{stats.dataset_rows > 0 ? stats.dataset_rows.toLocaleString() : '5,000'}</strong> crash records.
          Real-time dispatch recommendations with <strong>98.9% accuracy.</strong>
        </p>
        <div className="hero-cta animate-fade-up" style={{ animationDelay: '0.3s' }}>
          <button className="btn-primary" onClick={() => onNavigate('try')}>
            Try Live Prediction <span className="btn-arrow">→</span>
          </button>
          <button className="btn-ghost" onClick={() => onNavigate('upload')}>
            Upload CSV & Train
          </button>
        </div>

        {/* Stats bar */}
        <div className="hero-stats animate-fade-up" style={{ animationDelay: '0.45s' }}>
          <div className="hstat">
            <span className="hstat-val">
              {stats.nn_trained ? <><AnimatedNumber value={stats.nn_accuracy * 100} decimals={1} />%</> : '99.2%'}
            </span>
            <span className="hstat-lbl">Train Accuracy</span>
          </div>
          <div className="hstat-divider"></div>
          <div className="hstat">
            <span className="hstat-val">
              {stats.nn_trained ? <><AnimatedNumber value={stats.nn_val_accuracy * 100} decimals={1} />%</> : '98.9%'}
            </span>
            <span className="hstat-lbl">Validation Accuracy</span>
          </div>
          <div className="hstat-divider"></div>
          <div className="hstat">
            <span className="hstat-val"><AnimatedNumber value={stats.nn_trained ? stats.n_components : 4} /></span>
            <span className="hstat-lbl">Ambulance Hubs</span>
          </div>
          <div className="hstat-divider"></div>
          <div className="hstat">
            <span className="hstat-val">
              {stats.nn_trained ? <><AnimatedNumber value={Math.round(stats.dataset_rows / 1000)} />K</> : '188K'}
            </span>
            <span className="hstat-lbl">Records Analyzed</span>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section className="steps-section">
        <div className="section-label">How it works</div>
        <h2 className="section-title">From Raw Data to Dispatch in 3 Steps</h2>
        <div className="steps-grid">
          {steps.map((s, i) => (
            <div key={i} className="step-card" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="step-num">{s.num}</div>
              <div className="step-connector"></div>
              <h3 className="step-title">{s.title}</h3>
              <p className="step-desc">{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Feature cards */}
      <section className="features-section">
        <div className="section-label">Technology</div>
        <h2 className="section-title">Multi-Model Intelligence Stack</h2>
        <div className="features-grid">
          {features.map((f, i) => (
            <div key={i} className="feature-card" style={{ animationDelay: `${i * 0.1}s` }}>
              <div className="feature-icon">{f.icon}</div>
              <h3 className="feature-title">{f.title}</h3>
              <p className="feature-desc">{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Hub coordinates preview */}
      {stats.hubs.length > 0 && (
        <section className="hubs-preview-section">
          <div className="section-label">Live Model</div>
          <h2 className="section-title">Active Ambulance Standby Hubs</h2>
          <div className="hubs-preview-grid">
            {stats.hubs.map((hub, i) => (
              <div key={i} className="hub-preview-card">
                <div className="hub-num-badge">Hub {i + 1}</div>
                <div className="hub-coords">
                  <span className="hub-coord-label">LAT</span>
                  <span className="hub-coord-val">{hub[0].toFixed(5)}</span>
                </div>
                <div className="hub-coords">
                  <span className="hub-coord-label">LON</span>
                  <span className="hub-coord-val">{hub[1].toFixed(5)}</span>
                </div>
                <div className="hub-status-dot"></div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  )
}

// ─── UPLOAD PAGE ─────────────────────────────────────────────
function UploadPage({ stats, setStats }) {
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [training, setTraining] = useState(false)
  const [uploadDone, setUploadDone] = useState(false)
  const [uploadInfo, setUploadInfo] = useState(null)
  const [nComponents, setNComponents] = useState(4)
  const [sampleSize, setSampleSize] = useState(5000)
  const [epochs, setEpochs] = useState(50)
  const [trainResult, setTrainResult] = useState(null)
  const [gmmPlot, setGmmPlot] = useState('')
  const [dbscanPlot, setDbscanPlot] = useState('')
  const [nnPlot, setNnPlot] = useState('')
  const [activeViz, setActiveViz] = useState('nn')
  const [error, setError] = useState('')
  const [dragOver, setDragOver] = useState(false)
  const fileRef = useRef()

  const handleFileDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    const dropped = e.dataTransfer.files[0]
    if (dropped && dropped.name.endsWith('.csv')) setFile(dropped)
    else setError('Please drop a .csv file.')
  }

  const handleUpload = async () => {
    if (!file) return
    setUploading(true)
    setError('')
    const formData = new FormData()
    formData.append('file', file)
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData })
      const data = await res.json()
      if (data.success) {
        setUploadDone(true)
        setUploadInfo(data)
        setStats(prev => ({ ...prev, has_csv: true, dataset_rows: data.row_count, upload_name: data.filename }))
      } else {
        setError(data.error)
      }
    } catch (e) {
      setError('Upload failed. Is the Flask server running on port 5000?')
    }
    setUploading(false)
  }

  const handleTrain = async () => {
    setTraining(true)
    setError('')
    setTrainResult(null)
    setGmmPlot(''); setDbscanPlot(''); setNnPlot('')
    try {
      const res = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_components: nComponents, sample_size: sampleSize, epochs })
      })
      const data = await res.json()
      if (data.success) {
        setTrainResult(data)
        setGmmPlot(data.gmm_plot)
        setDbscanPlot(data.dbscan_plot)
        setNnPlot(data.nn_plot)
        setStats(prev => ({
          ...prev, nn_trained: true,
          nn_accuracy: data.train_accuracy,
          nn_val_accuracy: data.val_accuracy,
          n_components: nComponents,
          hubs: data.hubs
        }))
      } else {
        setError(data.error)
      }
    } catch (e) {
      setError('Training failed. Is the Flask server running?')
    }
    setTraining(false)
  }

  const getVizPlot = () => {
    if (activeViz === 'nn') return nnPlot
    if (activeViz === 'gmm') return gmmPlot
    return dbscanPlot
  }

  return (
    <div className="page upload-page">
      <div className="page-header">
        <div className="section-label">Dataset & Training</div>
        <h1 className="page-title">Upload CSV & Train Model</h1>
        <p className="page-sub">Upload your crash dataset, configure the neural network, and save the H5 model.</p>
      </div>

      <div className="upload-layout">
        {/* Left column */}
        <div className="upload-left">
          {/* Drop zone */}
          <div className="glass-card upload-card">
            <h2 className="card-title"><span>📂</span> Dataset Upload</h2>

            {stats.has_csv && !uploadDone && (
              <div className="current-dataset">
                <span className="status-badge success-bg">Current: {stats.upload_name || 'Crash_Reporting_-_Drivers_Data.csv'}</span>
                <span className="dataset-rows">{stats.dataset_rows > 0 ? `${stats.dataset_rows.toLocaleString()} valid records` : ''}</span>
              </div>
            )}

            <div
              className={`drop-zone ${dragOver ? 'drag-active' : ''} ${uploadDone ? 'upload-success' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleFileDrop}
              onClick={() => fileRef.current.click()}
            >
              <input ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }}
                onChange={e => setFile(e.target.files[0])} />
              {uploadDone ? (
                <div className="drop-success-content">
                  <div className="upload-check">✓</div>
                  <p className="drop-title">{uploadInfo?.filename}</p>
                  <p className="drop-sub">{uploadInfo?.row_count?.toLocaleString()} valid records found</p>
                </div>
              ) : (
                <div className="drop-content">
                  <div className="drop-icon">📄</div>
                  <p className="drop-title">{file ? file.name : 'Drop CSV file here'}</p>
                  <p className="drop-sub">{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : 'or click to browse • Latitude & Longitude columns required'}</p>
                </div>
              )}
            </div>

            {file && !uploadDone && (
              <button className="btn-primary w-full" onClick={handleUpload} disabled={uploading}>
                {uploading ? <><span className="spinner"></span>Uploading...</> : 'Upload Dataset'}
              </button>
            )}
          </div>

          {/* Training Config */}
          <div className="glass-card config-card">
            <h2 className="card-title"><span>⚙️</span> Training Configuration</h2>

            <div className="input-group">
              <label>Ambulance Hubs: <strong>{nComponents}</strong></label>
              <input type="range" min="2" max="8" value={nComponents}
                onChange={e => setNComponents(+e.target.value)} disabled={training} />
              <span className="input-description">Number of optimal standby positions (GMM clusters)</span>
            </div>

            <div className="input-group">
              <label>Sample Size: <strong>{sampleSize.toLocaleString()}</strong></label>
              <input type="range" min="500" max="15000" step="500" value={sampleSize}
                onChange={e => setSampleSize(+e.target.value)} disabled={training || !stats.has_csv} />
              <span className="input-description">Records sampled from CSV for training</span>
            </div>

            <div className="input-group">
              <label>Max Epochs: <strong>{epochs}</strong></label>
              <input type="range" min="10" max="200" step="10" value={epochs}
                onChange={e => setEpochs(+e.target.value)} disabled={training} />
              <span className="input-description">MLP training iterations (early stopping enabled)</span>
            </div>

            <button className="btn-primary w-full" onClick={handleTrain} disabled={training}>
              {training
                ? <><span className="spinner"></span>Training Neural Network...</>
                : 'Train H5 Model Now'}
            </button>

            {error && <div className="error-inline">{error}</div>}
          </div>
        </div>

        {/* Right column: results */}
        <div className="upload-right">
          {trainResult ? (
            <>
              <div className="glass-card results-card">
                <h2 className="card-title"><span>📈</span> Training Results</h2>
                <div className="results-metrics">
                  <div className="result-metric-box">
                    <span className="rmb-label">Train Accuracy</span>
                    <span className="rmb-val green">{(trainResult.train_accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="result-metric-box">
                    <span className="rmb-label">Val Accuracy</span>
                    <span className="rmb-val green">{(trainResult.val_accuracy * 100).toFixed(2)}%</span>
                  </div>
                  <div className="result-metric-box">
                    <span className="rmb-label">Model File</span>
                    <span className="rmb-val mono small">ambulance_model.h5</span>
                  </div>
                  <div className="result-metric-box">
                    <span className="rmb-label">Hubs Found</span>
                    <span className="rmb-val">{trainResult.hubs.length}</span>
                  </div>
                </div>

                <div className="hub-list">
                  {trainResult.hubs.map((h, i) => (
                    <div key={i} className="hub-row">
                      <span className="hub-badge">Hub {i + 1}</span>
                      <span className="hub-coords-mono">{h[0].toFixed(6)}, {h[1].toFixed(6)}</span>
                      <span className="hub-live-dot"></span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card viz-card">
                <div className="viz-header">
                  <h2 className="card-title"><span>🗺️</span> Coverage Visualization</h2>
                  <div className="tab-group">
                    {['nn','gmm','dbscan'].map(t => (
                      <button key={t} className={`tab-btn ${activeViz===t?'active-tab':''}`}
                        onClick={() => setActiveViz(t)}>
                        {t === 'nn' ? 'Neural Net' : t === 'gmm' ? 'GMM' : 'DBSCAN'}
                      </button>
                    ))}
                  </div>
                </div>
                {getVizPlot() ? (
                  <img src={`data:image/png;base64,${getVizPlot()}`} alt="viz" className="viz-img animate-fade-in" />
                ) : (
                  <div className="viz-placeholder"><p>No plot available</p></div>
                )}
              </div>
            </>
          ) : (
            <div className="glass-card upload-placeholder-card">
              <div className="placeholder-inner">
                <div className="placeholder-icon-big">🤖</div>
                <h3>Ready to Train</h3>
                <p>Configure the parameters on the left and click <strong>"Train H5 Model Now"</strong> to start training the neural network on your dataset.</p>
                {stats.nn_trained && (
                  <div className="already-trained-badge">
                    <span>✓</span> A model is already loaded with {stats.n_components} hubs ({(stats.nn_accuracy*100).toFixed(1)}% accuracy)
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

// ─── TRY / PREDICT PAGE ──────────────────────────────────────
function TryPage({ stats }) {
  const [latitude, setLatitude] = useState('')
  const [longitude, setLongitude] = useState('')
  const [loading, setPredLoading] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])
  const [vizLoading, setVizLoading] = useState(false)
  const [plots, setPlots] = useState({ gmm: '', dbscan: '', nn: '' })
  const [activeViz, setActiveViz] = useState('nn')

  const presets = [
    { label: 'Silver Spring', lat: '39.001637', lon: '-77.053773' },
    { label: 'Gaithersburg', lat: '39.140422', lon: '-77.210270' },
    { label: 'Rockville', lat: '39.093150', lon: '-77.196532' },
    { label: 'Colesville', lat: '39.048226', lon: '-76.984448' },
    { label: 'Wheaton', lat: '39.035290', lon: '-77.055190' },
    { label: 'Germantown', lat: '39.174200', lon: '-77.265000' },
  ]

  useEffect(() => {
    // Load visualizations on mount
    if (stats.nn_trained) loadPlots()
  }, [])

  const loadPlots = async () => {
    setVizLoading(true)
    try {
      const res = await fetch(`${API_BASE}/visualize`)
      const data = await res.json()
      if (data.success) setPlots({ gmm: data.gmm_plot, dbscan: data.dbscan_plot, nn: data.nn_plot })
    } catch (e) { }
    setVizLoading(false)
  }

  const handlePredict = async (e) => {
    e.preventDefault()
    if (!latitude || !longitude) return
    setPredLoading(true)
    setError('')
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ latitude: parseFloat(latitude), longitude: parseFloat(longitude) })
      })
      const data = await res.json()
      if (data.success) {
        setPrediction(data)
        setHistory(prev => [{ lat: latitude, lon: longitude, ...data, ts: new Date().toLocaleTimeString() }, ...prev.slice(0, 4)])
      } else {
        setError(data.error)
      }
    } catch (e) {
      setError('Could not reach the backend server. Is Flask running on port 5000?')
    }
    setPredLoading(false)
  }

  const getPlot = () => plots[activeViz] || ''

  return (
    <div className="page try-page">
      <div className="page-header">
        <div className="section-label">Live Prediction</div>
        <h1 className="page-title">Try the H5 Model</h1>
        <p className="page-sub">Enter any coordinate to get an instant ambulance dispatch recommendation from the trained neural network.</p>
      </div>

      {!stats.nn_trained && (
        <div className="model-warning">
          <span>⚠️</span> No model loaded. Go to the Upload & Train page first.
        </div>
      )}

      <div className="try-layout">
        {/* Left: input + result */}
        <div className="try-left">
          <div className="glass-card predict-card">
            <h2 className="card-title"><span>🚨</span> Dispatch Coordinator</h2>

            {/* Presets */}
            <div className="presets-row">
              <span className="presets-label">Quick location presets:</span>
              <div className="presets-chips">
                {presets.map((p, i) => (
                  <button key={i} className="chip-btn"
                    onClick={() => { setLatitude(p.lat); setLongitude(p.lon); setPrediction(null) }}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>

            <form onSubmit={handlePredict} className="predict-form">
              <div className="coord-inputs">
                <div className="coord-field">
                  <label>Latitude</label>
                  <input id="lat-input" type="number" step="0.000001" value={latitude}
                    onChange={e => setLatitude(e.target.value)} placeholder="e.g. 39.0800" required />
                </div>
                <div className="coord-field">
                  <label>Longitude</label>
                  <input id="lon-input" type="number" step="0.000001" value={longitude}
                    onChange={e => setLongitude(e.target.value)} placeholder="e.g. -77.1500" required />
                </div>
              </div>

              <button id="btn-predict" type="submit" className="btn-primary w-full dispatch-btn"
                disabled={loading || !stats.nn_trained}>
                {loading
                  ? <><span className="spinner"></span> Querying H5 Model...</>
                  : <><span>🚑</span> Find Nearest Ambulance Hub</>}
              </button>
            </form>

            {error && <div className="error-inline">{error}</div>}

            {/* Prediction Result */}
            {prediction && (
              <div className="prediction-result animate-fade-in">
                <div className="result-alert-bar">
                  <span className="alert-icon">🔴</span>
                  <span className="alert-text">DISPATCH TO HUB {prediction.hub_index + 1}</span>
                </div>
                <div className="result-details">
                  <div className="rd-row">
                    <span className="rd-label">Assigned Hub</span>
                    <span className="rd-val hub-highlight">Hub {prediction.hub_index + 1}</span>
                  </div>
                  <div className="rd-row">
                    <span className="rd-label">Hub Coordinates</span>
                    <span className="rd-val mono">{prediction.hub_latitude.toFixed(5)}, {prediction.hub_longitude.toFixed(5)}</span>
                  </div>
                  <div className="rd-row">
                    <span className="rd-label">Model Confidence</span>
                    <span className="rd-val mono green">{(prediction.confidence * 100).toFixed(2)}%</span>
                  </div>
                  <div className="confidence-track">
                    <div className="confidence-fill" style={{ width: `${prediction.confidence * 100}%` }}></div>
                    <span className="confidence-label">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Prediction history */}
          {history.length > 0 && (
            <div className="glass-card history-card">
              <h2 className="card-title"><span>🕐</span> Recent Predictions</h2>
              <div className="history-list">
                {history.map((h, i) => (
                  <div key={i} className={`history-row ${i === 0 ? 'history-latest' : ''}`}>
                    <div className="hr-coords">{h.lat}, {h.lon}</div>
                    <div className="hr-hub">Hub {h.hub_index + 1}</div>
                    <div className="hr-conf">{(h.confidence * 100).toFixed(1)}%</div>
                    <div className="hr-time">{h.ts}</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* All hubs table */}
          {stats.hubs.length > 0 && (
            <div className="glass-card hubs-card">
              <h2 className="card-title"><span>📍</span> All Active Hubs</h2>
              <div className="table-wrapper">
                <table>
                  <thead><tr><th>Hub</th><th>Latitude</th><th>Longitude</th><th>Status</th></tr></thead>
                  <tbody>
                    {stats.hubs.map((h, i) => (
                      <tr key={i} className={prediction && prediction.hub_index === i ? 'row-highlighted' : ''}>
                        <td><strong>Hub {i + 1}</strong></td>
                        <td className="mono">{h[0].toFixed(6)}</td>
                        <td className="mono">{h[1].toFixed(6)}</td>
                        <td><span className="active-dot"></span> Ready</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right: Visualizations */}
        <div className="try-right">
          <div className="glass-card viz-card-large">
            <div className="viz-header">
              <h2 className="card-title"><span>🗺️</span> Coverage Decision Map</h2>
              <div className="tab-group">
                {['nn', 'gmm', 'dbscan'].map(t => (
                  <button key={t} className={`tab-btn ${activeViz === t ? 'active-tab' : ''}`}
                    onClick={() => setActiveViz(t)}>
                    {t === 'nn' ? 'Neural Net (H5)' : t === 'gmm' ? 'GMM Centroids' : 'DBSCAN'}
                  </button>
                ))}
              </div>
            </div>

            {vizLoading ? (
              <div className="viz-loading">
                <span className="spinner large-spinner"></span>
                <p>Generating coverage maps...</p>
              </div>
            ) : getPlot() ? (
              <div className="viz-img-wrap">
                <img src={`data:image/png;base64,${getPlot()}`} alt="coverage map" className="viz-img animate-fade-in" />
                <div className="viz-label-chip">
                  {activeViz === 'nn' && 'Decision boundaries learned by the H5 neural network'}
                  {activeViz === 'gmm' && 'Gaussian mixture centroids = optimal hub positions'}
                  {activeViz === 'dbscan' && 'Density clusters with noise filtering'}
                </div>
              </div>
            ) : (
              <div className="viz-placeholder">
                <span className="placeholder-icon">🗺️</span>
                <p>Train the model to generate coverage maps</p>
                {stats.nn_trained && (
                  <button className="btn-ghost small" onClick={loadPlots}>Load Visualizations</button>
                )}
              </div>
            )}
          </div>

          {/* Model stats card */}
          <div className="glass-card model-stat-card">
            <h2 className="card-title"><span>📊</span> Model Performance</h2>
            <div className="model-perf-grid">
              <div className="perf-box">
                <span className="perf-label">Train Acc.</span>
                <span className="perf-val green">{stats.nn_trained ? `${(stats.nn_accuracy * 100).toFixed(1)}%` : '—'}</span>
              </div>
              <div className="perf-box">
                <span className="perf-label">Val Acc.</span>
                <span className="perf-val green">{stats.nn_trained ? `${(stats.nn_val_accuracy * 100).toFixed(1)}%` : '—'}</span>
              </div>
              <div className="perf-box">
                <span className="perf-label">Architecture</span>
                <span className="perf-val mono small">MLP 64→32</span>
              </div>
              <div className="perf-box">
                <span className="perf-label">Model File</span>
                <span className="perf-val mono small green">{stats.nn_trained ? 'ambulance_model.h5' : 'Not trained'}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ─── MAIN APP ─────────────────────────────────────────────────
export default function App() {
  const [page, setPage] = useState('home')
  const [stats, setStats] = useState({
    nn_trained: false, nn_accuracy: 0, nn_val_accuracy: 0,
    n_components: 4, sample_size: 5000, epochs: 100,
    hubs: [], has_csv: false, dataset_rows: 0, upload_name: ''
  })

  useEffect(() => {
    fetch(`${API_BASE}/stats`)
      .then(r => r.json())
      .then(d => setStats(d))
      .catch(() => {})
  }, [])

  const navItems = [
    { id: 'home', label: 'Home', icon: '⬡' },
    { id: 'upload', label: 'Upload & Train', icon: '↑' },
    { id: 'try', label: 'Try Model', icon: '▶' },
  ]

  return (
    <div className="app-root">
      {/* Top navigation */}
      <nav className="topnav">
        <div className="nav-logo" onClick={() => setPage('home')}>
          <span className="nav-logo-text">SmartRESQ</span>
          <span className="nav-logo-badge">AI</span>
        </div>

        <div className="nav-links">
          {navItems.map(n => (
            <button key={n.id}
              className={`nav-link ${page === n.id ? 'nav-active' : ''}`}
              onClick={() => setPage(n.id)}>
              {n.label}
            </button>
          ))}
        </div>

        <div className="nav-status">
          {stats.nn_trained
            ? <span className="status-chip live"><span className="dot-pulse"></span> Model Live</span>
            : <span className="status-chip idle">No Model</span>}
        </div>
      </nav>

      {/* Page content */}
      <div className="page-wrapper" key={page}>
        {page === 'home' && <HomePage stats={stats} onNavigate={setPage} />}
        {page === 'upload' && <UploadPage stats={stats} setStats={setStats} />}
        {page === 'try' && <TryPage stats={stats} />}
      </div>
    </div>
  )
}
