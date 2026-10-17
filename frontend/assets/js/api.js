/**
 * VitalSync — Same-origin REST API client.
 *
 * All requests use relative paths under `/api` so the browser never talks to a
 * hardcoded host or exposes any server-side credentials.
 */

const API_BASE = "/api";
const DEFAULT_TIMEOUT_MS = 28000;

export class ApiError extends Error {
  constructor(message, status, detail = null, path = "") {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
    this.path = path;
  }
}

function buildUrl(path, query = {}) {
  const cleanPath = path.startsWith("/") ? path : `/${path}`;
  const url = `${API_BASE}${cleanPath}`;
  const params = new URLSearchParams();
  Object.entries(query || {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      params.set(key, String(value));
    }
  });
  const qs = params.toString();
  return qs ? `${url}?${qs}` : url;
}

export async function apiRequest(path, { method = "GET", query = null, body = null, timeoutMs = DEFAULT_TIMEOUT_MS, signal = null } = {}) {
  const url = buildUrl(path, query);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error("Request timed out")), timeoutMs);

  if (signal) {
    if (signal.aborted) {
      clearTimeout(timer);
      throw new ApiError("Request aborted", 0, null, url);
    }
    signal.addEventListener("abort", () => controller.abort(), { once: true });
  }

  const options = {
    method,
    headers: {
      Accept: "application/json",
    },
    signal: controller.signal,
  };

  if (body !== null && body !== undefined) {
    options.headers["Content-Type"] = "application/json";
    options.body = JSON.stringify(body);
  }

  try {
    const response = await fetch(url, options);
    const text = await response.text();
    let payload = null;
    if (text) {
      try {
        payload = JSON.parse(text);
      } catch (_err) {
        if (!response.ok) {
          throw new ApiError(`HTTP ${response.status}: ${text.slice(0, 140)}`, response.status, text, url);
        }
        throw new ApiError("Invalid JSON response from server", response.status, text.slice(0, 140), url);
      }
    }

    if (!response.ok) {
      const detail = payload?.detail || payload;
      const errCode = detail?.error || payload?.error || `HTTP_${response.status}`;
      const hint = detail?.hint || payload?.hint || detail?.message || "";
      const msg = hint ? `${errCode}: ${hint}` : `Request failed (${errCode}, HTTP ${response.status})`;
      throw new ApiError(msg, response.status, detail, url);
    }

    return payload;
  } catch (err) {
    if (err instanceof ApiError) {
      throw err;
    }
    if (err?.name === "AbortError") {
      throw new ApiError("Request timed out or was cancelled.", 408, null, url);
    }
    throw new ApiError(err?.message || "Network connection error", 0, null, url);
  } finally {
    clearTimeout(timer);
  }
}

export const api = {
  // System & provenance
  getHealth: () => apiRequest("/health"),
  getConfig: () => apiRequest("/config"),
  getDataSources: () => apiRequest("/datasources"),
  getArchitecture: () => apiRequest("/architecture"),
  getProfile: () => apiRequest("/profile"),
  getProfileSummary: () => apiRequest("/profile/summary"),

  // Patients
  getPatients: () => apiRequest("/patients"),
  getPatientDetail: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}`, { query: params }),
  getPatientEhr: (patientId) => apiRequest(`/patients/${encodeURIComponent(patientId)}/ehr`),
  getPatientSensors: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/sensors`, { query: params }),
  getPatientCharts: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/charts`, { query: params }),

  // Predictions & explainability
  getPrediction: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/prediction`, { query: params }),
  getPredictionExplain: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/prediction/explain`, { query: params }),
  getPredictionTrajectory: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/prediction/trajectory`, { query: params }),
  getPredictionHistory: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/prediction/history`, { query: params }),

  // Digital Twin
  getTwinState: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/twin`, { query: params }),
  getTwinFusion: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/twin/fusion`, { query: params }),
  getTwinBaseline: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/twin/baseline`, { query: params }),
  getTwinDomainDetail: (patientId, domain, params) =>
    apiRequest(`/patients/${encodeURIComponent(patientId)}/twin/domains/${encodeURIComponent(domain)}`, { query: params }),

  // Timeline
  getTimeline: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/timeline`, { query: params }),
  getTimelinePattern: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/timeline/pattern`, { query: params }),
  getTimelineDay: (patientId, params) => apiRequest(`/patients/${encodeURIComponent(patientId)}/timeline/day`, { query: params }),

  // Simulation
  getSnapshot: (params) => apiRequest("/simulation/snapshot", { query: params }),
  startSimulation: (payload) => apiRequest("/simulation/start", { method: "POST", body: payload }),
  stepSimulation: (payload) => apiRequest("/simulation/step", { method: "POST", body: payload }),
  pauseSimulation: (sessionId) => apiRequest("/simulation/pause", { method: "POST", body: { session_id: sessionId } }),
  resumeSimulation: (sessionId) => apiRequest("/simulation/resume", { method: "POST", body: { session_id: sessionId } }),
  resetSimulation: (sessionId) => apiRequest("/simulation/reset", { method: "POST", body: { session_id: sessionId } }),
  jumpSimulation: (sessionId, minutes) => apiRequest("/simulation/jump", { method: "POST", body: { session_id: sessionId, minutes } }),
  setSimulationSpeed: (sessionId, speedMinutesPerTick, tickIntervalMs = null) =>
    apiRequest("/simulation/speed", {
      method: "POST",
      body: { session_id: sessionId, speed_minutes_per_tick: speedMinutesPerTick, tick_interval_ms: tickIntervalMs },
    }),
  getSimulationState: (sessionId) => apiRequest("/simulation/state", { query: { session_id: sessionId } }),

  // Clinical Insights (Groq / Deterministic Fallback)
  getInsightsStatus: () => apiRequest("/insights/status"),
  createInsight: (payload) => apiRequest("/insights", { method: "POST", body: payload, timeoutMs: 32000 }),
  getInsightContext: (params) => apiRequest("/insights/context", { query: params }),

  // Model Validation & Governance
  getModelInfo: () => apiRequest("/model/info"),
  getModelMetrics: (split = null) => apiRequest("/model/metrics", { query: split ? { split } : null }),
  getModelCalibration: () => apiRequest("/model/calibration"),
  getModelImportance: (limit = 25) => apiRequest("/model/importance", { query: { limit } }),
  getModelFeatures: (group = null) => apiRequest("/model/features", { query: group ? { group } : null }),
  getModelDataset: () => apiRequest("/model/dataset"),
  getModelExplain: (params) => apiRequest("/model/explain", { query: params }),
  getModelLimitations: () => apiRequest("/model/limitations"),
};
