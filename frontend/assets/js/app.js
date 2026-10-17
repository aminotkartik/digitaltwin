/**
 * VitalSync — Main Clinical Analytics Application Controller.
 *
 * Orchestrates patient selection, simulation replay, multi-tab synchronization,
 * Chart.js visualizations, TreeSHAP explainability, and clinical narratives.
 * All property accesses strictly match the FastAPI backend response schemas.
 */

import { api, ApiError } from "./api.js";
import {
  renderMainGlucoseChart,
  renderRiskTimelineChart,
  renderSensorSparkline,
  renderTwinRadarChart,
  renderLiveSignalCharts,
  renderAttributionCharts,
  renderGlobalImportanceChart,
  renderDayViewChart,
  renderValidationCharts,
} from "./charts.js";

const state = {
  patientId: "DT-1047",
  patientsList: [],
  dashboard: null,
  activeTab: "overview",

  // Simulation state
  sessionId: null,
  sessionRunning: false,
  currentIndex: null,
  currentClock: "10:45",
  simSpeedMinutes: 15,
  tickIntervalMs: 1100,
  tickTimer: null,
  simRequestInFlight: false,

  // Tab-specific cached payloads
  signalWindowHours: 6,
  sensorsPayload: null,
  ehrPayload: null,
  explainPayload: null,
  globalImportance: null,
  featureCatalogue: null,
  dayPayload: null,
  patternPayload: null,
  timelinePayload: null,
  timelineHours: 24,
  timelineCategory: "",
  timelineSeverity: "",
  validationLoaded: false,

  // Insights state
  insightAction: "summarize_changes",
  insightRequestCounter: 0,
};

function escapeHtml(val) {
  if (val === null || val === undefined) return "";
  return String(val)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtNum(val, decimals = 0, fallback = "—") {
  if (val === null || val === undefined || Number.isNaN(Number(val))) return fallback;
  return Number(val).toFixed(decimals);
}

function fmtClockShort(str) {
  if (!str) return "—";
  const s = String(str);
  if (/^\d{2}:\d{2}/.test(s)) return s.slice(0, 5);
  const m = s.match(/T(\d{2}:\d{2})/);
  return m ? m[1] : s;
}

function severityBadgeClass(levelOrStatus) {
  const s = String(levelOrStatus || "").toLowerCase();
  if (["high", "critical", "unstable", "poor", "alert", "adverse"].includes(s)) return "badge-high";
  if (["moderate", "warning", "elevated", "caution"].includes(s)) return "badge-moderate";
  if (["watch", "fair", "info", "neutral", "borderline"].includes(s)) return "badge-watch";
  return "badge-good";
}

function showAlert(message) {
  const banner = document.getElementById("global-alert-banner");
  const text = document.getElementById("global-alert-text");
  if (!banner || !text) return;
  text.textContent = message;
  banner.hidden = false;
}

function hideAlert() {
  const banner = document.getElementById("global-alert-banner");
  if (banner) banner.hidden = true;
}

// ============================================================================
// System Header & Cohort Roster Rendering
// ============================================================================

function renderSystemHeader(health, config) {
  const dotApi = document.getElementById("sys-dot-api");
  const labelApi = document.getElementById("sys-label-api");
  if (dotApi && labelApi) {
    const ok = health?.status === "ok" && health?.ready !== false;
    dotApi.className = `sys-dot ${ok ? "ok" : "err"}`;
    const nPatients = health?.components?.patient_repository?.patients_loaded ?? 3;
    labelApi.textContent = `API: ${ok ? "Online" : "Degraded"} (${nPatients} Twins)`;
  }

  const dotModel = document.getElementById("sys-dot-model");
  const labelModel = document.getElementById("sys-label-model");
  if (dotModel && labelModel) {
    const modelComp = health?.components?.model || config?.model || {};
    const ready = Boolean(modelComp.trained || modelComp.ready);
    dotModel.className = `sys-dot ${ready ? "ok" : "warn"}`;
    labelModel.textContent = ready
      ? `Model: ${config?.model?.estimator || "XGBoost Hist GBM"}`
      : "Model: Fallback";
  }

  const dotLlm = document.getElementById("sys-dot-llm");
  const labelLlm = document.getElementById("sys-label-llm");
  if (dotLlm && labelLlm) {
    const groqInfo = config?.groq || health?.components?.groq || {};
    if (groqInfo.enabled && !groqInfo.fallback_active) {
      dotLlm.className = "sys-dot ok";
      labelLlm.textContent = `Groq LLM: ${groqInfo.model || "Active"}`;
    } else {
      dotLlm.className = "sys-dot warn";
      labelLlm.textContent = "Insights: Deterministic Engine";
    }
  }

  const versionBadge = document.getElementById("app-version-badge");
  if (versionBadge && (config?.app?.version || health?.version)) {
    versionBadge.textContent = `v${config?.app?.version || health?.version}`;
  }
  const disclaimerEl = document.getElementById("header-disclaimer-text");
  if (disclaimerEl && (config?.disclaimer || health?.disclaimer)) {
    disclaimerEl.textContent = config?.disclaimer || health?.disclaimer;
  }
}

function renderPatientRoster(patients) {
  const container = document.getElementById("patient-roster-grid");
  if (!container) return;
  state.patientsList = patients || [];

  container.innerHTML = state.patientsList
    .map((p) => {
      const pid = p.patient_id || p.id;
      const isActive = pid === state.patientId;
      const riskProb = p.current_risk ?? p.twin?.risk ?? 0;
      const riskPct = Math.round(Number(riskProb) * 100);
      const riskBand = p.twin?.band || (riskProb >= 0.7 ? "HIGH" : riskProb >= 0.35 ? "MODERATE" : "LOW");
      const riskBadge = severityBadgeClass(riskBand);
      const demoTag = p.is_demo_patient ? `<span class="demo-tag">Primary Demo</span>` : "";
      const twinScore = p.twin?.composite_index;

      return `
        <button
          type="button"
          class="patient-roster-card ${isActive ? "active" : ""}"
          data-patient-id="${escapeHtml(pid)}"
          aria-pressed="${isActive ? "true" : "false"}"
        >
          <div class="roster-card-top">
            <div>
              <div class="roster-patient-name">
                <span>${escapeHtml(p.name)}</span>
                <span class="roster-patient-id">${escapeHtml(pid)}</span>
                ${demoTag}
              </div>
              <div class="roster-condition">
                ${escapeHtml(p.age)}y ${escapeHtml(p.sex)} · ${escapeHtml(p.primary_condition)}
              </div>
            </div>
            <span class="badge ${riskBadge} tabular">${riskPct}% ${escapeHtml(riskBand)}</span>
          </div>

          <div class="roster-metrics-row">
            <div class="roster-mini-stat">
              <span class="roster-mini-label">Glucose</span>
              <span class="roster-mini-val">${fmtNum(p.current_glucose, 0)} mg/dL</span>
            </div>
            <div class="roster-mini-stat">
              <span class="roster-mini-label">2h Spike Risk</span>
              <span class="roster-mini-val">${riskPct}%</span>
            </div>
            <div class="roster-mini-stat">
              <span class="roster-mini-label">HbA1c</span>
              <span class="roster-mini-val">${fmtNum(p.hba1c_pct, 1)}%</span>
            </div>
            <div class="roster-mini-stat">
              <span class="roster-mini-label">Twin Index</span>
              <span class="roster-mini-val">${fmtNum(twinScore, 1)} / 100</span>
            </div>
          </div>
        </button>
      `;
    })
    .join("");

  container.querySelectorAll(".patient-roster-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      const nextId = btn.getAttribute("data-patient-id");
      if (nextId && nextId !== state.patientId) {
        switchPatient(nextId);
      }
    });
  });
}

// ============================================================================
// Dashboard Snapshot Rendering (Tab 1 + Shared Banner & Sim Bar)
// ============================================================================

function renderDashboardSnapshot(dashboard, sessionInfo = null) {
  if (!dashboard) return;
  state.dashboard = dashboard;
  const clock = dashboard.clock || {};
  const patient = dashboard.patient || {};
  const twinStatus = dashboard.twin_status || {};
  const prediction = dashboard.prediction || {};
  const twin = dashboard.twin || {};
  const baseline = dashboard.baseline_comparison || {};
  const chart = dashboard.chart || {};
  const sleep = dashboard.sleep || {};
  const alerts = dashboard.alerts || [];
  const timeline = dashboard.timeline || {};
  const session = sessionInfo || dashboard.session || null;

  state.currentIndex = clock.index ?? state.currentIndex;
  state.currentClock = fmtClockShort(clock.clock || state.currentClock);

  const pid = patient.patient_id || state.patientId;

  // Keep active patient card in roster synchronized with current replay step
  if (state.patientsList.length > 0) {
    const idx = state.patientsList.findIndex((p) => (p.patient_id || p.id) === pid);
    if (idx >= 0) {
      state.patientsList[idx].current_glucose =
        prediction.trajectory?.anchor_glucose_mgdl ?? state.patientsList[idx].current_glucose;
      state.patientsList[idx].current_risk = prediction.probability ?? state.patientsList[idx].current_risk;
      if (state.patientsList[idx].twin) {
        state.patientsList[idx].twin.composite_index = twin.composite_index ?? state.patientsList[idx].twin.composite_index;
        state.patientsList[idx].twin.risk = prediction.probability ?? state.patientsList[idx].twin.risk;
        state.patientsList[idx].twin.band = prediction.risk?.label || state.patientsList[idx].twin.band;
      }
      renderPatientRoster(state.patientsList);
    }
  }

  // 1. Update Simulation Clock & Progress Bar
  const clockTimeEl = document.getElementById("sim-clock-time");
  const clockDateEl = document.getElementById("sim-clock-date");
  const phaseBadgeEl = document.getElementById("sim-phase-badge");
  const stepBadgeEl = document.getElementById("sim-step-badge");
  const shortClock = fmtClockShort(clock.clock);
  if (clockTimeEl) clockTimeEl.textContent = shortClock;
  if (clockDateEl) {
    clockDateEl.textContent = clock.date || "Scenario Day";
  }
  if (phaseBadgeEl) {
    const phase = String(clock.phase || "live");
    phaseBadgeEl.textContent = phase.replace("_", " ").toUpperCase();
    phaseBadgeEl.className = `phase-badge ${phase}`;
  }
  if (stepBadgeEl) {
    stepBadgeEl.textContent = `Sample ${clock.index ?? "—"} / ${clock.last_index ?? "—"}`;
  }

  // Highlight active preset clock pill if matched
  document.querySelectorAll(".js-clock-preset").forEach((btn) => {
    const c = btn.getAttribute("data-clock");
    btn.classList.toggle("active-pill", c === shortClock);
  });

  // Update progress bar
  const progressFill = document.getElementById("sim-progress-fill");
  const progressText = document.getElementById("sim-progress-text");
  if (progressFill && progressText) {
    let pct = 0;
    if (session && session.progress_pct !== undefined) {
      pct = Number(session.progress_pct);
    } else if (clock.index !== undefined && clock.last_index) {
      // Scenario day starts around index 576 (48h onboarding)
      const dayStart = 576;
      const denom = Math.max(1, clock.last_index - dayStart);
      pct = Math.max(0, Math.min(100, ((clock.index - dayStart) / denom) * 100));
    }
    progressFill.style.width = `${pct.toFixed(1)}%`;
    progressText.textContent = `${Math.round(pct)}% of day`;
  }

  // 2. Render Patient Banner
  const nameEl = document.getElementById("patient-name");
  const idEl = document.getElementById("patient-id");
  const demoBadge = document.getElementById("patient-demo-badge");
  const twinBadge = document.getElementById("patient-twin-badge");
  const demoLine = document.getElementById("patient-demographics");
  const chipsContainer = document.getElementById("patient-clinical-chips");

  if (nameEl) nameEl.textContent = patient.name || pid;
  if (idEl) idEl.textContent = `${pid} · ${patient.mrn || ""}`;
  if (demoBadge) demoBadge.hidden = !patient.is_demo_patient;
  if (twinBadge) {
    const compWord = twin.composite_status?.word || twinStatus.value || "Synchronized";
    twinBadge.textContent = `Twin: ${compWord} (${fmtNum(twin.composite_index, 1)})`;
    twinBadge.className = `badge ${severityBadgeClass(twin.composite_status?.tone || twin.composite_status?.status)}`;
  }
  if (demoLine) {
    const meds = Array.isArray(patient.medications) ? patient.medications.join(" · ") : "Lifestyle management";
    demoLine.innerHTML = `
      <span><strong>${escapeHtml(patient.age)}y ${escapeHtml(patient.sex)}</strong></span>
      <span>·</span>
      <span>${escapeHtml(patient.primary_condition)}</span>
      <span>·</span>
      <span>${escapeHtml(patient.care_setting || "")}</span>
      <span>·</span>
      <span>Rx: ${escapeHtml(meds)}</span>
    `;
  }
  if (chipsContainer) {
    const bp = patient.blood_pressure
      ? `${patient.blood_pressure.systolic}/${patient.blood_pressure.diastolic}`
      : "—";
    const anchorG = prediction.trajectory?.anchor_glucose_mgdl;
    chipsContainer.innerHTML = `
      <span class="clinical-chip tabular">Current Glucose: <strong>${fmtNum(anchorG, 0)} mg/dL</strong></span>
      <span class="clinical-chip tabular">HbA1c: <strong>${fmtNum(patient.hba1c_pct, 1)}%</strong></span>
      <span class="clinical-chip tabular">BMI: <strong>${fmtNum(patient.bmi, 1)} kg/m²</strong></span>
      <span class="clinical-chip tabular">BP: <strong>${escapeHtml(bp)} mmHg</strong></span>
      <span class="clinical-chip tabular">Engineered Features: <strong>${twinStatus.feature_count || 84}</strong></span>
    `;
  }

  // Twin Sync & Quality Box
  const qualBadge = document.getElementById("twin-quality-badge");
  const sourcesChips = document.getElementById("twin-sources-chips");
  const storyArc = document.getElementById("patient-story-arc");
  if (qualBadge) {
    const compPct = Math.round(Number(prediction.data_quality?.completeness ?? 1) * 100);
    qualBadge.textContent = `Completeness: ${compPct}%`;
    qualBadge.className = `badge ${compPct >= 95 ? "badge-good" : "badge-watch"}`;
  }
  if (sourcesChips && Array.isArray(twinStatus.sources_online)) {
    sourcesChips.innerHTML = twinStatus.sources_online
      .map(
        (s) =>
          `<span class="source-chip ${s.online ? "" : "offline"}">${escapeHtml(s.source)}</span>`
      )
      .join("");
  }
  if (storyArc) {
    const conds = (patient.conditions || []).slice(0, 3).join(" · ");
    storyArc.textContent = conds ? `Documented conditions: ${conds}` : "";
  }

  // 3. Hero Card: 2-Hour Risk Forecast
  const riskObj = prediction.risk || {};
  const riskPct = Number(riskObj.percentage ?? Number(prediction.probability || 0) * 100);
  const riskBand = String(riskObj.band || "moderate").toLowerCase();
  const riskValEl = document.getElementById("risk-pct-value");
  const riskBadgeEl = document.getElementById("risk-level-badge");
  const riskMarkerEl = document.getElementById("risk-scale-marker");
  const peakP90El = document.getElementById("risk-peak-p90");
  const peakMedEl = document.getElementById("risk-peak-median");
  const anchorChipEl = document.getElementById("chart-anchor-readout");

  if (riskValEl) {
    riskValEl.textContent = `${fmtNum(riskPct, 1)}%`;
    riskValEl.className = `risk-big-pct tabular ${riskBand}`;
  }
  if (riskBadgeEl) {
    riskBadgeEl.textContent = `${riskObj.label || riskBand.toUpperCase()} RISK`;
    riskBadgeEl.className = `badge ${severityBadgeClass(riskBand)}`;
  }
  if (riskMarkerEl) {
    const clamped = Math.max(2, Math.min(98, riskPct));
    riskMarkerEl.style.left = `${clamped}%`;
  }
  const traj = prediction.trajectory || {};
  if (peakP90El) {
    peakP90El.textContent = `${fmtNum(traj.projected_peak_p90, 0)} mg/dL`;
  }
  if (peakMedEl) {
    peakMedEl.textContent = `Median p50 peak: ${fmtNum(traj.projected_peak_p50, 0)} mg/dL`;
  }
  if (anchorChipEl) {
    anchorChipEl.textContent = `Now (${shortClock}): ${fmtNum(traj.anchor_glucose_mgdl, 0)} mg/dL`;
  }

  // Above-threshold explanation callout
  const aboveCallout = document.getElementById("risk-above-threshold-callout");
  if (aboveCallout) {
    const aat = prediction.already_above_threshold;
    if (aat && aat.above_threshold) {
      aboveCallout.hidden = false;
      aboveCallout.textContent =
        aat.explanation ||
        `Current glucose (${fmtNum(aat.current_mgdl, 0)} mg/dL) is already above ${fmtNum(aat.threshold_mgdl, 0)} mg/dL; the model scores whether an additional ≥25 mg/dL rise will occur within 2 hours.`;
    } else {
      aboveCallout.hidden = true;
    }
  }

  // Observed Outcome Callout (when replaying historical timestamps)
  const outcomeCallout = document.getElementById("risk-outcome-callout");
  if (outcomeCallout) {
    const obs = prediction.observed_outcome;
    if (obs && obs.available) {
      outcomeCallout.hidden = false;
      const eventWord = obs.event_occurred ? "SPIKE OCCURRED" : "NO SPIKE OCCURRED";
      outcomeCallout.innerHTML = `
        <strong>Replay Ground-Truth Verification (+2h Horizon):</strong>
        <strong>${eventWord}</strong> — Peak observed glucose over the subsequent 2h window was
        <strong>${fmtNum(obs.peak_glucose_mgdl ?? obs.max_future_glucose_mgdl, 0)} mg/dL</strong>.
      `;
    } else {
      outcomeCallout.hidden = true;
    }
  }

  // Ensemble & Baseline Model Concordance
  const modelIdEl = document.getElementById("risk-model-id");
  const modelsCompareEl = document.getElementById("risk-models-compare");
  if (modelIdEl) {
    modelIdEl.textContent = prediction.model?.model_id || "vitalsync-gbm";
  }
  if (modelsCompareEl) {
    const primPct = fmtNum(Number(prediction.probability || 0) * 100, 1);
    const secPct = fmtNum(Number(prediction.secondary_probability || 0) * 100, 1);
    const threshPct = fmtNum(Number(prediction.decision_threshold || 0.44) * 100, 0);
    modelsCompareEl.innerHTML = `
      <div style="background: var(--bg-surface); padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border-subtle);">
        <div style="color: var(--text-dim); font-size: 0.68rem;">Primary GBM</div>
        <div class="tabular" style="font-weight: 700; color: var(--text-primary);">${primPct}%</div>
      </div>
      <div style="background: var(--bg-surface); padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border-subtle);">
        <div style="color: var(--text-dim); font-size: 0.68rem;">Logistic L2</div>
        <div class="tabular" style="font-weight: 700; color: var(--text-secondary);">${secPct}%</div>
      </div>
      <div style="background: var(--bg-surface); padding: 6px 8px; border-radius: 6px; border: 1px solid var(--border-subtle);">
        <div style="color: var(--text-dim); font-size: 0.68rem;">Alert Threshold</div>
        <div class="tabular" style="font-weight: 700; color: var(--accent-blue);">${threshPct}%</div>
      </div>
    `;
  }

  // Confidence & Data Quality Bars
  const confBadge = document.getElementById("confidence-score-badge");
  const confBars = document.getElementById("confidence-bars-container");
  const dqSummary = document.getElementById("data-quality-summary");
  const conf = prediction.confidence || {};
  if (confBadge) {
    const rawScore = Number(conf.score ?? 80);
    const cPct = rawScore <= 1 ? Math.round(rawScore * 100) : Math.round(rawScore);
    confBadge.textContent = `Confidence: ${cPct}%`;
    confBadge.className = `badge tabular ${cPct >= 70 ? "badge-good" : "badge-watch"}`;
  }
  if (confBars && conf.components && typeof conf.components === "object") {
    const labelMap = {
      data_completeness: "Signal Completeness",
      model_agreement: "Primary / Secondary Model Concordance",
      calibration_trust: "Band Calibration Reliability",
    };
    confBars.innerHTML = Object.entries(conf.components)
      .map(([k, v]) => {
        const pct = Math.round(Number(v || 0) * 100);
        return `
          <div class="conf-bar-row">
            <div class="conf-bar-label">
              <span>${escapeHtml(labelMap[k] || k)}</span>
              <span class="tabular">${pct}%</span>
            </div>
            <div class="conf-track">
              <div class="conf-fill" style="width: ${pct}%;"></div>
            </div>
          </div>
        `;
      })
      .join("");
  }
  if (dqSummary) {
    dqSummary.textContent = conf.interpretation || "";
  }

  // 4. Render Main Glucose Trajectory Chart + Annotations + Risk Timeline Chart
  renderMainGlucoseChart("chart-main-glucose", chart);
  renderRiskTimelineChart("chart-risk-timeline", chart.past, chart.bands);

  const annotationsEl = document.getElementById("chart-annotations-strip");
  if (annotationsEl) {
    const recentMeals = (timeline.events || []).filter((e) => e.category === "meal").slice(0, 3);
    const recentAct = (timeline.events || []).filter((e) => e.category === "activity").slice(0, 2);
    const items = [
      ...recentMeals.map(
        (m) =>
          `<span class="annotation-pill">🍽️ <strong>${escapeHtml(m.clock)}</strong> ${escapeHtml(m.title)}</span>`
      ),
      ...recentAct.map(
        (b) =>
          `<span class="annotation-pill">🏃 <strong>${escapeHtml(b.clock)}</strong> ${escapeHtml(b.title)}</span>`
      ),
    ];
    annotationsEl.innerHTML = items.length
      ? items.join("")
      : `<span style="color: var(--text-dim);">${escapeHtml(traj.reading_note || "Shaded fan represents the 10th–90th percentile empirical trajectory conditioned on current risk.")}</span>`;
  }

  // 5. Render Sensor Cards + Mini Sparklines
  renderSensorCards(dashboard.sensor_cards || []);

  // 6. Render Digital Twin 5-Domain Assessment + Radar
  renderTwinSection(twin);

  // 7. Render Personal Baseline Comparison Table + Sleep Summary
  renderBaselineAndSleep(baseline, sleep);

  // 8. Render Active Threshold Alerts & Recent Timeline
  renderOverviewAlertsAndTimeline(alerts, timeline.events || []);
}

function renderSensorCards(cards) {
  const grid = document.getElementById("sensor-cards-grid");
  if (!grid) return;
  grid.innerHTML = cards
    .map((c) => {
      const badgeCls = severityBadgeClass(c.status);
      const canvasId = `sparkline-${c.key}`;
      const chg =
        c.change_pct !== undefined && c.change_pct !== null
          ? `${Number(c.change_pct) >= 0 ? "+" : ""}${fmtNum(c.change_pct, 1)}% vs baseline`
          : "";
      let extraText = `Trend: ${c.trend || "stable"}`;
      if (c.extra?.slope_30min !== undefined) {
        extraText = `30m slope: ${c.extra.slope_30min >= 0 ? "+" : ""}${fmtNum(c.extra.slope_30min, 1)} ${c.extra.slope_unit || "mg/dL/h"}`;
      }
      return `
        <div class="sensor-card">
          <div>
            <div class="sensor-card-top">
              <span class="sensor-card-title">${escapeHtml(c.title || c.key)}</span>
              <span class="badge ${badgeCls}">${escapeHtml(c.status_word || c.status)}</span>
            </div>
            <div class="sensor-value-row" style="margin-top: 6px;">
              <span class="sensor-value-num tabular">${fmtNum(c.current, c.unit === "MET" ? 1 : 0)}</span>
              <span class="sensor-value-unit">${escapeHtml(c.unit)}</span>
            </div>
            <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 2px;">
              ${escapeHtml(extraText)}
            </div>
          </div>

          <div class="sparkline-box">
            <canvas id="${escapeHtml(canvasId)}"></canvas>
          </div>

          <div class="sensor-baseline-row tabular">
            <span>Baseline: ${fmtNum(c.baseline, 0)} ${escapeHtml(c.unit)}</span>
            <span>${escapeHtml(chg)}</span>
          </div>
        </div>
      `;
    })
    .join("");

  cards.forEach((c) => {
    renderSensorSparkline(`sparkline-${c.key}`, c.sparkline, c.status);
  });
}

function renderTwinSection(twin) {
  if (!twin) return;
  const scoreEl = document.getElementById("twin-composite-score");
  const badgeEl = document.getElementById("twin-composite-badge");

  if (scoreEl) scoreEl.textContent = `${fmtNum(twin.composite_index, 1)}/100`;
  if (badgeEl) {
    badgeEl.textContent = twin.composite_status?.word || twin.composite_status?.status || "Stable";
    badgeEl.className = `badge ${severityBadgeClass(twin.composite_status?.tone || twin.composite_status?.status)}`;
  }

  renderTwinRadarChart("chart-twin-radar", twin.radar || []);

  const domainsList = document.getElementById("twin-domains-list");
  if (domainsList && twin.domains) {
    const order = [
      "metabolic_stability",
      "glucose_stability",
      "cardiovascular_state",
      "recovery",
      "activity",
    ];
    domainsList.innerHTML = order
      .map((key) => {
        const d = twin.domains[key];
        if (!d) return "";
        const score = Number(d.score || 0);
        const badgeCls = severityBadgeClass(d.tone || d.status);
        const topContrib = d.contributors?.[0]
          ? `${d.contributors[0].term}: ${d.contributors[0].input} ${d.contributors[0].unit || ""}`
          : "";
        return `
          <button type="button" class="domain-row-card" data-domain-key="${escapeHtml(key)}">
            <div class="domain-row-top">
              <span class="domain-name">${escapeHtml(d.label)}</span>
              <div style="display: flex; align-items: center; gap: 6px;">
                <span class="badge ${badgeCls}">${escapeHtml(d.word || d.status)}</span>
                <span class="domain-score-num tabular">${fmtNum(score, 1)}</span>
              </div>
            </div>
            <div class="domain-bar-track">
              <div class="domain-bar-fill" style="width: ${Math.max(4, Math.min(100, score))}%;"></div>
            </div>
            <div style="font-size: 0.72rem; color: var(--text-muted); margin-top: 4px;">
              ${escapeHtml(topContrib)} · Click for formula &amp; sub-scores
            </div>
          </button>
        `;
      })
      .join("");

    domainsList.querySelectorAll(".domain-row-card").forEach((btn) => {
      btn.addEventListener("click", () => {
        const domainKey = btn.getAttribute("data-domain-key");
        if (domainKey) openDomainModal(domainKey);
      });
    });
  }

  const driversList = document.getElementById("twin-drivers-list");
  if (driversList && Array.isArray(twin.drivers)) {
    driversList.innerHTML = twin.drivers
      .map(
        (d) => `
          <div class="driver-item ${escapeHtml(d.severity || "info")}">
            <strong>${escapeHtml(d.domain)}:</strong> ${escapeHtml(d.statement || d.text || "")}
          </div>
        `
      )
      .join("");
  }
}

function renderBaselineAndSleep(baseline, sleep) {
  const tbody = document.getElementById("table-baseline-body");
  const rows = baseline?.metrics || baseline?.comparisons || [];
  if (tbody && Array.isArray(rows)) {
    tbody.innerHTML = rows
      .map((row) => {
        const flagBadge = severityBadgeClass(row.flag || row.direction);
        const deltaAbs = row.delta_absolute ?? row.delta ?? 0;
        const deltaSign = Number(deltaAbs) > 0 ? "+" : "";
        return `
          <tr>
            <td><strong>${escapeHtml(row.metric || row.label)}</strong></td>
            <td class="tabular"><strong>${fmtNum(row.current, 1)}</strong> ${escapeHtml(row.unit)}</td>
            <td class="tabular">${fmtNum(row.baseline ?? row.personal_median, 1)} ${escapeHtml(row.unit)}</td>
            <td class="tabular">${deltaSign}${fmtNum(deltaAbs, 1)} (${Number(row.delta_pct) > 0 ? "+" : ""}${fmtNum(row.delta_pct, 1)}%)</td>
            <td style="font-size: 0.75rem;">${escapeHtml(row.interpretation || "")}</td>
            <td><span class="badge ${flagBadge}">${escapeHtml(row.flag || row.direction || "ok")}</span></td>
          </tr>
        `;
      })
      .join("");
  }

  const sleepBox = document.getElementById("sleep-summary-box");
  const ln = sleep?.last_night || sleep;
  if (sleepBox && ln) {
    const effPct = Math.round(Number(ln.efficiency || 0) * 100);
    const deepPct = Math.round(Number(ln.deep_fraction || 0) * 100);
    const remPct = Math.round(Number(ln.rem_fraction || 0) * 100);
    sleepBox.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 6px;">
        <span style="font-size: 0.76rem; font-weight: 700; text-transform: uppercase; color: var(--accent-teal);">
          Prior-Night Sleep &amp; Recovery (${escapeHtml(ln.bed_time || "23:15")} → ${escapeHtml(ln.wake_time || "06:15")})
        </span>
        <span class="clinical-chip tabular">Overnight HRV: <strong>${fmtNum(ln.mean_hrv_ms, 0)} ms</strong></span>
      </div>
      <div class="metric-stat-grid">
        <div class="metric-stat-box">
          <div class="metric-stat-label">Sleep Duration</div>
          <div class="metric-stat-val tabular">${fmtNum(ln.sleep_duration_h, 1)} h</div>
          <div class="metric-stat-sub">Time in bed: ${fmtNum(ln.time_in_bed_h, 1)} h</div>
        </div>
        <div class="metric-stat-box">
          <div class="metric-stat-label">Sleep Efficiency</div>
          <div class="metric-stat-val tabular">${effPct}%</div>
          <div class="metric-stat-sub">${fmtNum(ln.awakenings, 0)} awakenings</div>
        </div>
        <div class="metric-stat-box">
          <div class="metric-stat-label">Deep / REM Share</div>
          <div class="metric-stat-val tabular">${deepPct}% / ${remPct}%</div>
          <div class="metric-stat-sub">${escapeHtml(ln.note || "")}</div>
        </div>
      </div>
    `;
  }
}

function renderOverviewAlertsAndTimeline(alerts, events) {
  const alertsEl = document.getElementById("overview-alerts-list");
  if (alertsEl) {
    if (!Array.isArray(alerts) || alerts.length === 0) {
      alertsEl.innerHTML = `
        <div class="driver-item" style="border-left-color: var(--status-good);">
          <strong>No Active High-Severity Threshold Alerts:</strong> Physiological streams are currently operating within safe threshold bounds.
        </div>
      `;
    } else {
      alertsEl.innerHTML = alerts
        .map(
          (a) => `
            <div class="timeline-item ${escapeHtml(a.level || a.severity || "warning")}">
              <div>
                <h4 class="timeline-item-title">${escapeHtml(a.title)}</h4>
                <p class="timeline-item-detail">${escapeHtml(a.detail)}</p>
              </div>
              <span class="timeline-item-time">${escapeHtml(a.clock)}</span>
            </div>
          `
        )
        .join("");
    }
  }

  const timelineEl = document.getElementById("overview-timeline-list");
  if (timelineEl) {
    const recent = (events || []).slice(0, 8);
    if (recent.length === 0) {
      timelineEl.innerHTML = `<div class="empty-state">No clinical events recorded in this window.</div>`;
    } else {
      timelineEl.innerHTML = recent.map(renderTimelineItemHtml).join("");
    }
  }
}

function renderTimelineItemHtml(ev) {
  const sev = escapeHtml(ev.severity || "info");
  const badgeCls = severityBadgeClass(ev.severity);
  return `
    <div class="timeline-item ${sev}">
      <div style="flex: 1;">
        <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 2px; flex-wrap: wrap;">
          <span class="badge ${badgeCls}">${escapeHtml(ev.category_label || ev.category || "event")}</span>
          <h4 class="timeline-item-title">${escapeHtml(ev.title)}</h4>
        </div>
        <p class="timeline-item-detail">${escapeHtml(ev.detail)}</p>
      </div>
      <div style="text-align: right;">
        <div class="timeline-item-time">${escapeHtml(ev.clock || "")}</div>
        <div class="mono" style="font-size: 0.7rem; color: var(--text-dim);">${escapeHtml(ev.date || "")}</div>
      </div>
    </div>
  `;
}

// ============================================================================
// Clinical Insights (POST /api/insights — Groq or Deterministic Fallback)
// ============================================================================

async function refreshClinicalInsight() {
  const container = document.getElementById("insight-body-container");
  const modeBadge = document.getElementById("insight-mode-badge");
  const latencyEl = document.getElementById("insight-latency");
  if (!container) return;

  const reqId = ++state.insightRequestCounter;
  container.innerHTML = `
    <div style="color: var(--text-muted); font-size: 0.84rem;">
      Synthesizing structured clinical interpretation for <strong>${escapeHtml(state.patientId)}</strong> at clock <strong>${escapeHtml(state.currentClock)}</strong>…
    </div>
  `;

  try {
    const res = await api.createInsight({
      patient_id: state.patientId,
      action: state.insightAction,
      index: state.currentIndex,
      include_context: false,
    });
    if (reqId !== state.insightRequestCounter) return;

    const c = res.insight || {};
    if (modeBadge) {
      const isGroq = res.source === "groq";
      modeBadge.textContent = isGroq ? "Groq LLM Active" : "Deterministic Clinical Engine";
      modeBadge.className = `badge ${isGroq ? "badge-good" : "badge-info"}`;
    }
    if (latencyEl) {
      latencyEl.textContent = `${fmtNum(res.latency_ms, 0)} ms`;
    }

    const renderBullets = (arr) =>
      Array.isArray(arr) && arr.length
        ? `<ul>${arr.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
        : `<p style="margin:0; font-size:0.78rem; color:var(--text-muted);">None noted.</p>`;

    container.innerHTML = `
      <div>
        <h4 class="insight-headline">${escapeHtml(c.headline || "Clinical Assessment")}</h4>
        <p class="insight-summary" style="margin-top: 6px;">${escapeHtml(c.summary || "")}</p>
        ${
          c.risk_explanation
            ? `<p style="margin: 8px 0 0; font-size: 0.82rem; color: var(--text-secondary); background: var(--bg-surface); padding: 8px 12px; border-radius: 6px; border-left: 3px solid var(--accent-teal);">
                <strong>Quantitative Attribution Summary:</strong> ${escapeHtml(c.risk_explanation)}
              </p>`
            : ""
        }
      </div>

      <div class="insight-columns">
        <div class="insight-subbox">
          <h5>Key Physiological Changes</h5>
          ${renderBullets(c.key_changes)}
        </div>
        <div class="insight-subbox">
          <h5>Clinical Attention Points</h5>
          ${renderBullets(c.clinical_attention)}
        </div>
        <div class="insight-subbox">
          <h5>Monitoring Considerations</h5>
          ${renderBullets(c.monitoring_considerations)}
        </div>
        <div class="insight-subbox">
          <h5>Data Scope &amp; Limitations</h5>
          ${renderBullets(c.data_limitations)}
        </div>
      </div>

      <div style="font-size: 0.72rem; color: var(--text-dim); border-top: 1px solid var(--border-subtle); padding-top: 8px;">
        <strong>Provenance:</strong> ${escapeHtml(c.confidence_note || res.status_message || "Derived strictly from measured twin state.")}
      </div>
    `;
  } catch (err) {
    if (reqId !== state.insightRequestCounter) return;
    container.innerHTML = `
      <div class="empty-state">
        Unable to load clinical interpretation: ${escapeHtml(err.message)}
      </div>
    `;
  }
}

// ============================================================================
// Tab 2: Live Signals & Data Fusion
// ============================================================================

async function loadTabSignals() {
  try {
    const [sensors, fusion, ehr] = await Promise.all([
      api.getPatientSensors(state.patientId, { index: state.currentIndex, hours: state.signalWindowHours }),
      api.getTwinFusion(state.patientId, { index: state.currentIndex }),
      api.getPatientEhr(state.patientId),
    ]);
    state.sensorsPayload = sensors;
    state.ehrPayload = ehr;

    renderLiveSignalCharts(sensors);
    renderFusionBlocks(fusion);
    renderEhrSummary(ehr);
  } catch (err) {
    showAlert(`Failed to load live sensor streams: ${err.message}`);
  }
}

function renderFusionBlocks(fusion) {
  const header = document.getElementById("fusion-summary-header");
  const container = document.getElementById("fusion-blocks-container");
  if (header && fusion?.fused) {
    header.textContent = fusion.fused.fusion_note || "Static EHR and continuous wearable signals aligned onto a 5-minute feature grid.";
  }
  if (!container || !fusion) return;

  const renderGroups = (section) => {
    if (!section || !Array.isArray(section.groups)) return "";
    return `
      <div class="insight-subbox">
        <h5>${escapeHtml(section.title)} — <span style="color: var(--text-muted); font-weight: 400;">${escapeHtml(section.subtitle || "")}</span></h5>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px; margin-top: 6px;">
          ${section.groups
            .map(
              (g) => `
                <div style="background: var(--bg-card); padding: 8px 10px; border-radius: 6px; border: 1px solid var(--border-subtle);">
                  <div style="font-size: 0.72rem; font-weight: 700; color: var(--accent-blue); margin-bottom: 4px;">${escapeHtml(g.label)}</div>
                  ${(g.items || [])
                    .map(
                      (item) => `
                        <div style="display: flex; justify-content: space-between; gap: 6px; font-size: 0.75rem; padding: 1px 0;">
                          <span style="color: var(--text-muted);">${escapeHtml(item.k)}:</span>
                          <strong class="tabular" style="color: var(--text-primary); text-align: right;">${escapeHtml(String(item.v))}</strong>
                        </div>
                      `
                    )
                    .join("")}
                </div>
              `
            )
            .join("")}
        </div>
      </div>
    `;
  };

  container.innerHTML = `
    ${renderGroups(fusion.historical)}
    ${renderGroups(fusion.live)}
  `;
}

function renderEhrSummary(ehrPayload) {
  const container = document.getElementById("ehr-summary-container");
  if (!container || !ehrPayload) return;
  const rec = ehrPayload.record || {};
  const demo = rec.demographics || {};
  const diab = rec.diabetes || {};
  const labs = rec.lab_results || [];
  const meds = rec.medications || [];
  const diags = rec.diagnoses || [];
  const prevCgm = rec.previous_glucose_instability || {};

  container.innerHTML = `
    <div class="metric-stat-grid">
      <div class="metric-stat-box">
        <div class="metric-stat-label">Glycaemic History</div>
        <div class="metric-stat-val tabular">TIR ${fmtNum(prevCgm.time_in_range_70_180_pct, 1)}%</div>
        <div class="metric-stat-sub">Prior TAR &gt;180: ${fmtNum(prevCgm.time_above_180_pct, 1)}% · CV: ${fmtNum(prevCgm.cv_pct, 1)}%</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">Anthropometrics</div>
        <div class="metric-stat-val tabular">BMI ${fmtNum(demo.bmi, 1)} kg/m²</div>
        <div class="metric-stat-sub">${fmtNum(demo.weight_kg, 1)} kg · Waist ${fmtNum(demo.waist_cm, 0)} cm</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">Condition Duration</div>
        <div class="metric-stat-val tabular">${fmtNum(diab.duration_years, 1)} yrs</div>
        <div class="metric-stat-sub">${escapeHtml(diab.status || "")}</div>
      </div>
    </div>

    <div class="insight-subbox">
      <h5>Recent Laboratory Panel</h5>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>Test</th>
              <th>Value</th>
              <th>Reference</th>
              <th>Date</th>
            </tr>
          </thead>
          <tbody>
            ${labs
              .slice(0, 6)
              .map(
                (l) => `
                  <tr>
                    <td><strong>${escapeHtml(l.test || l.name)}</strong></td>
                    <td class="tabular"><strong>${escapeHtml(String(l.value))}</strong> ${escapeHtml(l.unit || "")}</td>
                    <td class="mono">${escapeHtml(l.reference_range || l.reference || "—")}</td>
                    <td class="tabular">${escapeHtml(l.date || "—")}</td>
                  </tr>
                `
              )
              .join("")}
          </tbody>
        </table>
      </div>
    </div>

    <div class="insight-subbox">
      <h5>Active Medications &amp; ICD-10 Diagnoses</h5>
      <p style="margin: 0 0 6px; font-size: 0.78rem; color: var(--text-secondary);">
        <strong>Medications:</strong>
        ${
          meds.length
            ? meds.map((m) => `${escapeHtml(m.name || m)} ${escapeHtml(m.dose || "")} (${escapeHtml(m.frequency || m.schedule || "")})`).join(" · ")
            : "Lifestyle therapy"
        }
      </p>
      <p style="margin: 0; font-size: 0.76rem; color: var(--text-muted);">
        <strong>Active ICD-10 Diagnoses:</strong>
        ${diags.map((d) => `${escapeHtml(d.code)} ${escapeHtml(d.name)}`).join(" · ") || "—"}
      </p>
    </div>
  `;
}

// ============================================================================
// Tab 3: Prediction & Explainability
// ============================================================================

async function loadTabExplain() {
  try {
    const promises = [
      api.getPredictionExplain(state.patientId, { index: state.currentIndex, top_k: 12 }),
    ];
    if (!state.globalImportance) promises.push(api.getModelImportance(25));
    if (!state.featureCatalogue) promises.push(api.getModelFeatures());

    const results = await Promise.all(promises);
    const explain = results[0];
    if (results[1] && !state.globalImportance) state.globalImportance = results[1];
    if (results[2] && !state.featureCatalogue) state.featureCatalogue = results[2];
    else if (results[1] && state.globalImportance && !state.featureCatalogue) state.featureCatalogue = results[1];

    state.explainPayload = explain;
    renderExplainView(explain, state.globalImportance, state.featureCatalogue);
  } catch (err) {
    showAlert(`Failed to load explainability view: ${err.message}`);
  }
}

function renderExplainView(explain, globalImp, catalogue) {
  const attr = explain?.attribution || {};
  const methodBadge = document.getElementById("explain-method-badge");
  const baseVsPred = document.getElementById("explain-base-vs-pred");
  const narrativeEl = document.getElementById("explain-narrative-list");

  if (methodBadge) {
    methodBadge.textContent = `Method: ${attr.method || "TreeSHAP"}`;
  }
  if (baseVsPred) {
    const basePct = fmtNum(Number(attr.reference_probability ?? 0) * 100, 1);
    const predPct = fmtNum(Number(attr.risk_probability ?? explain?.probability ?? 0) * 100, 1);
    baseVsPred.textContent = `Reference Median Risk: ${basePct}% → Patient Predicted Risk: ${predPct}%`;
  }
  if (narrativeEl) {
    const top3 = (attr.top_contributors || []).slice(0, 3);
    narrativeEl.innerHTML = [
      attr.method_detail
        ? `<div class="driver-item"><strong>Attribution Method:</strong> ${escapeHtml(attr.method_detail)}</div>`
        : "",
      ...top3.map((c) => {
        const pp = +(Number(c.contribution || 0) * 100).toFixed(1);
        const dirWord = pp >= 0 ? `increases predicted 2h risk by +${pp} percentage points` : `reduces predicted 2h risk by ${pp} percentage points`;
        return `<div class="driver-item ${pp >= 0 ? "warning" : ""}"><strong>${escapeHtml(c.label)}:</strong> Current value <strong>${fmtNum(c.value, 1)} ${escapeHtml(c.unit || "")}</strong> (vs training median ${fmtNum(c.reference_value, 1)} ${escapeHtml(c.unit || "")}) ${dirWord}.</div>`;
      }),
    ].join("");
  }

  renderAttributionCharts(attr);

  const localTbody = document.getElementById("table-explain-local-body");
  if (localTbody && Array.isArray(attr.top_contributors)) {
    localTbody.innerHTML = attr.top_contributors
      .map((c) => {
        const pp = +(Number(c.contribution || 0) * 100).toFixed(2);
        const up = pp >= 0;
        return `
          <tr>
            <td>
              <strong>${escapeHtml(c.label || c.feature)}</strong>
              <div class="mono" style="font-size: 0.7rem; color: var(--text-dim);">${escapeHtml(c.feature)}</div>
            </td>
            <td><span class="badge badge-watch">${escapeHtml(c.group_label || c.group)}</span></td>
            <td class="tabular"><strong>${fmtNum(c.value, 2)}</strong> ${escapeHtml(c.unit || "")}</td>
            <td class="tabular">${fmtNum(c.reference_value, 2)} ${escapeHtml(c.unit || "")}</td>
            <td class="tabular" style="font-weight: 700; color: ${up ? "#991b1b" : "#145337"};">
              ${up ? "+" : ""}${pp} pp
            </td>
            <td>
              <span class="badge ${up ? "badge-high" : "badge-good"}">
                ${up ? "↑ Increases Risk" : "↓ Protective"}
              </span>
            </td>
          </tr>
        `;
      })
      .join("");
  }

  if (globalImp) {
    renderGlobalImportanceChart(globalImp);
    const oddsTbody = document.getElementById("table-logistic-odds-body");
    const logRows = globalImp.logistic_coefficients || globalImp.logistic_odds_ratios || [];
    if (oddsTbody && Array.isArray(logRows)) {
      oddsTbody.innerHTML = logRows
        .slice(0, 12)
        .map((row) => {
          const coef = Number(row.standardised_coefficient ?? row.coef_std ?? 0);
          const orVal = Number(row.odds_ratio ?? row.odds_ratio_per_sd ?? 1);
          const up = coef >= 0;
          return `
            <tr>
              <td><strong>${escapeHtml(row.label || row.feature)}</strong></td>
              <td>${escapeHtml(row.group_label || row.group)}</td>
              <td class="tabular">${coef >= 0 ? "+" : ""}${fmtNum(coef, 3)}</td>
              <td class="tabular"><strong>${fmtNum(orVal, 2)}×</strong></td>
              <td><span class="badge ${up ? "badge-moderate" : "badge-good"}">${up ? "Risk +" : "Protective"}</span></td>
            </tr>
          `;
        })
        .join("");
    }
  }

  if (catalogue) {
    renderFeatureCatalogueTable();
  }
}

function renderFeatureCatalogueTable() {
  const tbody = document.getElementById("table-feature-catalogue-body");
  if (!tbody || !state.featureCatalogue?.features) return;
  const groupFilter = document.getElementById("select-feature-group")?.value || "";
  const searchStr = (document.getElementById("input-feature-search")?.value || "").trim().toLowerCase();

  const filtered = state.featureCatalogue.features.filter((f) => {
    if (groupFilter && f.group !== groupFilter) return false;
    if (searchStr) {
      const hay = `${f.name} ${f.label || ""} ${f.description || ""} ${f.group_label || ""}`.toLowerCase();
      return hay.includes(searchStr);
    }
    return true;
  });

  tbody.innerHTML = filtered
    .map(
      (f, idx) => `
        <tr>
          <td class="tabular">${idx + 1}</td>
          <td class="mono" style="color: var(--accent-blue);">${escapeHtml(f.name)}</td>
          <td><strong>${escapeHtml(f.label || f.name)}</strong></td>
          <td><span class="badge badge-watch">${escapeHtml(f.group_label || f.group)}</span></td>
          <td class="mono">${escapeHtml(f.unit || "—")}</td>
          <td>${escapeHtml(f.description || "")}</td>
        </tr>
      `
    )
    .join("");
}

// ============================================================================
// Tab 4: Clinical Timeline & Day View
// ============================================================================

async function loadTabTimeline() {
  try {
    const [day, pattern, timeline] = await Promise.all([
      api.getTimelineDay(state.patientId, { index: state.currentIndex }),
      api.getTimelinePattern(state.patientId, { index: state.currentIndex, hours: 24 }),
      api.getTimeline(state.patientId, {
        index: state.currentIndex,
        hours: state.timelineHours,
        category: state.timelineCategory || null,
        severity: state.timelineSeverity || null,
      }),
    ]);
    state.dayPayload = day;
    state.patternPayload = pattern;
    state.timelinePayload = timeline;

    renderDayViewChart(day);
    renderDaySummaryMetrics(day);
    renderPatternsSection(pattern);
    renderFullTimelineFeed(timeline);
  } catch (err) {
    showAlert(`Failed to load clinical timeline: ${err.message}`);
  }
}

function renderDaySummaryMetrics(day) {
  const el = document.getElementById("day-summary-metrics");
  if (!el || !day?.summary) return;
  const s = day.summary;
  el.innerHTML = `
    <span class="clinical-chip tabular">Mean: <strong>${fmtNum(s.glucose_mean, 0)} mg/dL</strong></span>
    <span class="clinical-chip tabular">Peak: <strong>${fmtNum(s.glucose_max, 0)} mg/dL</strong></span>
    <span class="clinical-chip tabular">Min: <strong>${fmtNum(s.glucose_min, 0)} mg/dL</strong></span>
    <span class="clinical-chip tabular">TAR (&gt;180): <strong>${fmtNum(s.time_above_180_pct, 1)}%</strong></span>
    <span class="clinical-chip tabular">Peak Risk: <strong>${fmtNum(Number(s.risk_max || 0) * 100, 0)}%</strong></span>
    <span class="clinical-chip tabular">Steps: <strong>${fmtNum(s.steps, 0)}</strong></span>
  `;
}

function renderPatternsSection(pattern) {
  const countersEl = document.getElementById("pattern-counters-strip");
  const gridEl = document.getElementById("pattern-cards-grid");
  if (countersEl && pattern?.glucose) {
    const g = pattern.glucose;
    countersEl.innerHTML = `
      <span class="clinical-chip tabular">24h TIR: <strong>${fmtNum(g.time_in_range_pct, 1)}%</strong></span>
      <span class="clinical-chip tabular">24h TAR (&gt;180): <strong>${fmtNum(g.time_above_180_pct, 1)}%</strong></span>
      <span class="clinical-chip tabular">24h CV: <strong>${fmtNum(g.cv_pct, 1)}%</strong></span>
      <span class="clinical-chip tabular">Excursions &gt;180: <strong>${fmtNum(g.excursions_above_180, 0)}</strong></span>
    `;
  }
  if (gridEl && pattern) {
    const g = pattern.glucose || {};
    const r = pattern.risk || {};
    const a = pattern.activity || {};
    const c = pattern.cardio || {};
    const cards = [
      {
        title: "24-Hour Glycaemic Excursion Profile",
        severity: g.excursions_above_180 > 0 ? "warning" : "good",
        detail: `Mean glucose ${fmtNum(g.mean_mgdl, 0)} mg/dL (range ${fmtNum(g.min_mgdl, 0)}–${fmtNum(g.max_mgdl, 0)} mg/dL), with ${fmtNum(g.time_in_range_pct, 1)}% Time-in-Range and ${fmtNum(g.excursions_above_180, 0)} hyperglycaemic excursion(s) ≥180 mg/dL.`,
      },
      {
        title: "2-Hour Spike Risk Trajectory Summary",
        severity: r.max >= 0.7 ? "critical" : r.max >= 0.35 ? "warning" : "good",
        detail: `Mean 2h spike probability ${fmtNum(Number(r.mean || 0) * 100, 1)}% (peak ${fmtNum(Number(r.max || 0) * 100, 1)}%). Patient spent ${fmtNum(r.share_of_time_high_pct, 1)}% of the last 24h in the HIGH risk band.`,
      },
      {
        title: "Autonomic & Physical Activity Balance",
        severity: a.longest_sedentary_minutes > 180 ? "watch" : "good",
        detail: `Accumulated ${fmtNum(a.steps, 0)} steps (mean ${fmtNum(a.mean_met, 2)} MET, longest sedentary bout ${fmtNum(a.longest_sedentary_minutes, 0)} min). Mean HR ${fmtNum(c.mean_hr, 0)} bpm, mean HRV ${fmtNum(c.mean_hrv_ms, 0)} ms.`,
      },
    ];
    gridEl.innerHTML = cards
      .map((p) => {
        const badgeCls = severityBadgeClass(p.severity);
        return `
          <div class="panel" style="background: var(--bg-card); padding: 12px 14px;">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 8px; margin-bottom: 6px;">
              <h4 style="margin: 0; font-size: 0.88rem; color: var(--text-primary);">${escapeHtml(p.title)}</h4>
              <span class="badge ${badgeCls}">${escapeHtml(p.severity)}</span>
            </div>
            <p style="margin: 0; font-size: 0.78rem; color: var(--text-secondary);">${escapeHtml(p.detail)}</p>
          </div>
        `;
      })
      .join("");
  }
}

function renderFullTimelineFeed(timelinePayload) {
  const feedEl = document.getElementById("timeline-full-feed");
  if (!feedEl) return;
  const events = timelinePayload?.events || [];
  if (events.length === 0) {
    feedEl.innerHTML = `<div class="empty-state">No clinical events match the current filter criteria.</div>`;
    return;
  }
  feedEl.innerHTML = events.map(renderTimelineItemHtml).join("");
}

// ============================================================================
// Tab 5: Model Validation, Provenance & Architecture
// ============================================================================

async function loadTabValidation() {
  if (state.validationLoaded) return;
  try {
    const [metrics, calibration, dataset, limitations, arch, profile] = await Promise.all([
      api.getModelMetrics(),
      api.getModelCalibration(),
      api.getModelDataset(),
      api.getModelLimitations(),
      api.getArchitecture(),
      api.getProfile(),
    ]);

    renderValidationSection(metrics, calibration, dataset, limitations, arch, profile);
    state.validationLoaded = true;
  } catch (err) {
    showAlert(`Failed to load model validation artifacts: ${err.message}`);
  }
}

function renderValidationSection(metrics, calibration, dataset, limitations, arch, profile) {
  const testSplit = metrics?.splits?.test || {};
  const splitStrat = metrics?.split_strategy || {};
  const subtitleEl = document.getElementById("validation-split-subtitle");
  if (subtitleEl && splitStrat.test_patients) {
    subtitleEl.textContent = `Evaluated on ${splitStrat.test_patients} unseen test patients (${splitStrat.test_rows} 5-minute windows) — zero patient overlap with ${splitStrat.train_patients} train / ${splitStrat.validation_patients} validation patients`;
  }

  const kpiGrid = document.getElementById("validation-kpi-grid");
  if (kpiGrid) {
    kpiGrid.innerHTML = `
      <div class="metric-stat-box">
        <div class="metric-stat-label">Held-Out Test AUROC</div>
        <div class="metric-stat-val tabular" style="color: var(--accent-teal);">${fmtNum(testSplit.auroc, 4)}</div>
        <div class="metric-stat-sub">n = ${fmtNum(testSplit.n_samples, 0)} windows</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">Held-Out Test AUPRC</div>
        <div class="metric-stat-val tabular" style="color: var(--accent-blue);">${fmtNum(testSplit.auprc, 4)}</div>
        <div class="metric-stat-sub">Positive event rate: ${fmtNum(Number(testSplit.positive_rate || 0) * 100, 1)}%</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">Brier Score / ECE</div>
        <div class="metric-stat-val tabular">${fmtNum(testSplit.brier_score ?? testSplit.brier, 4)}</div>
        <div class="metric-stat-sub">Calibration ECE: ${fmtNum(testSplit.calibration?.ece, 4)}</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">Sensitivity / Specificity</div>
        <div class="metric-stat-val tabular">${fmtNum(Number(testSplit.sensitivity || 0) * 100, 1)}% / ${fmtNum(Number(testSplit.specificity || 0) * 100, 1)}%</div>
        <div class="metric-stat-sub">At operating threshold ${fmtNum(testSplit.decision_threshold ?? metrics?.threshold, 2)}</div>
      </div>
      <div class="metric-stat-box">
        <div class="metric-stat-label">F1 / Precision (PPV)</div>
        <div class="metric-stat-val tabular">${fmtNum(testSplit.f1, 3)} / ${fmtNum(Number(testSplit.precision || 0) * 100, 1)}%</div>
        <div class="metric-stat-sub">NPV: ${fmtNum(Number(testSplit.npv || 0) * 100, 1)}%</div>
      </div>
    `;
  }

  renderValidationCharts(metrics, calibration);

  // Model comparison table
  const compTbody = document.getElementById("table-model-comparison-body");
  if (compTbody && metrics?.splits) {
    const byModel = metrics.splits.test_by_model || {};
    const rows = [
      {
        name: `Primary: ${metrics.estimator || "XGBoost Hist GBM"} (Test Split)`,
        m: byModel.gradient_boosting || testSplit,
        highlight: true,
      },
      {
        name: `Secondary: ${metrics.secondary_estimator || "Standardised L2 Logistic Regression"} (Test Split)`,
        m: byModel.logistic_regression || {},
      },
      {
        name: "Primary GBM (Validation Split)",
        m: metrics.splits.validation || {},
      },
      {
        name: "Primary GBM (Training Split)",
        m: metrics.splits.train || {},
      },
    ];
    compTbody.innerHTML = rows
      .map(
        (r) => `
          <tr style="${r.highlight ? "background: rgba(184, 74, 39, 0.10);" : ""}">
            <td><strong>${escapeHtml(r.name)}</strong></td>
            <td class="tabular"><strong>${fmtNum(r.m.auroc, 4)}</strong></td>
            <td class="tabular">${fmtNum(r.m.auprc, 4)}</td>
            <td class="tabular">${fmtNum(r.m.brier_score ?? r.m.brier, 4)}</td>
            <td class="tabular">${fmtNum(r.m.f1, 3)}</td>
            <td class="tabular">${fmtNum(Number(r.m.sensitivity || 0) * 100, 1)}%</td>
            <td class="tabular">${fmtNum(Number(r.m.specificity || 0) * 100, 1)}%</td>
          </tr>
        `
      )
      .join("");
  }

  // Calibration Reliability by Risk Band table (in #table-subgroup-body)
  const subTbody = document.getElementById("table-subgroup-body");
  if (subTbody && Array.isArray(calibration?.reliability_by_risk_band)) {
    subTbody.innerHTML = calibration.reliability_by_risk_band
      .map(
        (b) => `
          <tr>
            <td><strong>${escapeHtml((b.band || "").toUpperCase())} Risk Band (${fmtNum((b.range?.[0] || 0) * 100, 0)}–${fmtNum((b.range?.[1] || 1) * 100, 0)}%)</strong></td>
            <td class="tabular">${fmtNum(b.count ?? b.n, 0)}</td>
            <td class="tabular">${fmtNum(Number(b.observed_rate ?? b.event_rate ?? 0) * 100, 1)}%</td>
            <td class="tabular"><strong>${fmtNum(Number(b.mean_predicted ?? 0) * 100, 1)}% (Mean Pred)</strong></td>
            <td class="tabular">${fmtNum(Number(b.share_of_samples ?? 0) * 100, 1)}% share</td>
            <td class="tabular">${fmtNum(Math.abs(Number(b.mean_predicted || 0) - Number(b.observed_rate || 0)), 4)} gap</td>
          </tr>
        `
      )
      .join("");
  }

  // Dataset provenance
  const dsBox = document.getElementById("dataset-provenance-box");
  if (dsBox && dataset) {
    const meta = dataset.metadata || {};
    const sp = dataset.split || {};
    dsBox.innerHTML = `
      <div>
        <strong>Dataset Classification:</strong> <span class="mono">${escapeHtml(dataset.kind || "Synthetic validation dataset")}</span>
        · <strong>${fmtNum(meta.n_patients, 0)} synthetic patients</strong> (${fmtNum(meta.rows, 0)} scored 5-min windows)
      </div>
      <div>
        <strong>Patient-Wise Splits:</strong>
        Train: ${fmtNum(sp.train_patients, 0)} patients (${fmtNum(sp.train_rows, 0)} rows) ·
        Val: ${fmtNum(sp.validation_patients, 0)} patients (${fmtNum(sp.validation_rows, 0)} rows) ·
        Test: ${fmtNum(sp.test_patients, 0)} patients (${fmtNum(sp.test_rows, 0)} rows)
      </div>
      <div class="insight-subbox">
        <h5>Deterministic Regeneration &amp; Provenance</h5>
        <p style="margin: 0; font-size: 0.78rem;">
          Generator: <code class="mono">${escapeHtml(meta.generator || "datasets/synthetic/generate_cohort.py")}</code> (Seed: <code class="mono">${escapeHtml(String(meta.cohort_seed || 20260401))}</code>, SHA-256: <code class="mono">${escapeHtml(meta.sha256_16 || "")}</code>).
          Positive event rate: <strong>${fmtNum(Number(meta.positive_rate || 0) * 100, 2)}%</strong> across ${fmtNum(meta.features || 84, 0)} features.
        </p>
      </div>
    `;
  }

  // Limitations
  const limBox = document.getElementById("limitations-list-container");
  if (limBox && Array.isArray(limitations?.categories)) {
    limBox.innerHTML = limitations.categories
      .map(
        (cat) => `
          <div class="driver-item warning">
            <strong>${escapeHtml(cat.title)}:</strong>
            ${escapeHtml((cat.items || [])[0] || "")}
          </div>
        `
      )
      .join("");
  }

  // Architecture layers
  const archBox = document.getElementById("architecture-layers-container");
  if (archBox && Array.isArray(arch?.layers)) {
    archBox.innerHTML = arch.layers
      .map(
        (layer) => `
          <div class="arch-layer-card">
            <h4 class="arch-layer-title">${escapeHtml(layer.label || layer.title)}</h4>
            <div class="arch-nodes-grid">
              ${(layer.nodes || layer.items || [])
                .map(
                  (node) => `
                    <div class="arch-node">
                      <strong>${escapeHtml(node.label || node.name)}</strong>
                      <span>${escapeHtml(node.detail || "")}</span>
                    </div>
                  `
                )
                .join("")}
            </div>
          </div>
        `
      )
      .join("");
  }

  // Profile Attribution
  const profBox = document.getElementById("profile-attribution-box");
  if (profBox && profile) {
    const prod = profile.product || {};
    const team = profile.team || {};
    const contact = profile.contact || {};
    profBox.innerHTML = `
      <div class="panel-header" style="margin-bottom: 6px;">
        <div>
          <div class="section-kicker">Project &amp; Platform Metadata</div>
          <h3 class="panel-title">${escapeHtml(prod.name || "VitalSync")} — ${escapeHtml(prod.tagline || "")}</h3>
          <p class="panel-subtitle">${escapeHtml(prod.clinical_question || "")}</p>
        </div>
        <div style="display: flex; gap: 8px; flex-wrap: wrap;">
          ${
            contact.repository && !String(contact.repository).includes("PLACEHOLDER")
              ? `<a href="${escapeHtml(contact.repository)}" target="_blank" rel="noopener noreferrer" class="btn btn-sm">GitHub Repository</a>`
              : `<a href="https://github.com/aminotkartik/digitaltwin" target="_blank" rel="noopener noreferrer" class="btn btn-sm">GitHub Repository</a>`
          }
        </div>
      </div>
      <p style="margin: 0; font-size: 0.82rem; color: var(--text-secondary);">
        ${escapeHtml(prod.problem_statement || "")}
      </p>
    `;
  }
}

// ============================================================================
// Digital Twin Domain Drill-Down Modal
// ============================================================================

async function openDomainModal(domainKey) {
  const backdrop = document.getElementById("domain-modal-backdrop");
  const titleEl = document.getElementById("domain-modal-title");
  const bodyEl = document.getElementById("domain-modal-body");
  if (!backdrop || !bodyEl) return;

  backdrop.hidden = false;
  titleEl.textContent = "Loading Domain Assessment…";
  bodyEl.innerHTML = `<div style="color: var(--text-muted);">Fetching domain sub-terms and deterministic formula…</div>`;

  try {
    const res = await api.getTwinDomainDetail(state.patientId, domainKey, { index: state.currentIndex });
    titleEl.textContent = `${res.label || domainKey} — Score ${fmtNum(res.score, 1)}/100 (${(res.word || res.status || "").toUpperCase()})`;

    const contribs = (res.contributors || [])
      .map(
        (c) => `
          <tr>
            <td><strong>${escapeHtml(c.term)}</strong></td>
            <td class="tabular">${fmtNum(c.input, 1)} ${escapeHtml(c.unit || "")}</td>
            <td class="tabular"><strong>${fmtNum(c.score, 1)} / 100</strong></td>
            <td class="tabular">${Math.round(Number(c.weight || 0) * 100)}%</td>
          </tr>
        `
      )
      .join("");

    bodyEl.innerHTML = `
      <div class="driver-item">
        <strong>Deterministic Scoring Formula:</strong> <span class="mono">${escapeHtml(res.formula || "")}</span>
      </div>
      <div class="table-scroll">
        <table class="data-table">
          <thead>
            <tr>
              <th>Sub-Term</th>
              <th>Measured Input</th>
              <th>Sub-Score</th>
              <th>Domain Weight</th>
            </tr>
          </thead>
          <tbody>
            ${contribs}
          </tbody>
        </table>
      </div>
      <p style="margin: 0; font-size: 0.76rem; color: var(--text-muted);">${escapeHtml(res.note || "")}</p>
    `;
  } catch (err) {
    bodyEl.innerHTML = `<div class="empty-state">Could not load domain detail: ${escapeHtml(err.message)}</div>`;
  }
}

function closeDomainModal() {
  const backdrop = document.getElementById("domain-modal-backdrop");
  if (backdrop) backdrop.hidden = true;
}

// ============================================================================
// Simulation & Replay Controller Logic
// ============================================================================

function updatePlayButtonUi() {
  const btn = document.getElementById("btn-sim-play");
  const label = document.getElementById("btn-sim-play-label");
  if (!btn || !label) return;
  if (state.sessionRunning) {
    label.textContent = "⏸ Pause Auto-Replay";
    btn.classList.add("btn-warn");
  } else {
    label.textContent = "▶ Start Auto-Replay";
    btn.classList.remove("btn-warn");
  }
}

function stopAutoTickTimer() {
  if (state.tickTimer) {
    clearInterval(state.tickTimer);
    state.tickTimer = null;
  }
  state.sessionRunning = false;
  updatePlayButtonUi();
}

function startAutoTickTimer() {
  stopAutoTickTimer();
  state.sessionRunning = true;
  updatePlayButtonUi();
  state.tickTimer = setInterval(async () => {
    if (!state.sessionRunning || state.simRequestInFlight) return;
    await stepSimulationOnce();
  }, state.tickIntervalMs);
}

async function ensureSimulationSession(startRunning = false) {
  const res = await api.startSimulation({
    patient_id: state.patientId,
    from_index: state.currentIndex,
    from_clock: state.currentIndex === null ? state.currentClock : null,
    speed_minutes_per_tick: state.simSpeedMinutes,
    tick_interval_ms: state.tickIntervalMs,
    session_id: state.sessionId,
  });
  state.sessionId = res.session?.session_id || state.sessionId;
  if (!startRunning && state.sessionId) {
    await api.pauseSimulation(state.sessionId);
  }
  return res;
}

async function stepSimulationOnce() {
  if (state.simRequestInFlight) return;
  state.simRequestInFlight = true;
  try {
    if (!state.sessionId) {
      const started = await ensureSimulationSession(true);
      renderDashboardSnapshot(started.dashboard, started.session);
      refreshActiveSecondaryTab();
      return;
    }
    let res;
    try {
      res = await api.stepSimulation({
        session_id: state.sessionId,
        minutes: state.simSpeedMinutes,
      });
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        const started = await ensureSimulationSession(true);
        res = await api.stepSimulation({
          session_id: started.session.session_id,
          minutes: state.simSpeedMinutes,
        });
      } else {
        throw err;
      }
    }

    if (res?.dashboard) {
      renderDashboardSnapshot(res.dashboard, res.session);
      refreshActiveSecondaryTab();
    }
    if (res?.session?.at_stream_end || res?.session?.at_end) {
      stopAutoTickTimer();
      refreshClinicalInsight();
    }
  } catch (err) {
    stopAutoTickTimer();
    showAlert(`Simulation replay paused: ${err.message}`);
  } finally {
    state.simRequestInFlight = false;
  }
}

async function toggleSimulationPlay() {
  hideAlert();
  if (state.sessionRunning) {
    stopAutoTickTimer();
    if (state.sessionId) {
      try {
        await api.pauseSimulation(state.sessionId);
      } catch (_e) {
        // ignore pause errors
      }
    }
    refreshClinicalInsight();
    return;
  }

  try {
    const res = await ensureSimulationSession(true);
    if (res?.dashboard) {
      renderDashboardSnapshot(res.dashboard, res.session);
    }
    startAutoTickTimer();
  } catch (err) {
    showAlert(`Could not start simulation: ${err.message}`);
  }
}

async function jumpSimulationMinutes(minutes) {
  hideAlert();
  stopAutoTickTimer();
  if (state.simRequestInFlight) return;
  state.simRequestInFlight = true;
  try {
    if (!state.sessionId) {
      await ensureSimulationSession(false);
    }
    let res;
    try {
      res = await api.jumpSimulation(state.sessionId, minutes);
    } catch (err) {
      if (err instanceof ApiError && err.status === 404) {
        await ensureSimulationSession(false);
        res = await api.jumpSimulation(state.sessionId, minutes);
      } else {
        throw err;
      }
    }
    if (res?.dashboard) {
      renderDashboardSnapshot(res.dashboard, res.session);
      refreshActiveSecondaryTab();
      refreshClinicalInsight();
    }
  } catch (err) {
    showAlert(`Could not step timeline: ${err.message}`);
  } finally {
    state.simRequestInFlight = false;
  }
}

async function resetSimulationToStart() {
  hideAlert();
  stopAutoTickTimer();
  try {
    const res = await api.startSimulation({
      patient_id: state.patientId,
      from_clock: "09:30",
      speed_minutes_per_tick: state.simSpeedMinutes,
      tick_interval_ms: state.tickIntervalMs,
      session_id: state.sessionId,
    });
    state.sessionId = res.session?.session_id || state.sessionId;
    if (state.sessionId) {
      await api.pauseSimulation(state.sessionId);
    }
    if (res?.dashboard) {
      renderDashboardSnapshot(res.dashboard, res.session);
      refreshActiveSecondaryTab();
      refreshClinicalInsight();
    }
  } catch (err) {
    showAlert(`Could not reset simulation: ${err.message}`);
  }
}

async function jumpToClockPreset(clockStr) {
  hideAlert();
  stopAutoTickTimer();
  try {
    const snap = await api.getSnapshot({ patient_id: state.patientId, at: clockStr });
    state.sessionId = null;
    renderDashboardSnapshot(snap.dashboard, snap.session);
    refreshActiveSecondaryTab();
    refreshClinicalInsight();
  } catch (err) {
    showAlert(`Could not jump to ${clockStr}: ${err.message}`);
  }
}

// ============================================================================
// Patient Switching & Tab Navigation
// ============================================================================

async function switchPatient(patientId) {
  hideAlert();
  stopAutoTickTimer();
  state.patientId = patientId;
  state.sessionId = null;
  state.currentIndex = null;
  state.currentClock = "10:45";

  try {
    const snap = await api.getSnapshot({ patient_id: patientId, at: state.currentClock });
    renderDashboardSnapshot(snap.dashboard, snap.session);
    refreshActiveSecondaryTab();
    refreshClinicalInsight();
  } catch (err) {
    showAlert(`Failed to switch patient to ${patientId}: ${err.message}`);
  }
}

function switchTab(tabKey) {
  state.activeTab = tabKey;
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    const isTarget = btn.getAttribute("data-tab") === tabKey;
    btn.setAttribute("aria-selected", isTarget ? "true" : "false");
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.hidden = panel.id !== `panel-${tabKey}`;
  });
  refreshActiveSecondaryTab();
}

function refreshActiveSecondaryTab() {
  if (state.activeTab === "signals") {
    loadTabSignals();
  } else if (state.activeTab === "explain") {
    loadTabExplain();
  } else if (state.activeTab === "timeline") {
    loadTabTimeline();
  } else if (state.activeTab === "validation") {
    loadTabValidation();
  }
}

// ============================================================================
// Event Listeners & Application Bootstrap
// ============================================================================

function bindEventListeners() {
  document.getElementById("btn-dismiss-alert")?.addEventListener("click", hideAlert);

  // Navigation tabs
  document.querySelectorAll(".nav-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.getAttribute("data-tab");
      if (tab) switchTab(tab);
    });
  });

  // Quick jump buttons
  document.getElementById("btn-jump-explain")?.addEventListener("click", () => switchTab("explain"));
  document.getElementById("btn-view-full-timeline")?.addEventListener("click", () => switchTab("timeline"));

  // Simulation transport buttons
  document.getElementById("btn-sim-play")?.addEventListener("click", toggleSimulationPlay);
  document.getElementById("btn-sim-back-15")?.addEventListener("click", () => jumpSimulationMinutes(-15));
  document.getElementById("btn-sim-fwd-15")?.addEventListener("click", () => jumpSimulationMinutes(15));
  document.getElementById("btn-sim-fwd-30")?.addEventListener("click", () => jumpSimulationMinutes(30));
  document.getElementById("btn-sim-fwd-60")?.addEventListener("click", () => jumpSimulationMinutes(60));
  document.getElementById("btn-sim-reset")?.addEventListener("click", resetSimulationToStart);

  document.getElementById("select-sim-speed")?.addEventListener("change", async (e) => {
    state.simSpeedMinutes = Number(e.target.value) || 15;
    if (state.sessionId) {
      try {
        await api.setSimulationSpeed(state.sessionId, state.simSpeedMinutes, state.tickIntervalMs);
      } catch (_err) {
        // ignore if session expired
      }
    }
  });

  document.querySelectorAll(".js-clock-preset").forEach((btn) => {
    btn.addEventListener("click", () => {
      const clock = btn.getAttribute("data-clock");
      if (clock) jumpToClockPreset(clock);
    });
  });

  // Insight action pills
  document.querySelectorAll(".js-insight-focus").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".js-insight-focus").forEach((b) => b.classList.remove("active-pill"));
      btn.classList.add("active-pill");
      state.insightAction = btn.getAttribute("data-focus") || "summarize_changes";
      refreshClinicalInsight();
    });
  });

  // Tab 2: Live signal window selector
  document.getElementById("select-signal-hours")?.addEventListener("change", (e) => {
    state.signalWindowHours = Number(e.target.value) || 6;
    loadTabSignals();
  });

  // Tab 3: Feature catalogue filters
  document.getElementById("select-feature-group")?.addEventListener("change", renderFeatureCatalogueTable);
  document.getElementById("input-feature-search")?.addEventListener("input", renderFeatureCatalogueTable);

  // Tab 4: Timeline filters
  document.getElementById("select-timeline-hours")?.addEventListener("change", (e) => {
    state.timelineHours = Number(e.target.value) || 24;
    loadTabTimeline();
  });
  document.getElementById("select-timeline-category")?.addEventListener("change", (e) => {
    state.timelineCategory = e.target.value || "";
    loadTabTimeline();
  });
  document.getElementById("select-timeline-severity")?.addEventListener("change", (e) => {
    state.timelineSeverity = e.target.value || "";
    loadTabTimeline();
  });

  // Modal close handlers
  document.getElementById("btn-close-domain-modal")?.addEventListener("click", closeDomainModal);
  document.getElementById("domain-modal-backdrop")?.addEventListener("click", (e) => {
    if (e.target.id === "domain-modal-backdrop") closeDomainModal();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDomainModal();
  });
}

async function initApp() {
  bindEventListeners();

  try {
    const [health, config, patientsRes] = await Promise.all([
      api.getHealth(),
      api.getConfig(),
      api.getPatients(),
    ]);

    renderSystemHeader(health, config);
    const defaultId = patientsRes?.demo_patient_id || config?.demo?.patient_id || "DT-1047";
    state.patientId = defaultId;
    renderPatientRoster(patientsRes?.patients || []);

    const snap = await api.getSnapshot({
      patient_id: state.patientId,
      at: config?.demo?.now_clock || "10:45",
    });
    renderDashboardSnapshot(snap.dashboard, snap.session);
    refreshClinicalInsight();
  } catch (err) {
    showAlert(`Initialisation error: ${err.message}`);
  }
}

document.addEventListener("DOMContentLoaded", initApp);
