/**
 * VitalSync — Chart.js clinical visualizations (Warm High-Contrast Palette).
 *
 * Uses the locally vendored Chart.js 4.4.7 UMD bundle (`window.Chart`).
 * Every chart instance is tracked by canvas ID and cleanly destroyed/updated
 * so rapid simulation ticks or patient switches never leak memory.
 */

const chartRegistry = new Map();

function getChartLib() {
  return typeof window !== "undefined" ? window.Chart : null;
}

export function destroyChart(canvasId) {
  const existing = chartRegistry.get(canvasId);
  if (existing) {
    try {
      existing.destroy();
    } catch (_e) {
      // ignore destroy errors on detached nodes
    }
    chartRegistry.delete(canvasId);
  }
}

function createOrReplaceChart(canvasId, config) {
  const ChartLib = getChartLib();
  if (!ChartLib) return null;
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  destroyChart(canvasId);
  const ctx = canvas.getContext("2d");
  const instance = new ChartLib(ctx, config);
  chartRegistry.set(canvasId, instance);
  return instance;
}

// Warm umber grid & high-contrast espresso/umber tick labels
const GRID_COLOR = "rgba(97, 78, 68, 0.14)";
const TICK_COLOR = "#4e3c33";
const LABEL_STRONG = "#241914";
const TOOLTIP_STYLE = {
  backgroundColor: "rgba(31, 22, 18, 0.96)",
  titleColor: "#faf6f0",
  bodyColor: "#ede3d7",
  borderColor: "rgba(217, 119, 6, 0.55)",
  borderWidth: 1,
  padding: 10,
  displayColors: true,
};

function fmtClock(isoOrClock) {
  if (!isoOrClock) return "";
  const str = String(isoOrClock);
  if (/^\d{2}:\d{2}/.test(str)) return str.slice(0, 5);
  try {
    const d = new Date(str);
    if (!Number.isNaN(d.getTime())) {
      const hh = String(d.getHours()).padStart(2, "0");
      const mm = String(d.getMinutes()).padStart(2, "0");
      return `${hh}:${mm}`;
    }
  } catch (_e) {
    // fallback below
  }
  const match = str.match(/T(\d{2}:\d{2})/);
  return match ? match[1] : str;
}

// Custom plugin to draw the vertical "NOW" divider between historical and forecast regions
const nowDividerPlugin = {
  id: "vitalsyncNowDivider",
  afterDraw(chart, _args, options) {
    const nowIndex = options?.nowIndex;
    if (nowIndex === undefined || nowIndex === null || nowIndex < 0) return;
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    if (!xScale || !yScale) return;
    const x = xScale.getPixelForValue(nowIndex);
    if (Number.isNaN(x) || x < xScale.left || x > xScale.right) return;

    const ctx = chart.ctx;
    ctx.save();

    // Shade forecast region with warm amber tint
    ctx.fillStyle = "rgba(217, 119, 6, 0.07)";
    ctx.fillRect(x, yScale.top, xScale.right - x, yScale.bottom - yScale.top);

    // Draw vertical NOW line
    ctx.beginPath();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "rgba(154, 52, 18, 0.85)";
    ctx.lineWidth = 1.6;
    ctx.moveTo(x, yScale.top);
    ctx.lineTo(x, yScale.bottom);
    ctx.stroke();

    // High-contrast label pill
    const label = options?.label || "NOW";
    ctx.font = "700 10px -apple-system, BlinkMacSystemFont, sans-serif";
    const textWidth = ctx.measureText(label).width;
    const pillW = textWidth + 12;
    const pillH = 17;
    const pillX = Math.min(Math.max(x - pillW / 2, xScale.left + 2), xScale.right - pillW - 2);
    const pillY = yScale.top + 4;

    ctx.fillStyle = "#261913";
    ctx.strokeStyle = "#d97706";
    ctx.setLineDash([]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(pillX, pillY, pillW, pillH, 4);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#fde68a";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(label, pillX + pillW / 2, pillY + pillH / 2);

    ctx.restore();
  },
};

/**
 * 1. Main Glucose History + 2-Hour Forecast Trajectory Chart
 */
export function renderMainGlucoseChart(canvasId, chartData) {
  if (!chartData) return null;
  const past = chartData.past || {};
  const forecast = chartData.forecast || {};
  const bands = chartData.bands || {};

  const pastClocks = past.clocks && past.clocks.length ? past.clocks : (past.timestamps || []).map(fmtClock);
  const pastGlucose = past.glucose_mgdl || [];

  const forecastPoints = forecast.points || [];
  const trendPoints =
    forecast.trend_continuation?.points ||
    (Array.isArray(forecast.trend_continuation) ? forecast.trend_continuation : []);

  const forecastLabels = forecastPoints.map((p) => fmtClock(p.timestamp) || `+${p.minutes_ahead}m`);
  const labels = [...pastClocks, ...forecastLabels];
  const nowIdx = Math.max(0, pastGlucose.length - 1);
  const currentGlucose = pastGlucose.length ? pastGlucose[nowIdx] : forecast.anchor_glucose_mgdl ?? null;

  const historySeries = [...pastGlucose, ...new Array(forecastPoints.length).fill(null)];

  const forecastP50 = new Array(pastGlucose.length).fill(null);
  const forecastP90 = new Array(pastGlucose.length).fill(null);
  const forecastP10 = new Array(pastGlucose.length).fill(null);
  if (pastGlucose.length > 0 && currentGlucose !== null) {
    forecastP50[nowIdx] = currentGlucose;
    forecastP90[nowIdx] = currentGlucose;
    forecastP10[nowIdx] = currentGlucose;
  }
  forecastPoints.forEach((pt) => {
    forecastP50.push(pt.p50 ?? null);
    forecastP90.push(pt.p90 ?? null);
    forecastP10.push(pt.p10 ?? null);
  });

  const trendMap = new Map();
  trendPoints.forEach((tp) => {
    trendMap.set(Number(tp.minutes_ahead), tp.p50);
  });
  const trendSeries = new Array(pastGlucose.length).fill(null);
  if (pastGlucose.length > 0 && currentGlucose !== null) {
    trendSeries[nowIdx] = currentGlucose;
  }
  forecastPoints.forEach((pt) => {
    const val = trendMap.get(Number(pt.minutes_ahead));
    trendSeries.push(val !== undefined ? val : null);
  });

  const highThreshold = Number(bands.glucose_high_mgdl ?? bands.threshold_mgdl ?? 180);
  const lowThreshold = Number(bands.glucose_low_mgdl ?? bands.low_threshold_mgdl ?? 70);
  const personalMedian = bands.personal_median ? Number(bands.personal_median) : null;

  const highLine = new Array(labels.length).fill(highThreshold);
  const lowLine = new Array(labels.length).fill(lowThreshold);
  const medianLine = personalMedian !== null ? new Array(labels.length).fill(personalMedian) : null;

  const datasets = [
    {
      label: "Historical Glucose (CGM)",
      data: historySeries,
      borderColor: "#b84a27",
      backgroundColor: "rgba(184, 74, 39, 0.12)",
      borderWidth: 2.5,
      pointRadius: 0,
      pointHoverRadius: 4,
      tension: 0.25,
      spanGaps: false,
      order: 1,
    },
    {
      label: "Forecast Median (p50)",
      data: forecastP50,
      borderColor: "#d97706",
      backgroundColor: "#d97706",
      borderWidth: 2.3,
      borderDash: [6, 4],
      pointRadius: 2.5,
      pointHoverRadius: 4,
      tension: 0.25,
      spanGaps: true,
      order: 2,
    },
    {
      label: "Forecast Upper Band (p90)",
      data: forecastP90,
      borderColor: "rgba(180, 83, 9, 0.42)",
      backgroundColor: "rgba(217, 119, 6, 0.20)",
      borderWidth: 1,
      pointRadius: 0,
      fill: "+1",
      tension: 0.25,
      spanGaps: true,
      order: 4,
    },
    {
      label: "Forecast Lower Band (p10)",
      data: forecastP10,
      borderColor: "rgba(180, 83, 9, 0.42)",
      backgroundColor: "transparent",
      borderWidth: 1,
      pointRadius: 0,
      fill: false,
      tension: 0.25,
      spanGaps: true,
      order: 5,
    },
    {
      label: "Personal Trend Extrapolation",
      data: trendSeries,
      borderColor: "#7c2d12",
      borderWidth: 1.7,
      borderDash: [2, 3],
      pointRadius: 0,
      tension: 0.25,
      spanGaps: true,
      order: 3,
    },
    {
      label: `Hyperglycaemic Threshold (${highThreshold} mg/dL)`,
      data: highLine,
      borderColor: "rgba(185, 28, 28, 0.78)",
      borderWidth: 1.4,
      borderDash: [5, 5],
      pointRadius: 0,
      fill: false,
      order: 6,
    },
    {
      label: `Hypoglycaemic Threshold (${lowThreshold} mg/dL)`,
      data: lowLine,
      borderColor: "rgba(124, 45, 18, 0.55)",
      borderWidth: 1.1,
      borderDash: [3, 5],
      pointRadius: 0,
      fill: false,
      order: 7,
    },
  ];

  if (medianLine) {
    datasets.push({
      label: `Personal Median (${personalMedian} mg/dL)`,
      data: medianLine,
      borderColor: "rgba(97, 78, 68, 0.55)",
      borderWidth: 1.1,
      borderDash: [2, 4],
      pointRadius: 0,
      fill: false,
      order: 8,
    });
  }

  return createOrReplaceChart(canvasId, {
    type: "line",
    data: { labels, datasets },
    plugins: [nowDividerPlugin],
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label(ctx) {
              if (ctx.parsed.y === null || ctx.parsed.y === undefined) return null;
              return `${ctx.dataset.label}: ${Math.round(ctx.parsed.y)} mg/dL`;
            },
          },
        },
        vitalsyncNowDivider: {
          nowIndex: nowIdx,
          label: `NOW (${pastClocks[nowIdx] || ""})`,
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, maxTicksLimit: 12, font: { size: 11, weight: "600" } },
          title: {
            display: true,
            text: "Scenario Clock Time (Historical → 2h Forecast Horizon)",
            color: TICK_COLOR,
            font: { size: 11, weight: "600" },
          },
        },
        y: {
          min: 50,
          suggestedMax: 260,
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, font: { size: 11, weight: "600" }, callback: (v) => `${v}` },
          title: {
            display: true,
            text: "Interstitial Glucose (mg/dL)",
            color: TICK_COLOR,
            font: { size: 11, weight: "600" },
          },
        },
      },
    },
  });
}

/**
 * 2. Historical 2-Hour Risk Probability Trace Chart
 */
export function renderRiskTimelineChart(canvasId, pastData, bands = {}, secondarySeries = null) {
  if (!pastData) return null;
  const labels = pastData.clocks && pastData.clocks.length
    ? pastData.clocks
    : (pastData.timestamps || []).map(fmtClock);
  const riskPct = (pastData.risk || pastData.risk_primary || []).map((v) => +(Number(v) * 100).toFixed(1));
  const modThreshold = Number(bands.risk_moderate ?? bands.moderate ?? 0.35) * 100;
  const highThreshold = Number(bands.risk_high ?? bands.high ?? 0.70) * 100;

  const datasets = [
    {
      label: "Primary Model 2h Spike Risk (%)",
      data: riskPct,
      borderColor: "#c2410c",
      backgroundColor: "rgba(194, 65, 12, 0.15)",
      borderWidth: 2.2,
      pointRadius: 0,
      pointHoverRadius: 4,
      fill: true,
      tension: 0.25,
    },
    {
      label: `High Risk Band (${highThreshold.toFixed(0)}%)`,
      data: new Array(labels.length).fill(highThreshold),
      borderColor: "rgba(185, 28, 28, 0.75)",
      borderWidth: 1.2,
      borderDash: [4, 4],
      pointRadius: 0,
      fill: false,
    },
    {
      label: `Moderate / Alert Threshold (${modThreshold.toFixed(0)}%)`,
      data: new Array(labels.length).fill(modThreshold),
      borderColor: "rgba(180, 83, 9, 0.65)",
      borderWidth: 1.1,
      borderDash: [3, 4],
      pointRadius: 0,
      fill: false,
    },
  ];

  if (Array.isArray(secondarySeries) && secondarySeries.length === labels.length) {
    datasets.splice(1, 0, {
      label: "Secondary Baseline Model (%)",
      data: secondarySeries.map((v) => +(Number(v) * 100).toFixed(1)),
      borderColor: "#7c2d12",
      borderWidth: 1.5,
      borderDash: [3, 3],
      pointRadius: 0,
      fill: false,
      tension: 0.25,
    });
  }

  return createOrReplaceChart(canvasId, {
    type: "line",
    data: { labels, datasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: {
          display: true,
          labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } },
        },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y}%`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, maxTicksLimit: 10, font: { size: 10, weight: "600" } },
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, stepSize: 25, font: { size: 10, weight: "600" }, callback: (v) => `${v}%` },
          title: { display: true, text: "2h Risk (%)", color: TICK_COLOR, font: { size: 10, weight: "600" } },
        },
      },
    },
  });
}

/**
 * 3. Sensor Card Mini Sparklines
 */
export function renderSensorSparkline(canvasId, sparkline, status = "good") {
  if (!sparkline || !Array.isArray(sparkline.values) || sparkline.values.length === 0) {
    destroyChart(canvasId);
    return null;
  }
  const colorMap = {
    critical: "#b91c1c",
    warning: "#b45309",
    watch: "#9a4a06",
    good: "#1f6f4a",
  };
  const stroke = colorMap[status] || "#b84a27";

  return createOrReplaceChart(canvasId, {
    type: "line",
    data: {
      labels: sparkline.labels || sparkline.values.map((_, i) => i),
      datasets: [
        {
          data: sparkline.values,
          borderColor: stroke,
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    },
  });
}

/**
 * 4. Digital Twin 5-Domain Radar Chart
 */
export function renderTwinRadarChart(canvasId, radarItems = []) {
  if (!Array.isArray(radarItems) || radarItems.length === 0) return null;
  const labels = radarItems.map((r) => r.label);
  const values = radarItems.map((r) => Number(r.value || 0));

  return createOrReplaceChart(canvasId, {
    type: "radar",
    data: {
      labels,
      datasets: [
        {
          label: "Domain Score (0-100)",
          data: values,
          borderColor: "#b84a27",
          backgroundColor: "rgba(184, 74, 39, 0.20)",
          pointBackgroundColor: "#9a3412",
          pointBorderColor: "#fffaf2",
          borderWidth: 2.2,
          pointRadius: 4,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        r: {
          min: 0,
          max: 100,
          angleLines: { color: "rgba(97, 78, 68, 0.22)" },
          grid: { color: "rgba(97, 78, 68, 0.22)" },
          pointLabels: { color: LABEL_STRONG, font: { size: 11, weight: "700" } },
          ticks: {
            stepSize: 25,
            color: TICK_COLOR,
            backdropColor: "transparent",
            font: { size: 9, weight: "600" },
          },
        },
      },
    },
  });
}

/**
 * 5. Live Multi-Signal Charts (Tab 2)
 */
export function renderLiveSignalCharts(sensorsPayload) {
  if (!sensorsPayload || !sensorsPayload.series) return;
  const s = sensorsPayload.series;
  const labels = s.clocks || (s.timestamps || []).map(fmtClock);
  const thresholds = sensorsPayload.thresholds || {};
  const baseline = sensorsPayload.personal_baseline || {};

  // Chart A: Glucose + Personal Baseline
  createOrReplaceChart("chart-live-glucose", {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Glucose (mg/dL)",
          data: s.glucose_mgdl || [],
          borderColor: "#b84a27",
          backgroundColor: "rgba(184, 74, 39, 0.14)",
          borderWidth: 2.2,
          pointRadius: 0,
          fill: true,
          tension: 0.25,
        },
        {
          label: `High Threshold (${thresholds.glucose_high_mgdl || 180} mg/dL)`,
          data: new Array(labels.length).fill(thresholds.glucose_high_mgdl || 180),
          borderColor: "rgba(185, 28, 28, 0.78)",
          borderDash: [5, 5],
          borderWidth: 1.3,
          pointRadius: 0,
        },
        {
          label: `Personal Median (${Math.round(baseline.glucose_median || 140)} mg/dL)`,
          data: new Array(labels.length).fill(baseline.glucose_median || 140),
          borderColor: "rgba(97, 78, 68, 0.6)",
          borderDash: [2, 4],
          borderWidth: 1.1,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, maxTicksLimit: 10 } },
        y: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR },
          title: { display: true, text: "mg/dL", color: TICK_COLOR, font: { weight: "600" } },
        },
      },
    },
  });

  // Chart B: Heart Rate & HRV
  createOrReplaceChart("chart-live-cardio", {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Heart Rate (bpm)",
          data: s.heart_rate_bpm || [],
          borderColor: "#b91c1c",
          borderWidth: 2,
          pointRadius: 0,
          yAxisID: "y",
          tension: 0.25,
        },
        {
          label: "HRV RMSSD (ms)",
          data: s.hrv_rmssd_ms || [],
          borderColor: "#1f6f4a",
          borderWidth: 2,
          pointRadius: 0,
          yAxisID: "y1",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, maxTicksLimit: 10 } },
        y: {
          type: "linear",
          position: "left",
          grid: { color: GRID_COLOR },
          ticks: { color: "#b91c1c", font: { weight: "600" } },
          title: { display: true, text: "HR (bpm)", color: "#b91c1c", font: { weight: "600" } },
        },
        y1: {
          type: "linear",
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#1f6f4a", font: { weight: "600" } },
          title: { display: true, text: "HRV (ms)", color: "#1f6f4a", font: { weight: "600" } },
        },
      },
    },
  });

  // Chart C: Steps & Activity MET
  createOrReplaceChart("chart-live-activity", {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          type: "bar",
          label: "Steps / Interval",
          data: s.steps_5min || [],
          backgroundColor: "rgba(184, 74, 39, 0.50)",
          borderColor: "#b84a27",
          borderWidth: 1,
          yAxisID: "y",
        },
        {
          type: "line",
          label: "Activity (MET)",
          data: s.activity_met || [],
          borderColor: "#9a4a06",
          borderWidth: 2,
          pointRadius: 0,
          yAxisID: "y1",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, maxTicksLimit: 10 } },
        y: {
          position: "left",
          grid: { color: GRID_COLOR },
          ticks: { color: "#b84a27", font: { weight: "600" } },
          title: { display: true, text: "Steps", color: "#b84a27", font: { weight: "600" } },
        },
        y1: {
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#9a4a06", font: { weight: "600" } },
          title: { display: true, text: "MET", color: "#9a4a06", font: { weight: "600" } },
        },
      },
    },
  });

  // Chart D: SpO2 & Primary vs Secondary Risk
  const riskPrimary = (sensorsPayload.risk_series?.primary || []).map((v) => +(Number(v) * 100).toFixed(1));
  const riskSecondary = (sensorsPayload.risk_series?.secondary || []).map((v) => +(Number(v) * 100).toFixed(1));
  createOrReplaceChart("chart-live-spo2-risk", {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Primary Model Risk (%)",
          data: riskPrimary,
          borderColor: "#c2410c",
          borderWidth: 2.2,
          pointRadius: 0,
          yAxisID: "y",
          tension: 0.25,
        },
        {
          label: "Secondary Logistic Risk (%)",
          data: riskSecondary,
          borderColor: "#7c2d12",
          borderDash: [4, 3],
          borderWidth: 1.7,
          pointRadius: 0,
          yAxisID: "y",
          tension: 0.25,
        },
        {
          label: "SpO₂ (%)",
          data: s.spo2_pct || [],
          borderColor: "#1f6f4a",
          borderWidth: 1.7,
          pointRadius: 0,
          yAxisID: "y1",
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, maxTicksLimit: 10 } },
        y: {
          min: 0,
          max: 100,
          position: "left",
          grid: { color: GRID_COLOR },
          ticks: { color: "#c2410c", font: { weight: "600" }, callback: (v) => `${v}%` },
          title: { display: true, text: "2h Risk (%)", color: "#c2410c", font: { weight: "600" } },
        },
        y1: {
          min: 88,
          max: 100,
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#1f6f4a", font: { weight: "600" }, callback: (v) => `${v}%` },
          title: { display: true, text: "SpO₂ (%)", color: "#1f6f4a", font: { weight: "600" } },
        },
      },
    },
  });
}

/**
 * 6. Local Explainability Charts (Tab 3)
 */
export function renderAttributionCharts(attribution) {
  if (!attribution) return;
  const top = attribution.top_contributors || [];
  const labels = top.map((c) => c.label || c.feature);
  const values = top.map((c) => +(Number(c.contribution || 0) * 100).toFixed(2));
  const colors = values.map((v) => (v >= 0 ? "rgba(185, 28, 28, 0.80)" : "rgba(31, 111, 74, 0.80)"));
  const borders = values.map((v) => (v >= 0 ? "#991b1b" : "#145337"));

  createOrReplaceChart("chart-explain-local", {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Risk Contribution (percentage points)",
          data: values,
          backgroundColor: colors,
          borderColor: borders,
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label(ctx) {
              const item = top[ctx.dataIndex];
              const sign = ctx.parsed.x >= 0 ? "+" : "";
              return `${sign}${ctx.parsed.x} pp (Patient: ${item?.value ?? "—"} ${item?.unit || ""}, Ref: ${item?.reference_value ?? "—"})`;
            },
          },
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, font: { weight: "600" }, callback: (v) => `${v > 0 ? "+" : ""}${v} pp` },
          title: {
            display: true,
            text: "Shift in Predicted 2h Risk (Percentage Points vs Reference)",
            color: TICK_COLOR,
            font: { weight: "600" },
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: LABEL_STRONG, font: { size: 11, weight: "600" } },
        },
      },
    },
  });

  const groups = attribution.by_group || [];
  const gLabels = groups.map((g) => g.group_label || g.group);
  const gValues = groups.map((g) => +(Number(g.contribution || 0) * 100).toFixed(2));
  createOrReplaceChart("chart-explain-groups", {
    type: "bar",
    data: {
      labels: gLabels,
      datasets: [
        {
          label: "Group Net Contribution (pp)",
          data: gValues,
          backgroundColor: gValues.map((v) => (v >= 0 ? "rgba(194, 65, 12, 0.78)" : "rgba(31, 111, 74, 0.78)")),
          borderColor: gValues.map((v) => (v >= 0 ? "#9a3412" : "#145337")),
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label: (ctx) => `${ctx.parsed.x >= 0 ? "+" : ""}${ctx.parsed.x} percentage points`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, font: { weight: "600" }, callback: (v) => `${v > 0 ? "+" : ""}${v} pp` },
        },
        y: {
          grid: { display: false },
          ticks: { color: LABEL_STRONG, font: { size: 11, weight: "600" } },
        },
      },
    },
  });
}

/**
 * 7. Global Permutation Importance Chart (Tab 3)
 */
export function renderGlobalImportanceChart(importancePayload) {
  if (!importancePayload || !Array.isArray(importancePayload.global)) return;
  const top = importancePayload.global.slice(0, 14);
  const labels = top.map((r) => r.label || r.feature);
  const values = top.map((r) => +(Number(r.importance_mean || 0) * 100).toFixed(3));

  createOrReplaceChart("chart-explain-global", {
    type: "bar",
    data: {
      labels,
      datasets: [
        {
          label: "Held-Out AUROC Drop (×100)",
          data: values,
          backgroundColor: "rgba(184, 74, 39, 0.75)",
          borderColor: "#9a3412",
          borderWidth: 1,
          borderRadius: 4,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          ...TOOLTIP_STYLE,
          callbacks: {
            label: (ctx) => `AUROC drop: ${(ctx.parsed.x / 100).toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          grid: { color: GRID_COLOR },
          ticks: { color: TICK_COLOR, font: { weight: "600" } },
          title: {
            display: true,
            text: "Mean ROC-AUC Drop on Held-Out Patients (×100)",
            color: TICK_COLOR,
            font: { weight: "600" },
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: LABEL_STRONG, font: { size: 11, weight: "600" } },
        },
      },
    },
  });
}

/**
 * 8. Full Scenario-Day Chart (Tab 4)
 */
export function renderDayViewChart(dayPayload) {
  if (!dayPayload || !dayPayload.series) return;
  const s = dayPayload.series;
  const labels = (s.timestamps || []).map(fmtClock);
  const glucose = s.glucose_mgdl || [];
  const riskPct = (s.risk || []).map((v) => +(Number(v) * 100).toFixed(1));

  createOrReplaceChart("chart-timeline-day", {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Glucose (mg/dL)",
          data: glucose,
          borderColor: "#b84a27",
          backgroundColor: "rgba(184, 74, 39, 0.12)",
          borderWidth: 2.2,
          pointRadius: 0,
          fill: true,
          yAxisID: "y",
          tension: 0.25,
        },
        {
          label: "2h Spike Risk (%)",
          data: riskPct,
          borderColor: "#d97706",
          borderWidth: 2,
          borderDash: [4, 3],
          pointRadius: 0,
          yAxisID: "y1",
          tension: 0.25,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
        tooltip: TOOLTIP_STYLE,
      },
      scales: {
        x: { grid: { color: GRID_COLOR }, ticks: { color: TICK_COLOR, maxTicksLimit: 12, font: { weight: "600" } } },
        y: {
          position: "left",
          grid: { color: GRID_COLOR },
          ticks: { color: "#b84a27", font: { weight: "600" } },
          title: { display: true, text: "Glucose (mg/dL)", color: "#b84a27", font: { weight: "600" } },
        },
        y1: {
          min: 0,
          max: 100,
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#b45309", font: { weight: "600" }, callback: (v) => `${v}%` },
          title: { display: true, text: "2h Risk (%)", color: "#b45309", font: { weight: "600" } },
        },
      },
    },
  });
}

/**
 * 9. Model Validation Charts: ROC Curve, Calibration Reliability Diagram, Threshold Selection (Tab 5)
 */
export function renderValidationCharts(metricsPayload, calibrationPayload) {
  const testSplit = metricsPayload?.splits?.test || {};
  const roc = testSplit.roc_curve || {};
  if (Array.isArray(roc.fpr) && Array.isArray(roc.tpr) && roc.fpr.length > 0) {
    const rocPoints = roc.fpr.map((x, idx) => ({ x: Number(x), y: Number(roc.tpr[idx]) }));
    createOrReplaceChart("chart-val-roc", {
      type: "scatter",
      data: {
        datasets: [
          {
            label: `Primary Model ROC (AUROC ${testSplit.auroc ?? "—"})`,
            data: rocPoints,
            showLine: true,
            borderColor: "#b84a27",
            backgroundColor: "rgba(184, 74, 39, 0.16)",
            borderWidth: 2.3,
            pointRadius: 0,
            fill: true,
            tension: 0.15,
          },
          {
            label: "Chance Diagonal (0.50)",
            data: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ],
            showLine: true,
            borderColor: "rgba(97, 78, 68, 0.55)",
            borderDash: [4, 4],
            borderWidth: 1.2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
          tooltip: TOOLTIP_STYLE,
        },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "1 − Specificity (False Positive Rate)", color: TICK_COLOR, font: { weight: "600" } },
          },
          y: {
            type: "linear",
            min: 0,
            max: 1,
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "Sensitivity (True Positive Rate)", color: TICK_COLOR, font: { weight: "600" } },
          },
        },
      },
    });
  }

  // Calibration Reliability Curve
  const curve = calibrationPayload?.test_curve || testSplit?.calibration?.curve || {};
  if (Array.isArray(curve.mean_predicted) && Array.isArray(curve.observed_frequency)) {
    const calPoints = [];
    for (let i = 0; i < curve.mean_predicted.length; i += 1) {
      if (curve.mean_predicted[i] !== null && curve.observed_frequency[i] !== null) {
        calPoints.push({
          x: Number(curve.mean_predicted[i]),
          y: Number(curve.observed_frequency[i]),
        });
      }
    }
    createOrReplaceChart("chart-val-calibration", {
      type: "scatter",
      data: {
        datasets: [
          {
            label: `Empirical Calibration (ECE ${testSplit?.calibration?.ece ?? "—"})`,
            data: calPoints,
            showLine: true,
            borderColor: "#9a4a06",
            backgroundColor: "#9a4a06",
            borderWidth: 2.2,
            pointRadius: 4,
          },
          {
            label: "Perfectly Calibrated (y = x)",
            data: [
              { x: 0, y: 0 },
              { x: 1, y: 1 },
            ],
            showLine: true,
            borderColor: "rgba(97, 78, 68, 0.55)",
            borderDash: [4, 4],
            borderWidth: 1.2,
            pointRadius: 0,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
          tooltip: TOOLTIP_STYLE,
        },
        scales: {
          x: {
            type: "linear",
            min: 0,
            max: 1,
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "Mean Predicted Risk Probability", color: TICK_COLOR, font: { weight: "600" } },
          },
          y: {
            type: "linear",
            min: 0,
            max: 1,
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "Observed Event Frequency", color: TICK_COLOR, font: { weight: "600" } },
          },
        },
      },
    });
  }

  // Threshold Selection Curve
  const threshSel = metricsPayload?.threshold_selection || {};
  if (Array.isArray(threshSel.grid) && Array.isArray(threshSel.f1_at_grid) && threshSel.grid.length > 0) {
    createOrReplaceChart("chart-val-threshold", {
      type: "line",
      data: {
        labels: threshSel.grid.map((t) => Number(t).toFixed(2)),
        datasets: [
          {
            label: `Validation F1 (Selected Threshold: ${Number(threshSel.selected ?? 0.44).toFixed(2)})`,
            data: threshSel.f1_at_grid,
            borderColor: "#c2410c",
            backgroundColor: "rgba(194, 65, 12, 0.14)",
            borderWidth: 2.2,
            pointRadius: 3,
            fill: true,
            tension: 0.25,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { labels: { color: TICK_COLOR, boxWidth: 12, font: { size: 11, weight: "600" } } },
          tooltip: TOOLTIP_STYLE,
        },
        scales: {
          x: {
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "Candidate Operating Threshold", color: TICK_COLOR, font: { weight: "600" } },
          },
          y: {
            grid: { color: GRID_COLOR },
            ticks: { color: TICK_COLOR },
            title: { display: true, text: "Validation F1 Score", color: TICK_COLOR, font: { weight: "600" } },
          },
        },
      },
    });
  }
}
