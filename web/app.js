const LABELS = {
  age: "Age",
  gender: "Gender",
  country: "Country",
  device_type: "Device type",
  device_os: "Device OS",
  merchant_category: "Merchant category",
  transaction_amount: "Transaction amount",
  transaction_type: "Transaction type",
  time_of_day: "Hour of day (0-23)",
  day_of_week: "Day of week (1-7)",
  transaction_duration: "Transaction duration (s)",
  num_prev_transactions: "# previous transactions",
  avg_transaction_amount: "Avg. transaction amount",
  std_transaction_amount: "Std. dev. transaction amount",
  transactions_last_24h: "Transactions in last 24h",
  transactions_last_1h: "Transactions in last 1h",
  failed_login_attempts: "Failed login attempts",
  ip_risk_score: "IP risk score (0-1)",
  device_trust_score: "Device trust score (0-1)",
  account_age_days: "Account age (days)",
  has_chargeback_history: "Chargeback history",
  shared_ip_users: "Users sharing this IP",
  shared_device_users: "Users sharing this device",
  merchant_risk: "Merchant risk score",
  country_risk: "Country risk score",
  distance_from_home: "Distance from home (km)",
  is_new_country: "New country for this user",
};

const GROUPS = [
  {
    title: "Transaction",
    cols: ["transaction_amount", "transaction_type", "merchant_category", "time_of_day", "day_of_week", "transaction_duration", "distance_from_home"],
  },
  {
    title: "Customer",
    cols: ["age", "gender", "country", "account_age_days", "is_new_country"],
  },
  {
    title: "Behavior history",
    cols: ["num_prev_transactions", "avg_transaction_amount", "std_transaction_amount", "transactions_last_24h", "transactions_last_1h"],
  },
  {
    title: "Device & security signals",
    cols: ["device_type", "device_os", "device_trust_score", "failed_login_attempts", "has_chargeback_history", "shared_ip_users", "shared_device_users", "ip_risk_score", "merchant_risk", "country_risk"],
  },
];

const state = { pipeline: null, logreg: null, tree: null, featureIndex: null, samples: null, metrics: null, currentSampleLabel: null };

async function loadJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`Failed to load ${path}: ${res.status}`);
  return res.json();
}

function fieldKind(col, pipeline) {
  if (pipeline.bin_cols.includes(col)) return "bin";
  if (pipeline.nom_cols.includes(col)) return "nom";
  return "num";
}

function buildField(col, pipeline) {
  const kind = fieldKind(col, pipeline);
  const wrap = document.createElement("div");
  wrap.className = "field";
  const label = document.createElement("label");
  label.textContent = LABELS[col] || col;
  label.htmlFor = `f_${col}`;
  wrap.appendChild(label);

  let input;
  if (kind === "num") {
    input = document.createElement("input");
    input.type = "number";
    input.step = "any";
  } else {
    input = document.createElement("select");
    const options = kind === "bin" ? Object.keys(pipeline.ordinal_maps[col]) : pipeline.onehot_maps[col];
    for (const opt of options) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = opt;
      input.appendChild(o);
    }
  }
  input.id = `f_${col}`;
  input.name = col;
  input.dataset.kind = kind;
  wrap.appendChild(input);
  return wrap;
}

function buildForm() {
  const container = document.getElementById("form-groups");
  container.innerHTML = "";
  for (const group of GROUPS) {
    const fs = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = group.title;
    fs.appendChild(legend);
    const grid = document.createElement("div");
    grid.className = "field-grid";
    for (const col of group.cols) {
      grid.appendChild(buildField(col, state.pipeline));
    }
    fs.appendChild(grid);
    container.appendChild(fs);
  }
}

function readForm() {
  const raw = {};
  for (const col of state.pipeline.raw_feature_order) {
    const el = document.getElementById(`f_${col}`);
    raw[col] = el.value === "" ? null : el.value;
  }
  return raw;
}

function fillForm(raw) {
  for (const col of state.pipeline.raw_feature_order) {
    const el = document.getElementById(`f_${col}`);
    if (!el) continue;
    const v = raw[col];
    el.value = v === undefined || v === "" ? "" : v;
  }
}

function clearForm() {
  for (const col of state.pipeline.raw_feature_order) {
    const el = document.getElementById(`f_${col}`);
    if (el && el.tagName === "INPUT") el.value = "";
  }
  state.currentSampleLabel = null;
  document.getElementById("sample-tag").textContent = "";
}

function loadRandomSample() {
  const sample = state.samples[Math.floor(Math.random() * state.samples.length)];
  fillForm(sample.raw);
  state.currentSampleLabel = sample.is_fraud;
  document.getElementById("sample-tag").textContent =
    `Loaded a real row from train.csv — actual label: ${sample.is_fraud ? "FRAUD" : "not fraud"}`;
}

function renderMetrics() {
  const el = document.getElementById("metrics-row");
  const m = state.metrics;
  const card = (title, stats) => `
    <div class="metric-card">
      <h3>${title}</h3>
      ${Object.entries(stats).map(([k, v]) => `<div class="stat-row"><span>${k}</span><span>${(v * 100).toFixed(1)}%</span></div>`).join("")}
    </div>`;
  el.innerHTML =
    card(`Logistic Regression (t=${m.logreg_threshold.toFixed(2)})`, m.logreg) +
    card(`Decision Tree (t=${m.dtl_threshold.toFixed(2)}, ${m.dtl_nodes} nodes)`, m.dtl) +
    `<div class="metric-card"><h3>Dataset</h3><div class="stat-row"><span>Training rows</span><span>${m.n_train_rows.toLocaleString()}</span></div><div class="stat-row"><span>Fraud rate</span><span>${(m.fraud_prevalence * 100).toFixed(1)}%</span></div></div>`;
}

function verdictHTML(proba, threshold) {
  const pct = (proba * 100).toFixed(1);
  const isFraud = proba >= threshold;
  return `
    <span class="verdict ${isFraud ? "fraud" : "safe"}">${isFraud ? "FLAGGED AS FRAUD" : "LOOKS LEGITIMATE"}</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
    <div class="prob-label"><span>0%</span><span>${pct}% fraud probability</span><span>100%</span></div>
    <div class="hint" style="margin-top:8px">flagged when probability &ge; ${(threshold * 100).toFixed(0)}% (F1-optimal threshold on validation set, not 50/50 — fraud is only ${(state.metrics.fraud_prevalence * 100).toFixed(1)}% of the data)</div>
  `;
}

function runPrediction() {
  const raw = readForm();
  const vec = buildFeatureVector(raw, state.pipeline);
  const mode = document.querySelector('input[name="model"]:checked').value;

  const resultsEl = document.getElementById("results");
  resultsEl.innerHTML = "";

  if (mode === "logreg" || mode === "both") {
    const p = predictLogReg(vec, state.logreg);
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `<h3>Logistic Regression</h3>${verdictHTML(p, state.metrics.logreg_threshold)}`;
    resultsEl.appendChild(card);
  }
  if (mode === "dtl" || mode === "both") {
    const p = predictTree(vec, state.tree, state.featureIndex);
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `<h3>Decision Tree (C4.5)</h3>${verdictHTML(p, state.metrics.dtl_threshold)}`;
    resultsEl.appendChild(card);
  }

  if (state.currentSampleLabel !== null) {
    const note = document.createElement("div");
    note.className = "truth-note";
    note.style.gridColumn = "1 / -1";
    note.textContent = `Ground truth for this sampled row: ${state.currentSampleLabel ? "FRAUD" : "not fraud"}`;
    resultsEl.appendChild(note);
  }
}

async function init() {
  const [pipeline, logreg, tree, metrics, samples] = await Promise.all([
    loadJSON("data/pipeline.json"),
    loadJSON("data/logreg.json"),
    loadJSON("data/dtl_tree.json"),
    loadJSON("data/metrics.json"),
    loadJSON("data/samples.json"),
  ]);
  state.pipeline = pipeline;
  state.logreg = logreg;
  state.tree = tree;
  state.metrics = metrics;
  state.samples = samples;
  state.featureIndex = buildFeatureIndex(pipeline.feature_order);

  renderMetrics();
  buildForm();

  document.getElementById("btn-sample").addEventListener("click", loadRandomSample);
  document.getElementById("btn-clear").addEventListener("click", clearForm);
  document.getElementById("btn-predict").addEventListener("click", runPrediction);
}

init().catch((err) => {
  document.getElementById("form-groups").innerHTML = `<p style="color:var(--danger)">Failed to load model artifacts: ${err.message}</p>`;
  console.error(err);
});
