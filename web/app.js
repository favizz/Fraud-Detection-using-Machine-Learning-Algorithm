const LABELS = {
  age: "Age",
  gender: "Gender",
  country: "Country",
  device_type: "Device type",
  device_os: "Device OS",
  merchant_category: "Merchant category",
  transaction_amount: "Transaction amount",
  transaction_type: "Transaction type",
  time_of_day: "Time of transaction",
  day_of_week: "Day of the week",
  transaction_duration: "Transaction duration (s)",
  num_prev_transactions: "# previous transactions",
  avg_transaction_amount: "Avg. transaction amount",
  std_transaction_amount: "Std. dev. transaction amount",
  transactions_last_24h: "Transactions in last 24h",
  transactions_last_1h: "Transactions in last 1h",
  failed_login_attempts: "Failed login attempts",
  ip_risk_score: "IP risk score",
  device_trust_score: "Device trust score",
  account_age_days: "Account age (days)",
  has_chargeback_history: "Chargeback history",
  shared_ip_users: "Users sharing this IP",
  shared_device_users: "Users sharing this device",
  merchant_risk: "Merchant risk score",
  country_risk: "Country risk score",
  distance_from_home: "Distance from home",
  is_new_country: "New country for this user",
};

// Plain-language explanations shown when someone taps the "!" next to a field.
const FIELD_HELP = {
  age: "The account holder's age, in years.",
  gender: "The gender on file for this account.",
  country: "The country the transaction is being made from.",
  device_type: "The kind of device used to make the transaction.",
  device_os: "The operating system running on that device.",
  merchant_category: "What kind of business the payment is going to, e.g. groceries or travel.",
  transaction_amount: "The amount being charged, in the account's local currency.",
  transaction_type: "Whether this is a purchase, refund, transfer, or top-up.",
  time_of_day: "What time the transaction happened, in 24-hour time. 0 is midnight, 14 is 2pm, 23 is 11pm.",
  day_of_week: "Which day of the week the transaction happened on.",
  transaction_duration: "How long the transaction took to complete, in seconds.",
  num_prev_transactions: "How many transactions this account has made before this one.",
  avg_transaction_amount: "This account's average transaction amount, historically.",
  std_transaction_amount: "How much this account's transaction amounts normally vary.",
  transactions_last_24h: "How many transactions this account has made in the past 24 hours.",
  transactions_last_1h: "How many transactions this account has made in the past hour.",
  failed_login_attempts: "How many failed login attempts happened right before this transaction.",
  ip_risk_score: "A 0 to 1 score for how risky the transaction's IP address looks. Higher is riskier.",
  device_trust_score: "A 0 to 1 score for how trusted this device is. Higher is more trusted.",
  account_age_days: "How long ago this account was created, in days.",
  has_chargeback_history: "Whether this account has ever had a chargeback filed against it.",
  shared_ip_users: "How many other accounts have used this same IP address.",
  shared_device_users: "How many other accounts have used this same device.",
  merchant_risk: "A 0 to 1 score for how risky this merchant has been historically. Higher is riskier.",
  country_risk: "A 0 to 1 score for how risky this country has been historically. Higher is riskier.",
  distance_from_home: "How far this transaction's location is from the account holder's home address, in kilometers.",
  is_new_country: "Whether this is the first time this account has transacted from this country.",
};

// Bool-ish fields displayed as Yes/No instead of raw 0/1 codes.
const BOOL_OPTION_LABELS = { 0: "No", 1: "Yes" };

// Fields a person actually knows/enters when checking a single transaction.
// Everything else is real-time telemetry the system would already have on
// file (account history, device trust, risk scores) - kept out of the way
// under "Advanced", pre-seeded with typical values from the dataset.
const PRIMARY_COLS = [
  "transaction_amount", "merchant_category", "transaction_type",
  "time_of_day", "country", "device_type", "age", "gender",
  "is_new_country", "distance_from_home",
];

const ADVANCED_GROUPS = [
  {
    title: "Account & history",
    cols: ["day_of_week", "transaction_duration", "account_age_days", "num_prev_transactions", "avg_transaction_amount", "std_transaction_amount", "transactions_last_24h", "transactions_last_1h"],
  },
  {
    title: "Device & risk signals",
    cols: ["device_os", "failed_login_attempts", "ip_risk_score", "device_trust_score", "has_chargeback_history", "shared_ip_users", "shared_device_users", "merchant_risk", "country_risk"],
  },
];

// Numeric fields that represent counts/days/whole units - shown with a numeric (not decimal) keypad.
const INTEGER_COLS = new Set(["age", "time_of_day", "account_age_days", "num_prev_transactions", "transactions_last_24h", "transactions_last_1h", "failed_login_attempts", "shared_ip_users", "shared_device_users", "day_of_week"]);

const MODEL_INFO = {
  logreg: { name: "Model A", subtitle: "Statistical scoring" },
  dtl: { name: "Model B", subtitle: "Decision rules" },
};

const METRIC_LABELS = {
  accuracy: "Overall accuracy",
  recall: "Fraud caught",
  precision: "Accuracy when flagged",
};

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

function seededDefault(col, kind, pipeline) {
  if (kind === "num") {
    const median = pipeline.impute_medians[col];
    return INTEGER_COLS.has(col) ? String(Math.round(median)) : String(Math.round(median * 100) / 100);
  }
  return pipeline.impute_modes[col];
}

// Single shared tooltip element, positioned next to whichever "!" button was tapped.
let popoverEl = null;

function ensurePopover() {
  if (popoverEl) return popoverEl;
  popoverEl = document.createElement("div");
  popoverEl.className = "field-popover";
  popoverEl.hidden = true;
  document.body.appendChild(popoverEl);
  document.addEventListener("click", (e) => {
    if (!popoverEl.hidden && !popoverEl.contains(e.target) && !e.target.classList.contains("info-btn")) {
      popoverEl.hidden = true;
    }
  });
  window.addEventListener("scroll", () => { popoverEl.hidden = true; }, true);
  window.addEventListener("resize", () => { popoverEl.hidden = true; });
  return popoverEl;
}

function toggleFieldHelp(btn, col) {
  const el = ensurePopover();
  if (!el.hidden && el.dataset.for === col) {
    el.hidden = true;
    return;
  }
  el.textContent = FIELD_HELP[col] || "";
  el.dataset.for = col;
  el.hidden = false;
  const rect = btn.getBoundingClientRect();
  el.style.top = `${rect.bottom + window.scrollY + 6}px`;
  el.style.left = `${rect.left + window.scrollX}px`;
  requestAnimationFrame(() => {
    const maxLeft = window.scrollX + document.documentElement.clientWidth - el.offsetWidth - 10;
    if (rect.left > maxLeft) el.style.left = `${Math.max(10, maxLeft)}px`;
  });
}

function buildField(col, pipeline, { seed = false } = {}) {
  const kind = fieldKind(col, pipeline);
  const wrap = document.createElement("div");
  wrap.className = "field";

  const labelRow = document.createElement("div");
  labelRow.className = "field-label-row";
  const label = document.createElement("label");
  label.textContent = LABELS[col] || col;
  label.htmlFor = `f_${col}`;
  labelRow.appendChild(label);

  if (FIELD_HELP[col]) {
    const helpBtn = document.createElement("button");
    helpBtn.type = "button";
    helpBtn.className = "info-btn";
    helpBtn.textContent = "!";
    helpBtn.setAttribute("aria-label", `What does ${LABELS[col] || col} mean?`);
    helpBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleFieldHelp(helpBtn, col);
    });
    labelRow.appendChild(helpBtn);
  }
  wrap.appendChild(labelRow);

  let input;
  if (kind === "num") {
    input = document.createElement("input");
    input.type = "text";
    input.inputMode = INTEGER_COLS.has(col) ? "numeric" : "decimal";
    input.autocomplete = "off";
  } else {
    input = document.createElement("select");
    const options = kind === "bin" ? Object.keys(pipeline.ordinal_maps[col]) : pipeline.onehot_maps[col];
    for (const opt of options) {
      const o = document.createElement("option");
      o.value = opt;
      o.textContent = BOOL_OPTION_LABELS[opt] ?? opt;
      input.appendChild(o);
    }
  }
  input.id = `f_${col}`;
  input.name = col;
  input.dataset.kind = kind;
  if (seed) input.value = seededDefault(col, kind, pipeline);
  wrap.appendChild(input);
  return wrap;
}

function fieldGrid(cols, pipeline, { seed = false } = {}) {
  const grid = document.createElement("div");
  grid.className = "field-grid";
  for (const col of cols) grid.appendChild(buildField(col, pipeline, { seed }));
  return grid;
}

function buildForm() {
  const primaryEl = document.getElementById("primary-fields");
  primaryEl.innerHTML = "";
  primaryEl.appendChild(fieldGrid(PRIMARY_COLS, state.pipeline, { seed: false }));

  const advancedEl = document.getElementById("advanced-groups");
  advancedEl.innerHTML = "";
  for (const group of ADVANCED_GROUPS) {
    const fs = document.createElement("fieldset");
    const legend = document.createElement("legend");
    legend.textContent = group.title;
    fs.appendChild(legend);
    fs.appendChild(fieldGrid(group.cols, state.pipeline, { seed: true }));
    advancedEl.appendChild(fs);
  }

  seedTimeDefaults();
}

// "Daily basis" convenience: default the time fields to right now.
function seedTimeDefaults() {
  const now = new Date();
  document.getElementById("f_time_of_day").value = now.getHours();
  document.getElementById("f_day_of_week").value = String(now.getDay());
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
  buildForm();
  document.getElementById("advanced-details").open = false;
  state.currentSampleLabel = null;
  document.getElementById("sample-tag").textContent = "";
  document.getElementById("results").innerHTML = "";
}

function loadRandomSample() {
  const sample = state.samples[Math.floor(Math.random() * state.samples.length)];
  fillForm(sample.raw);
  state.currentSampleLabel = sample.is_fraud;
  document.getElementById("advanced-details").open = true;
  document.getElementById("sample-tag").textContent =
    `Loaded a real transaction from our records. Actual outcome: ${sample.is_fraud ? "Fraud" : "Not fraud"}`;
}

function renderMetrics() {
  const el = document.getElementById("metrics-row");
  const m = state.metrics;
  const card = (title, subtitle, stats) => `
    <div class="metric-card">
      <h3>${title}<br><span class="hint-inline">${subtitle}</span></h3>
      ${Object.entries(stats)
        .filter(([k]) => METRIC_LABELS[k])
        .map(([k, v]) => `<div class="stat-row"><span>${METRIC_LABELS[k]}</span><span>${(v * 100).toFixed(1)}%</span></div>`)
        .join("")}
    </div>`;
  el.innerHTML =
    card(MODEL_INFO.logreg.name, MODEL_INFO.logreg.subtitle, m.logreg) +
    card(MODEL_INFO.dtl.name, MODEL_INFO.dtl.subtitle, m.dtl) +
    `<div class="metric-card"><h3>This dataset</h3><div class="stat-row"><span>Transactions analyzed</span><span>${m.n_train_rows.toLocaleString()}</span></div><div class="stat-row"><span>Fraud rate</span><span>${(m.fraud_prevalence * 100).toFixed(1)}%</span></div></div>`;
}

// How close (in percentage points) a score has to be to a model's own threshold
// before it counts as a close call rather than a confident read.
const RISK_MARGIN = 0.07;

function riskTier(proba, threshold) {
  if (proba >= threshold + RISK_MARGIN) return "high";
  if (proba < threshold - RISK_MARGIN) return "low";
  return "borderline";
}

const TIER_META = {
  low: { label: "LOW RISK", cls: "safe" },
  borderline: { label: "BORDERLINE", cls: "warn" },
  high: { label: "HIGH RISK", cls: "fraud" },
};

function verdictHTML(proba, threshold) {
  const pct = (proba * 100).toFixed(1);
  const tier = riskTier(proba, threshold);
  const meta = TIER_META[tier];
  const distance = (Math.abs(proba - threshold) * 100).toFixed(1);
  const note = tier === "borderline"
    ? `Only ${distance} points from the ${(threshold * 100).toFixed(0)}% cutoff. Treat this as a close call.`
    : `${distance} points ${proba >= threshold ? "above" : "below"} the ${(threshold * 100).toFixed(0)}% cutoff.`;
  return `
    <span class="verdict ${meta.cls}">${meta.label}</span>
    <div class="prob-bar-track"><div class="prob-bar-fill" style="width:${pct}%"></div></div>
    <div class="prob-label"><span>0%</span><span>${pct}% risk score</span><span>100%</span></div>
    <div class="hint" style="margin-top:8px">${note}</div>
  `;
}

// Turns one or two model tiers into a single, actionable call.
function actionSummary(tiers) {
  if (tiers.length === 1) {
    if (tiers[0] === "low") return { label: "APPROVE", cls: "safe", action: "This looks safe to approve as normal." };
    if (tiers[0] === "high") return { label: "ESCALATE", cls: "fraud", action: "This looks risky. Hold it and escalate for manual review." };
    return { label: "REVIEW", cls: "warn", action: "This is a close call. Send it for manual review before deciding." };
  }
  const allLow = tiers.every((t) => t === "low");
  const allHigh = tiers.every((t) => t === "high");
  const disagree = tiers.includes("low") && tiers.includes("high");
  if (allLow) return { label: "APPROVE", cls: "safe", action: "Both models agree this looks safe to approve." };
  if (allHigh) return { label: "ESCALATE", cls: "fraud", action: "Both models agree this looks risky. Hold it and escalate for manual review." };
  if (disagree) return { label: "REVIEW", cls: "warn", action: "The two models disagree on this one. Send it for manual review." };
  return { label: "REVIEW", cls: "warn", action: "At least one model flagged this as a close call. Send it for manual review." };
}

function runPrediction() {
  const raw = readForm();
  const vec = buildFeatureVector(raw, state.pipeline);
  const mode = document.querySelector('input[name="model"]:checked').value;

  const resultsEl = document.getElementById("results");
  resultsEl.innerHTML = "";

  const modelResults = [];
  if (mode === "logreg" || mode === "both") {
    const p = predictLogReg(vec, state.logreg);
    modelResults.push({ key: "logreg", proba: p, threshold: state.metrics.logreg_threshold });
  }
  if (mode === "dtl" || mode === "both") {
    const p = predictTree(vec, state.tree, state.featureIndex);
    modelResults.push({ key: "dtl", proba: p, threshold: state.metrics.dtl_threshold });
  }

  const summary = actionSummary(modelResults.map((r) => riskTier(r.proba, r.threshold)));
  const summaryCard = document.createElement("div");
  summaryCard.className = "result-card summary-card";
  summaryCard.style.gridColumn = "1 / -1";
  summaryCard.innerHTML = `
    <span class="verdict ${summary.cls}">${summary.label}</span>
    <p class="summary-text">${summary.action}</p>
  `;
  resultsEl.appendChild(summaryCard);

  for (const r of modelResults) {
    const info = MODEL_INFO[r.key];
    const card = document.createElement("div");
    card.className = "result-card";
    card.innerHTML = `<h3>${info.name} <span class="hint-inline">${info.subtitle}</span></h3>${verdictHTML(r.proba, r.threshold)}`;
    resultsEl.appendChild(card);
  }

  if (state.currentSampleLabel !== null) {
    const note = document.createElement("div");
    note.className = "truth-note";
    note.style.gridColumn = "1 / -1";
    note.textContent = `Actual outcome for this transaction: ${state.currentSampleLabel ? "Fraud" : "Not fraud"}`;
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
  document.getElementById("primary-fields").innerHTML = `<p style="color:var(--danger)">Something went wrong loading the risk checker. Please refresh the page.</p>`;
  console.error(err);
});
