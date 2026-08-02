// Client-side re-implementation of the notebook's run_universal_pipeline
// (impute -> feature engineering -> outlier clip/log -> encode -> scale),
// driven entirely by the constants exported in data/pipeline.json.
// See scripts/export_model.py for the Python side these constants come from.

function buildFeatureVector(rawInput, pipeline) {
  const numVals = {};
  for (const c of pipeline.num_cols) {
    const raw = rawInput[c];
    let v = (raw === null || raw === undefined || raw === "") ? null : Number(raw);
    if (v === null || Number.isNaN(v)) v = pipeline.impute_medians[c];
    numVals[c] = v;
  }

  const catVals = {};
  for (const c of [...pipeline.bin_cols, ...pipeline.nom_cols]) {
    const raw = rawInput[c];
    let v = (raw === null || raw === undefined || raw === "") ? null : String(raw);
    if (v === null) v = pipeline.impute_modes[c];
    catVals[c] = v;
  }

  // generate_new_features
  const safeAmt = Math.max(numVals.transaction_amount, 0);
  numVals.risk_log_amount = numVals.ip_risk_score * Math.log1p(safeAmt);
  numVals.amount_ratio = numVals.transaction_amount / (Math.abs(numVals.avg_transaction_amount) + 1e-6);
  numVals.hour_sin = Math.sin((2 * Math.PI * numVals.time_of_day) / 24);
  numVals.hour_cos = Math.cos((2 * Math.PI * numVals.time_of_day) / 24);

  // handle_outliers_and_skew
  for (const c of [...pipeline.num_cols, ...pipeline.engineered_num_cols]) {
    const info = pipeline.outlier[c];
    let v = numVals[c];
    if (info.log1p) v = Math.log1p(v);
    v = Math.min(Math.max(v, info.lower), info.upper);
    numVals[c] = v;
  }

  // perform_encoding
  const binEncoded = {};
  for (const c of pipeline.bin_cols) {
    const map = pipeline.ordinal_maps[c];
    const key = catVals[c];
    binEncoded[c] = Object.prototype.hasOwnProperty.call(map, key) ? map[key] : -1;
  }
  const onehot = {};
  for (const c of pipeline.nom_cols) {
    for (const cat of pipeline.onehot_maps[c]) {
      onehot[`${c}_${cat}`] = catVals[c] === cat ? 1 : 0;
    }
  }

  // assemble in feature_order, then perform_scaling
  const vec = new Float64Array(pipeline.feature_order.length);
  pipeline.feature_order.forEach((name, j) => {
    let raw;
    if (Object.prototype.hasOwnProperty.call(numVals, name)) raw = numVals[name];
    else if (Object.prototype.hasOwnProperty.call(binEncoded, name)) raw = binEncoded[name];
    else raw = onehot[name];
    const s = pipeline.scaling[name];
    const std = s.std || 1;
    vec[j] = (raw - s.mean) / std;
  });
  return vec;
}

function predictLogReg(vec, logreg) {
  let z = logreg.bias;
  for (let i = 0; i < vec.length; i++) z += vec[i] * logreg.weights[i];
  return 1 / (1 + Math.exp(-z));
}

function buildFeatureIndex(featureOrder) {
  const idx = {};
  featureOrder.forEach((name, j) => { idx[name] = j; });
  return idx;
}

function predictTree(vec, tree, featureIndex) {
  let node = tree;
  while (node.value === undefined) {
    const j = featureIndex[node.feature];
    node = vec[j] <= node.threshold ? node.left : node.right;
  }
  return node.value;
}
