// Cross-checks web/preprocess.js against the Python pipeline's expected outputs
// (web/data/parity.json, produced by scripts/export_model.py) to catch any
// discrepancy between the JS re-implementation and the real preprocessing/model math.
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const WEB_DIR = path.join(__dirname, "..", "web");
const DATA_DIR = path.join(WEB_DIR, "data");

function loadJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

const preprocessSrc = fs.readFileSync(path.join(WEB_DIR, "preprocess.js"), "utf-8");
const sandbox = {};
vm.createContext(sandbox);
vm.runInContext(preprocessSrc, sandbox);

const pipeline = loadJSON(path.join(DATA_DIR, "pipeline.json"));
const logreg = loadJSON(path.join(DATA_DIR, "logreg.json"));
const tree = loadJSON(path.join(DATA_DIR, "dtl_tree.json"));
const parity = loadJSON(path.join(DATA_DIR, "parity.json"));

const featureIndex = sandbox.buildFeatureIndex(pipeline.feature_order);

let maxLogregDiff = 0;
let maxDtlDiff = 0;
let failures = 0;

for (const [i, row] of parity.entries()) {
  const vec = sandbox.buildFeatureVector(row.raw, pipeline);
  const logregP = sandbox.predictLogReg(vec, logreg);
  const dtlP = sandbox.predictTree(vec, tree, featureIndex);

  const logregDiff = Math.abs(logregP - row.logreg_proba);
  const dtlDiff = Math.abs(dtlP - row.dtl_proba);
  maxLogregDiff = Math.max(maxLogregDiff, logregDiff);
  maxDtlDiff = Math.max(maxDtlDiff, dtlDiff);

  const ok = logregDiff < 1e-6 && dtlDiff < 1e-9;
  if (!ok) failures++;
  console.log(
    `row ${i}: logreg py=${row.logreg_proba.toFixed(6)} js=${logregP.toFixed(6)} diff=${logregDiff.toExponential(2)} | ` +
    `dtl py=${row.dtl_proba.toFixed(6)} js=${dtlP.toFixed(6)} diff=${dtlDiff.toExponential(2)} ${ok ? "OK" : "MISMATCH"}`
  );
}

console.log(`\nmax logreg diff: ${maxLogregDiff.toExponential(3)}`);
console.log(`max dtl diff: ${maxDtlDiff.toExponential(3)}`);
console.log(failures === 0 ? "\nPARITY OK" : `\n${failures} MISMATCHES`);
process.exit(failures === 0 ? 0 : 1);
