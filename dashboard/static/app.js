const state = {
  scanRuns: [],
  summary: null,
};

const $ = (id) => document.getElementById(id);
const money = (value) => `$${Number(value || 0).toFixed(2)}`;
const pct = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const num = (value, digits = 0) => {
  if (value === null || value === undefined || value === "") return "";
  return Number(value).toLocaleString(undefined, { maximumFractionDigits: digits });
};
const cents = (value) => (value || value === 0 ? `${Number(value).toFixed(0)}c` : "");

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json();
}

function paramsFromFilters() {
  const params = new URLSearchParams();
  const pairs = [
    ["scan_run_id", $("scan-run").value],
    ["outcome", $("outcome").value],
    ["stage", $("stage").value],
    ["category", $("category").value],
    ["ticker", $("ticker").value],
    ["skip_reason_key", $("skip-reason").value],
    ["min_liquidity", $("min-liquidity").value],
    ["max_spread", $("max-spread").value],
    ["min_minutes_to_close", $("min-minutes").value],
    ["max_minutes_to_close", $("max-minutes").value],
  ];
  for (const [key, value] of pairs) {
    if (value) params.set(key, value);
  }
  return params;
}

function setText(id, value) {
  $(id).textContent = value;
}

function renderSummary(summary) {
  const latest = summary.latest_scan_run || {};
  const paper = summary.paper || {};
  const passRate = latest.normalized_markets
    ? Number(latest.passed_count || 0) / Number(latest.normalized_markets)
    : 0;

  setText("raw-markets", num(latest.raw_markets));
  setText("normalized-markets", num(latest.normalized_markets));
  setText("passed-count", num(latest.passed_count));
  setText("rejected-count", num(latest.rejected_count));
  setText("pass-rate", pct(passRate));
  setText("candidates-analyzed", num(latest.candidates_analyzed));
  setText("paper-inserts", num(latest.paper_trades_inserted));
  setText("open-exposure", money(paper.open_exposure));
  setText("paper-pnl", money(paper.total_pnl));
  setText(
    "latest-run-label",
    latest.id ? `${latest.started_at} (${latest.mode || "unknown"})` : "No scan runs",
  );
}

function renderScanRuns(runs) {
  const select = $("scan-run");
  const current = select.value;
  select.innerHTML = `<option value="">Latest / all</option>`;
  for (const run of runs) {
    const option = document.createElement("option");
    option.value = run.id;
    option.textContent = `${run.started_at} | raw ${run.raw_markets || 0} | pass ${run.passed_count || 0}`;
    select.appendChild(option);
  }
  if (current) select.value = current;
}

function renderSkipReasons(rows) {
  const body = $("skip-reasons-body");
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="3" class="muted">No skipped markets recorded yet.</td></tr>`;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><button class="reason-link" type="button">${escapeHtml(row.reason)}</button></td>
      <td class="numeric">${num(row.count)}</td>
      <td>${escapeHtml((row.examples || []).filter(Boolean).join(", "))}</td>
    `;
    tr.querySelector("button").addEventListener("click", () => {
      $("skip-reason").value = row.reason;
      $("outcome").value = "skipped";
      refreshAudit();
    });
    body.appendChild(tr);
  }
}

function renderAudit(rows) {
  const body = $("audit-body");
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="16" class="muted">No audit records match the current filters.</td></tr>`;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const outcome = row.outcome || "unknown";
    tr.innerHTML = `
      <td>${escapeHtml(row.created_at || "")}</td>
      <td><strong>${escapeHtml(row.ticker || "")}</strong></td>
      <td title="${escapeAttr(row.title || "")}">${escapeHtml(row.title || "")}</td>
      <td>${escapeHtml(row.category || "")}</td>
      <td>${escapeHtml(row.stage || "")}</td>
      <td><span class="stage stage-${escapeAttr(outcome)}">${escapeHtml(outcome)}</span></td>
      <td title="${escapeAttr(row.skip_reason || "")}">${escapeHtml(row.skip_reason_key || row.skip_reason || "")}</td>
      <td class="numeric">${cents(row.yes_bid)} / ${cents(row.yes_ask)}</td>
      <td class="numeric">${cents(row.spread_cents)}</td>
      <td class="numeric">${num(row.volume_24h)}</td>
      <td class="numeric">${money(row.liquidity_dollars)}</td>
      <td class="numeric">${num(row.minutes_to_close, 1)}</td>
      <td class="numeric">${num(row.edge_cents, 2)}</td>
      <td class="numeric">${num(row.confidence, 2)}</td>
      <td>${escapeHtml(row.decision_action || "")}</td>
      <td title="${escapeAttr(row.risk_summary || "")}">${escapeHtml(row.risk_summary || "")}</td>
    `;
    body.appendChild(tr);
  }
}

function renderTrades(rows) {
  const body = $("trades-body");
  $("trade-count").textContent = `${rows.length} trades`;
  body.innerHTML = "";
  if (!rows.length) {
    body.innerHTML = `<tr><td colspan="11" class="muted">No paper trades in the ledger yet.</td></tr>`;
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><strong>${escapeHtml(row.ticker || "")}</strong></td>
      <td>${escapeHtml(row.side || "")}</td>
      <td class="numeric">${num(row.contracts)}</td>
      <td class="numeric">${cents(row.entry_price_cents)}</td>
      <td class="numeric">${money(row.dollars_at_risk)}</td>
      <td class="numeric">${pct(row.estimated_yes_prob)}</td>
      <td>${escapeHtml(row.timestamp || "")}</td>
      <td>${escapeHtml(row.result || "")}</td>
      <td class="numeric">${cents(row.exit_price_cents)}</td>
      <td class="numeric">${row.pnl_dollars === null ? "" : money(row.pnl_dollars)}</td>
      <td title="${escapeAttr(row.thesis || "")}">${escapeHtml(row.thesis || "")}</td>
    `;
    body.appendChild(tr);
  }
}

async function refreshAudit() {
  const params = paramsFromFilters();
  const query = params.toString();
  const rows = await getJson(`/api/audit${query ? `?${query}` : ""}`);
  renderAudit(rows);
  $("export-audit").href = `/api/export/audit.csv${query ? `?${query}` : ""}`;
}

async function refreshAll() {
  const [summary, runs, skips, trades] = await Promise.all([
    getJson("/api/summary"),
    getJson("/api/scan-runs"),
    getJson("/api/skip-reasons"),
    getJson("/api/paper-trades"),
  ]);
  state.summary = summary;
  state.scanRuns = runs;
  renderSummary(summary);
  renderScanRuns(runs);
  renderSkipReasons(skips);
  renderTrades(trades);
  await refreshAudit();
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttr(value) {
  return escapeHtml(value).replaceAll("\n", " ");
}

for (const id of [
  "scan-run",
  "outcome",
  "stage",
  "category",
  "ticker",
  "skip-reason",
  "min-liquidity",
  "max-spread",
  "min-minutes",
  "max-minutes",
]) {
  $(id).addEventListener("input", refreshAudit);
}

$("refresh").addEventListener("click", refreshAll);
refreshAll().catch((error) => {
  console.error(error);
  $("audit-body").innerHTML = `<tr><td colspan="16">Dashboard failed to load: ${escapeHtml(error.message)}</td></tr>`;
});
