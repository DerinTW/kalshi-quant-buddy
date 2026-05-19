import { scannerSummary } from "./mockScannerData.js";

const percent = (value) => `${(Number(value || 0) * 100).toFixed(1)}%`;
const money = (value) => `$${Number(value || 0).toFixed(2)}`;
const cents = (value) => `${Number(value || 0).toFixed(1)}c`;

function StatusPill({ children, tone = "neutral" }) {
  return <span className={`pill pill-${tone}`}>{children}</span>;
}

function SummaryCard({ label, value, detail, tone = "default" }) {
  return (
    <section className={`summary-card summary-card-${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
      <small>{detail}</small>
    </section>
  );
}

function SafetyBanner({ data }) {
  return (
    <section className="safety-banner" aria-label="Prototype safety banner">
      <div>
        <p className="eyebrow">Prototype mode</p>
        <h1>Black Gibbie Paper Scanner Dashboard</h1>
        <p>
          Mock scanner output only. No Kalshi API calls, no live trading path, and no
          environment secrets are read by this frontend.
        </p>
      </div>
      <div className="safety-status" aria-label="Safety status">
        <StatusPill tone={data.execute_paper ? "warning" : "safe"}>
          execute_paper: {String(data.execute_paper)}
        </StatusPill>
        <StatusPill tone={data.dry_run ? "safe" : "warning"}>
          dry_run: {String(data.dry_run)}
        </StatusPill>
        <StatusPill tone="neutral">local mock data</StatusPill>
      </div>
    </section>
  );
}

function ScanSummary({ data }) {
  const rejectedShare = data.normalized_markets
    ? data.rejected_count / data.normalized_markets
    : 0;

  return (
    <section className="section-block" aria-labelledby="scan-summary-title">
      <div className="section-heading">
        <p className="eyebrow">Scan Summary</p>
        <h2 id="scan-summary-title">One pass, paper-only view</h2>
      </div>
      <div className="summary-grid">
        <SummaryCard
          label="Raw Markets"
          value={data.raw_markets}
          detail="Fetched before normalization"
        />
        <SummaryCard
          label="Normalized"
          value={data.normalized_markets}
          detail="Markets ready for filtering"
        />
        <SummaryCard
          label="Passed"
          value={data.passed_count}
          detail={`${percent(data.pass_rate)} pass rate`}
          tone="good"
        />
        <SummaryCard
          label="Rejected"
          value={data.rejected_count}
          detail={`${percent(rejectedShare)} of normalized`}
          tone="warn"
        />
        <SummaryCard
          label="Analyzed"
          value={data.candidates_analyzed}
          detail="Candidates sent through scoring"
        />
        <SummaryCard
          label="Paper Inserts"
          value={data.paper_trades_inserted}
          detail="Ledger writes in this mock run"
          tone={data.paper_trades_inserted > 0 ? "warn" : "good"}
        />
      </div>
    </section>
  );
}

function SkipReasonTable({ data }) {
  const rows = Object.entries(data.skip_reason_counts).sort((a, b) => b[1] - a[1]);

  return (
    <section className="section-block table-section" aria-labelledby="skip-title">
      <div className="section-heading inline-heading">
        <div>
          <p className="eyebrow">Filters</p>
          <h2 id="skip-title">Skip Reason Counts</h2>
        </div>
        <StatusPill tone="neutral">{rows.length} reasons</StatusPill>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Reason</th>
              <th className="numeric">Count</th>
              <th>Examples</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([reason, count]) => (
              <tr key={reason}>
                <td>
                  <code>{reason}</code>
                </td>
                <td className="numeric">{count}</td>
                <td className="example-list">
                  {(data.skip_reason_examples[reason] || []).join(", ") || "No examples"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function TradeDecisionsTable({ data }) {
  return (
    <section className="section-block table-section" aria-labelledby="decisions-title">
      <div className="section-heading inline-heading">
        <div>
          <p className="eyebrow">Candidate Analysis</p>
          <h2 id="decisions-title">Trade Decisions</h2>
        </div>
        <StatusPill tone="neutral">{data.decisions.length} decisions</StatusPill>
      </div>
      <div className="table-wrap wide-table">
        <table>
          <thead>
            <tr>
              <th>Ticker</th>
              <th>Action</th>
              <th>Side</th>
              <th className="numeric">Edge</th>
              <th className="numeric">Confidence</th>
              <th className="numeric">Limit</th>
              <th className="numeric">Contracts</th>
              <th className="numeric">Size</th>
              <th>Risk</th>
              <th>Execution</th>
            </tr>
          </thead>
          <tbody>
            {data.decisions.map((record) => {
              const decision = record.decision || {};
              const action = record.action || decision.action || "UNKNOWN";
              const actionTone =
                action === "NO_TRADE" ? "neutral" : record.risk_approved ? "safe" : "warning";

              return (
                <tr key={record.ticker}>
                  <td>
                    <strong>{record.ticker}</strong>
                    <small>{decision.thesis}</small>
                  </td>
                  <td>
                    <StatusPill tone={actionTone}>{action}</StatusPill>
                  </td>
                  <td>{decision.side || "-"}</td>
                  <td className="numeric">{cents(decision.edge_cents)}</td>
                  <td className="numeric">{percent(decision.confidence)}</td>
                  <td className="numeric">
                    {decision.limit_price_cents ? `${decision.limit_price_cents}c` : "-"}
                  </td>
                  <td className="numeric">{decision.contracts || "-"}</td>
                  <td className="numeric">{money(decision.dollar_size)}</td>
                  <td>
                    <span className="risk-text">{decision.risk_summary}</span>
                  </td>
                  <td>
                    {record.executed ? (
                      <StatusPill tone="warning">executed</StatusPill>
                    ) : (
                      <code>{record.execution_skip_reason || "not_executed"}</code>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function PaperTradesSummary({ data }) {
  const wouldTrade = data.decisions.filter((item) =>
    ["BUY_YES", "BUY_NO"].includes(item.action),
  );
  const notExecuted = data.decisions.filter((item) => !item.executed);

  return (
    <section className="section-block split-section" aria-labelledby="paper-title">
      <div className="section-heading">
        <p className="eyebrow">Paper Trades</p>
        <h2 id="paper-title">Ledger impact summary</h2>
      </div>
      <div className="paper-grid">
        <div className="paper-metric">
          <span>Paper trades inserted</span>
          <strong>{data.paper_trades_inserted}</strong>
        </div>
        <div className="paper-metric">
          <span>Would-trade decisions</span>
          <strong>{wouldTrade.length}</strong>
        </div>
        <div className="paper-metric">
          <span>Not executed</span>
          <strong>{notExecuted.length}</strong>
        </div>
      </div>
      <p className="muted">
        This mock run keeps <code>execute_paper</code> false, so approved BUY decisions
        are visible without inserting paper trades.
      </p>
    </section>
  );
}

function ErrorsPanel({ data }) {
  return (
    <section className="section-block errors-panel" aria-labelledby="errors-title">
      <div className="section-heading inline-heading">
        <div>
          <p className="eyebrow">Reliability</p>
          <h2 id="errors-title">Errors</h2>
        </div>
        <StatusPill tone={data.errors.length ? "warning" : "safe"}>
          {data.errors.length} logged
        </StatusPill>
      </div>
      {data.errors.length === 0 ? (
        <p className="empty-state">No errors recorded for this mock scan.</p>
      ) : (
        <div className="error-list">
          {data.errors.map((error) => (
            <article className="error-item" key={`${error.stage}-${error.ticker}`}>
              <strong>{error.ticker}</strong>
              <span>{error.stage}</span>
              <code>{error.err}</code>
            </article>
          ))}
        </div>
      )}
    </section>
  );
}

export default function App() {
  return (
    <main className="dashboard-shell">
      <SafetyBanner data={scannerSummary} />
      <ScanSummary data={scannerSummary} />
      <div className="two-column">
        <SkipReasonTable data={scannerSummary} />
        <PaperTradesSummary data={scannerSummary} />
      </div>
      <TradeDecisionsTable data={scannerSummary} />
      <ErrorsPanel data={scannerSummary} />
    </main>
  );
}
