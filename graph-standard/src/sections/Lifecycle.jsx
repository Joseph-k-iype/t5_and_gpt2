import { Band, SectionHead, Note, Table, Grid } from "../components/Primitives.jsx";
import GateList from "../components/GateList.jsx";

const TIERS = [
  ["Purpose", "Exploration, PoC", "Validated use case, pre-production", "Business-consumed"],
  ["Isolation", "Shared instance, per-user graph key", "Dedicated graph key", "Dedicated graph key, capacity-reviewed"],
  ["Model governance", "None", "Registry required", "Registry + constraints enforced"],
  ["Ingestion contracts", "Not required", "Required", "Required, reviewed"],
  ["Provenance properties", "Encouraged", "Required", "Enforced by constraint"],
  ["Backup", "None", "Snapshot on request", "Scheduled, tested restore"],
  ["Support", "Best effort", "Business hours", "Per service level agreement"],
  ["Default TTL", "60 days", "180 days, renewable", "None — annual recertification"],
];

const INCUBATION = [
  { k: "inc1", text: "Named business owner and named technical owner" },
  { k: "inc2", text: "Stated use case with expected consumers" },
  { k: "inc3", text: "Labels, relationship types and property keys registered in the catalogue" },
  { k: "inc4", text: "At least one ingestion contract registered for each source" },
  { k: "inc5", text: "Identity strategy declared per label" },
  { k: "inc6", text: "Estimated node and edge volume at 12 months, with a supernode assessment" },
  { k: "inc7", text: "Top five query patterns documented, with required indexes identified" },
  { k: "inc8", text: "Data classification agreed, PII fields listed" },
];

const PRODUCTION = [
  { k: "prd1", text: "Model reviewed by architecture forum; deviations documented and accepted" },
  { k: "prd2", text: <><code>UNIQUE</code> constraints on <code>entityId</code> operational for every label</> },
  { k: "prd3", text: <><code>MANDATORY</code> constraints on the core governance property set</> },
  { k: "prd4", text: "Contracts idempotent and re-runnable, demonstrated" },
  { k: "prd5", text: "Freshness SLO defined per source and monitored" },
  { k: "prd6", text: "Backup and restore exercised end to end" },
  { k: "prd7", text: "Retention and deletion path defined, including subject access and right to erasure" },
  { k: "prd8", text: "Runbook: who is called, and what they do" },
  { k: "prd9", text: "Conformance score at or above the agreed threshold" },
];

const TIMELINE = [
  ["Day 0", "Created", "Graph created with an owner and a TTL recorded in the catalogue. No owner, no graph."],
  ["TTL − 14d", "Notified", "Owner notified. Renewal is a one-click acknowledgement with a stated reason."],
  ["TTL", "Frozen", <>Reads permitted, writes blocked, marked <code>Dormant</code> in the catalogue.</>],
  ["TTL + 30d", "Archived", "Snapshot exported to cold storage; the graph key is deleted."],
  ["+ 90d", "Gone", "Snapshot deleted. This is the last point at which anything is recoverable."],
];

export default function Lifecycle() {
  return (
    <Band id="p4" aud="biz ops eng">
      <SectionHead
        index="06 · Pillar four"
        title="Tenant lifecycle"
        aud={["Business & governance", "Platform ops", "Data engineering"]}
      >
        A graph is created in a moment and lives for years. Three tiers, explicit gates between them, and
        a reaping cycle that keeps the inventory honest without punishing experimentation.
      </SectionHead>

      <div className="stack">
        <div id="p4-tiers">
          <h3 data-reveal>6.1 · Three tiers</h3>
          <Table head={["", "Sandbox", "Incubation", "Production"]} className="matrix">
            {TIERS.map((t) => (
              <tr key={t[0]}>
                <td>{t[0]}</td><td>{t[1]}</td><td>{t[2]}</td><td>{t[3]}</td>
              </tr>
            ))}
          </Table>
        </div>

        <div id="p4-promote">
          <h3 data-reveal>6.2 · Promotion gates</h3>
          <p data-reveal>
            Promotion is a checklist, not a conversation. Tick these against a real tenant — the state is
            remembered in this browser so you can work through it over a few sittings.
          </p>
          <div className="grid g2 full items-start" data-reveal-group>
            <GateList
              title="Sandbox → Incubation"
              storageKey="gs.gate.incubation"
              items={INCUBATION}
            />
            <GateList
              title="Incubation → Production"
              storageKey="gs.gate.production"
              intro="Everything on the left, plus:"
              items={PRODUCTION}
            />
          </div>
        </div>

        <div id="p4-reap">
          <h3 data-reveal>6.3 · Expiry and reaping</h3>
          <p data-reveal>
            Dead PoCs are the main cause of sandbox sprawl. This sequence is deliberately asymmetric:{" "}
            <strong>renewal is easy, deletion is slow.</strong> The goal is a true inventory, not friction.
          </p>

          <ol className="timeline my-m" data-reveal-group>
            {TIMELINE.map(([when, head, body]) => (
              <li key={when}>
                <span className="when">{when}</span>
                <div>
                  <h4>{head}</h4>
                  <p className="small flat">{body}</p>
                </div>
              </li>
            ))}
          </ol>

          <Note kind="warn" eyebrow="Inactivity trigger">
            <p>
              Runs in parallel with the TTL clock: <strong>no queries for 45 days</strong> moves a graph
              into the same sequence regardless of how much TTL remains. A graph nobody queries is not a
              graph anybody needs.
            </p>
          </Note>
        </div>

        <div id="p4-sandbox">
          <h3 data-reveal>6.4 · Shared sandbox rules</h3>
          <p data-reveal>
            The shared sandbox is the most valuable thing on the platform and the easiest to ruin. Four
            rules keep it usable.
          </p>
          <Grid cols={2}>
            <div className="panel">
              <h4>One graph key per user or PoC</h4>
              <p className="small flat">Named <code>dev.sandbox.&lt;user-or-poc&gt;</code>. Shared graph keys become nobody&rsquo;s graph key.</p>
            </div>
            <div className="panel">
              <h4>Per-graph size cap</h4>
              <p className="small flat">Breaching the cap is a signal the PoC should be <em>promoted</em>, not grown in place.</p>
            </div>
            <div className="panel bl-stop">
              <h4>No sensitive production data</h4>
              <p className="small flat">Nothing above the agreed classification threshold in the shared sandbox. Confidential or restricted data requires a dedicated incubation tenant with proper access controls — even for a PoC, even for a week.</p>
            </div>
            <div className="panel">
              <h4>No backup, stated up front</h4>
              <p className="small flat">Sandbox graphs are not backed up, and that is said at creation time rather than discovered at loss time.</p>
            </div>
          </Grid>
        </div>
      </div>
    </Band>
  );
}
