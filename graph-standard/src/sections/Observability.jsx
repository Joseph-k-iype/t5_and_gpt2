import { Band, SectionHead, Grid } from "../components/Primitives.jsx";
import ScoreModel from "../components/ScoreModel.jsx";

export default function Observability() {
  return (
    <Band id="p5" aud="ops eng biz">
      <SectionHead
        index="07 · Pillar five"
        title="Observability and lineage"
        aud={["Platform ops", "Data engineering", "Business & governance"]}
      >
        A standard nobody measures is a preference. Everything here rolls up into one number per graph,
        published weekly, next to the owner&rsquo;s name.
      </SectionHead>

      <div className="stack">
        <div id="p5-metrics">
          <h3 data-reveal>7.1 · What we measure</h3>
          <Grid cols={3}>
            <div className="panel">
              <span className="eyebrow">Daily, per graph</span>
              <h4 className="tight">Structural</h4>
              <ul className="small list-in">
                <li>Node count by label; relationship count by type</li>
                <li>Distinct property keys, and key count by label</li>
                <li>Graph memory usage</li>
                <li>Index inventory and type per label/property</li>
                <li>Constraint inventory and status — anything not <code>OPERATIONAL</code> is an alert</li>
                <li>Degree distribution; count over the supernode threshold</li>
                <li>Orphan node count by label</li>
              </ul>
            </div>
            <div className="panel">
              <span className="eyebrow">Per contract</span>
              <h4 className="tight">Freshness &amp; lineage</h4>
              <ul className="small list-in">
                <li><code>max(_ingestedAt)</code> per <code>_contractId</code> against declared cadence → <strong>staleness alert</strong></li>
                <li>Records written, updated, rejected per run</li>
                <li>Rejection reasons, grouped</li>
                <li>Run duration and its trend</li>
                <li>Volume deviation against the trailing baseline</li>
              </ul>
            </div>
            <div className="panel">
              <span className="eyebrow">Continuous</span>
              <h4 className="tight">Query</h4>
              <ul className="small list-in">
                <li>Latency percentiles — p50 / p95 / p99 per graph</li>
                <li>Slow query log, ranked by <em>total</em> time, not worst single execution</li>
                <li>Query count per graph — feeds the inactivity trigger</li>
                <li>Full scans on labels that have an index available</li>
              </ul>
            </div>
          </Grid>
          <p className="small mt-s" data-reveal>
            That last one is usually a query-authoring defect and occasionally a missing index. Both are
            worth knowing; neither shows up anywhere else.
          </p>
        </div>

        <div id="p5-score">
          <h3 data-reveal>7.2 · The conformance score</h3>
          <p data-reveal>
            One number per graph, published weekly, so the standard is measurable rather than
            aspirational. Move the sliders to see how a real tenant would score — the weights are the ones
            proposed for ratification.
          </p>
          <ScoreModel />
          <p className="mt-m" data-reveal>
            <strong>Publish the score alongside the owner&rsquo;s name.</strong> Visibility does more than
            policy. A team that can see its own graph ranked below its peers fixes it without anyone
            having to send an email.
          </p>
        </div>

        <div id="p5-lineage">
          <h3 data-reveal>7.3 · What lineage actually answers</h3>
          <p data-reveal>
            Node-level provenance gives you record lineage. Contract-level metadata gives you dataset
            lineage. Together they answer the three questions that arrive as incidents, not as requests.
          </p>
          <Grid cols={3}>
            <div className="panel">
              <h4>&ldquo;We&rsquo;re decommissioning this source system — what breaks?&rdquo;</h4>
              <p className="small flat mt-s">
                Traverse <code>SourceSystem ← READS_FROM ← IngestionContract → POPULATES → LabelDef → DECLARED_BY → Tenant</code>.
                You get every affected graph, its tier, and its owning team.
              </p>
            </div>
            <div className="panel">
              <h4>&ldquo;This customer record is wrong — where did it come from?&rdquo;</h4>
              <p className="small flat mt-s">
                Read <code>_sourceSystem</code> and <code>_sourceRecordId</code> straight off the node. No
                archaeology, no guessing which of four pipelines wrote it.
              </p>
            </div>
            <div className="panel">
              <h4>&ldquo;That source had a bad batch on Tuesday — what do I reprocess?&rdquo;</h4>
              <p className="small flat mt-s">
                Filter on <code>_contractId</code> and <code>_ingestedAt</code>, then re-run the
                idempotent contract. This is the payoff for the idempotency rule.
              </p>
            </div>
          </Grid>
        </div>
      </div>
    </Band>
  );
}
