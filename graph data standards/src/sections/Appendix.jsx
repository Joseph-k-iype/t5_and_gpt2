import { Band, SectionHead, Grid, Table } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import { CONTRACT_YAML, REL_YAML } from "../lib/samples.js";

const DECISIONS = [
  ["Property key case", <>camelCase is assumed throughout. If the wider enterprise standard is snake_case, change it <em>now</em>, before anything is registered.</>],
  ["Bitemporality", <>Is <code>_validFrom</code>/<code>_validTo</code> mandatory across the estate, or opt-in per label? Retrofitting it is expensive.</>],
  ["Soft delete", <>Facet label, <code>validTo</code>, or hard delete? Pick one. A mixed estate is unqueryable — every consumer has to know which convention each tenant chose.</>],
  ["Reference data", "One shared reference graph replicated per tenant, or reference nodes duplicated per tenant? Since graphs cannot be joined, replication is likely the only option — agree the sync mechanism and who owns it."],
  ["Vector embeddings", "If GraphRAG is in scope, embeddings need their own standard: which property key, where the model version is recorded, which index type and similarity function, and the re-embedding policy on model upgrade. An estate with three embedding conventions has none."],
  ["Supernode threshold", <><span className="num">100,000</span> is a placeholder. Set it from your actual latency measurements, not from this document.</>],
  ["Conformance thresholds and grace periods", "What score is required at production tier, and by when? Without a number and a date, the ladder is decoration."],
];

export default function Appendix() {
  return (
    <Band id="appendix">
      <SectionHead
        index="11 · Reference"
        title="Appendices"
      >
        Copy these. They are the working artefacts — everything above exists to explain why they look the
        way they do.
      </SectionHead>

      <div className="stack">
        <div id="ax-a">
          <h3 className="mb-s" data-reveal>Appendix A · Ingestion contract template</h3>
          <Code caption="contracts/ing.kyc.corereg.v2.yaml" code={CONTRACT_YAML} />
        </div>

        <div id="ax-b">
          <h3 className="mb-s" data-reveal>Appendix B · Label registration template</h3>
          <p data-reveal>The worked example is in <a href="#p1-schema">§3.2</a>. Field requirements:</p>
          <Grid cols={2}>
            <div className="panel">
              <span className="eyebrow c-stop">Required</span>
              <p className="small flat mt-s">
                <code>label</code>, <code>domain</code>, <code>owner</code>, <code>steward</code>,{" "}
                <code>definition</code>, <code>identity</code>, <code>properties.required</code>,{" "}
                <code>indexes</code>
              </p>
            </div>
            <div className="panel">
              <span className="eyebrow">Optional</span>
              <p className="small flat mt-s">
                <code>facets</code>, <code>properties.optional</code>,{" "}
                <code>properties.prohibited</code>, <code>alignsTo</code>, <code>expectedVolume</code>,{" "}
                <code>degreeThreshold</code>
              </p>
            </div>
          </Grid>
        </div>

        <div id="ax-c">
          <h3 className="mb-s" data-reveal>Appendix C · Relationship type registration template</h3>
          <Code caption="registry/relationships/holds_account.yaml" code={REL_YAML} />
        </div>

        <div id="ax-d">
          <h3 className="mb-s" data-reveal>Appendix D · Measurement specification</h3>
          <p data-reveal>
            What the daily introspection job must produce for every graph. This is a specification, not an
            implementation: it says what has to be true and when someone must be told, and leaves the
            mechanics to the platform team and the version they are actually running.
          </p>

          <div className="mb-m">
            <Table head={["Measure", "Definition", "Cadence", "Raise it when"]}>
              <tr><td>Label inventory</td><td>Every node label present, with its node count</td><td>Daily</td><td>A label appears that the registry does not declare</td></tr>
              <tr><td>Relationship inventory</td><td>Every relationship type present, with its count and observed endpoint label pairs</td><td>Daily</td><td>An endpoint pair occurs that the registry does not permit</td></tr>
              <tr><td>Property key inventory</td><td>Every property key present; per label, sampled with the sample size recorded</td><td>Daily</td><td>A key is absent from the dictionary and appears on more nodes than the agreed threshold</td></tr>
              <tr><td>Index inventory</td><td>Each index, its type, and the label and properties it covers</td><td>Daily</td><td>An index a contract declares a dependency on is missing</td></tr>
              <tr><td>Constraint status</td><td>Each constraint and its current state</td><td>Daily</td><td>Any constraint is in any state other than operational</td></tr>
              <tr><td>Duplicate identity</td><td>Count of <code>entityId</code> values occurring more than once within a label</td><td>Daily, and before every constraint change</td><td>Greater than zero — at production tier this blocks the promotion</td></tr>
              <tr><td>Missing identity</td><td>Count of nodes per label with no <code>entityId</code></td><td>Daily</td><td>Greater than zero in incubation or production</td></tr>
              <tr><td>Governance completeness</td><td>Per label, the share of nodes carrying each governance property</td><td>Daily</td><td>Below the conformance threshold, or falling week on week</td></tr>
              <tr><td>Orphan rate</td><td>Nodes per label with no relationships, excluding declared reference data</td><td>Daily</td><td>Above the agreed rate, or rising after a contract change</td></tr>
              <tr><td>Degree distribution</td><td>Per label, the degree spread and the count over the supernode threshold</td><td>Daily</td><td>Any node crosses the threshold, or the p99 degree doubles</td></tr>
              <tr><td>Freshness</td><td>Per contract, the most recent write timestamp against the declared cadence</td><td>Hourly</td><td>The gap exceeds the freshness SLO in the contract</td></tr>
              <tr><td>Run outcome</td><td>Per contract run: written, updated, rejected, duration, watermark</td><td>Per run</td><td>Rejections are non-zero, volume falls outside tolerance, or duration trends past its window</td></tr>
              <tr><td>Query profile</td><td>Latency percentiles, slowest patterns by total time, query count</td><td>Continuous</td><td>p99 regresses, or a graph records no queries for 45 days</td></tr>
              <tr><td>Memory</td><td>Graph memory usage against the tenant&rsquo;s allocation</td><td>Daily</td><td>Growth rate would breach the allocation inside the next quarter</td></tr>
            </Table>
          </div>

          <p className="small" data-reveal>
            Two of these are pre-flight checks as well as monitors: <strong>duplicate identity</strong> and{" "}
            <strong>missing identity</strong> must both return zero before any unique constraint is
            created, on a restored snapshot first. See the warning in <a href="#assumptions">§2</a>.
          </p>
        </div>

        <div id="ax-e">
          <h3 className="mb-s" data-reveal>Appendix E · Open decisions for the review forum</h3>
          <p data-reveal>
            Seven decisions this draft does not make. Each one is cheap now and expensive later — which is
            precisely why they are listed rather than assumed.
          </p>
          <ol className="decisions full" data-reveal-group>
            {DECISIONS.map(([h, p]) => (
              <li key={h}>
                <h4>{h}</h4>
                <p className="small flat mt-s">{p}</p>
              </li>
            ))}
          </ol>
        </div>
      </div>
    </Band>
  );
}
