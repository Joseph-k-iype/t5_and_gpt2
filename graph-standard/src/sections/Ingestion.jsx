import { Band, SectionHead, Note, Table, Grid } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import { IDEMPOTENT_WRITE } from "../lib/samples.js";

const FIELDS = [
  ["contractId", <>Stable identifier, e.g. <code>ing.kyc.corereg.v2</code></>],
  ["targetGraph", "Graph key this contract writes to"],
  ["sourceSystem", "Registered source system identifier"],
  ["sourceObject", "Table, topic, file pattern, or API endpoint"],
  ["owner", "Accountable team"],
  ["dataOwner", "Named accountable individual for the source data"],
  ["steward", "Named individual accountable for the model mapping"],
  ["classification", "Highest classification of any field in the payload"],
  ["mode", <><code>batch</code> · <code>stream</code> · <code>cdc</code> · <code>manual</code></>],
  ["cadence", "Frequency, and the freshness SLO derived from it"],
  ["writesLabels", "Labels this contract may create or update"],
  ["writesRelTypes", "Relationship types this contract may create"],
  ["mapping", "Source field → label + property key, with transform"],
  ["identityRule", <>How <code>entityId</code> is derived for each label written</>],
  ["mergeSemantics", <><code>create-only</code> · <code>upsert</code> · <code>full-replace</code></>],
  ["indexRequirements", "Indexes this contract depends on"],
  ["retention", "How long records persist; the deletion trigger"],
  ["lineageUpstream", "Upstream datasets or contracts this depends on"],
  ["piiFields", "Explicit list, driving masking and subject-access response"],
];

const GOV = [
  ["_contractId", "string", "Which contract wrote this", <code>MANDATORY</code>],
  ["_sourceSystem", "string", "Originating system", <code>MANDATORY</code>],
  ["_sourceRecordId", "string", "Key in the source system, for round-tripping", "Conformance report"],
  ["_pipelineVersion", "string", "Code version that produced it", "Conformance report"],
  ["_ingestedAt", "datetime", "Write timestamp — drives freshness metrics", <code>MANDATORY</code>],
  ["_classification", "string", "Sensitivity, inherited from the contract", "Conformance report"],
  ["_validFrom / _validTo", "datetime", "Business validity, where bitemporality is required", "Conformance report"],
];

const REJECTS = [
  <>It attempts to write an <strong>unregistered</strong> label, relationship type or property key.</>,
  <>An identity key is <strong>null or empty</strong>, or collides with a different <code>_sourceRecordId</code>.</>,
  <>A <strong>required property is absent</strong> for the label being written.</>,
  <>A relationship <strong>endpoint label pair is not permitted</strong> for that relationship type.</>,
];

export default function Ingestion() {
  return (
    <Band id="p2" aud="eng biz ops">
      <SectionHead
        index="04 · Pillar two"
        title="Ingestion contracts"
        aud={["Data engineering", "Business & governance", "Platform ops"]}
      >
        This is the single control that converts <em>&ldquo;any data can go into the graph from any
        source&rdquo;</em> into a governed estate. Everything else in this document depends on it.
      </SectionHead>

      <div className="stack">
        <div id="p2-gate">
          <Note eyebrow="The gate">
            <p>
              <strong>No data enters an incubation or production graph without a registered ingestion
              contract.</strong> The contract is a versioned artefact in source control, reviewed by the
              graph steward and the data owner, and referenced by ID from every node and relationship it
              creates.
            </p>
          </Note>
        </div>

        <div id="p2-fields">
          <h3 data-reveal>4.2 · Contract fields</h3>
          <p data-reveal>
            A contract is not documentation written after the fact — it is the executable declaration the
            pipeline is validated against. The full template is in <a href="#ax-a">Appendix A</a>.
          </p>
          <Table head={["Field", "Description"]} className="matrix">
            {FIELDS.map(([k, v]) => (
              <tr key={k}>
                <td><code>{k}</code></td>
                <td>{v}</td>
              </tr>
            ))}
          </Table>
        </div>

        <div id="p2-write">
          <h3 data-reveal>4.3 · Write-path rules</h3>
          <Grid cols={2}>
            <div className="panel">
              <h4>Upsert, don&rsquo;t blind-create</h4>
              <p className="small flat">Default write pattern is <code>MERGE</code> on the identity key, then <code>SET</code> properties. Blind <code>CREATE</code> is permitted only for immutable event nodes.</p>
            </div>
            <div className="panel">
              <h4>Idempotency is mandatory</h4>
              <p className="small flat">Re-running a contract over the same input must produce the same graph. This is what makes recovery possible — and recovery is the reason the rule exists.</p>
            </div>
            <div className="panel">
              <h4>Contracts own their labels</h4>
              <p className="small flat">Two contracts writing the same property on the same label is a governance failure. Declare one authoritative and the other read-only, or split the property.</p>
            </div>
            <div className="panel">
              <h4>Batch, don&rsquo;t write node-by-node</h4>
              <p className="small flat">Parameterised Cypher with <code>UNWIND</code> over a batch. Tune against your instance; start around <span className="num">1,000–10,000</span> elements and measure.</p>
            </div>
            <div className="panel span-all">
              <h4>Index before bulk load, or index after — but decide deliberately</h4>
              <p className="small flat">Indexes accelerate the <code>MERGE</code> lookup but add write overhead on every insert. For a large initial load, loading first and indexing after is usually faster; for incremental upserts, the index must already exist.</p>
            </div>
          </Grid>
          <div className="mt-m">
            <Code caption="the canonical idempotent write" code={IDEMPOTENT_WRITE} />
          </div>
        </div>

        <div id="p2-gov">
          <h3 data-reveal>4.4 · The governance property set</h3>
          <p data-reveal>
            Every node and every relationship created by a contract carries these system-managed
            properties. The leading underscore marks them platform-owned and off-limits to application
            logic.
          </p>
          <div className="mb-s">
            <Table head={["Property", "Type", "Purpose", "Production enforcement"]}>
              {GOV.map((g) => (
                <tr key={g[0]}>
                  <td><code>{g[0]}</code></td>
                  <td>{g[1]}</td>
                  <td>{g[2]}</td>
                  <td>{g[3]}</td>
                </tr>
              ))}
            </Table>
          </div>
          <p data-reveal>
            Together these answer four questions that are otherwise unanswerable after the fact:{" "}
            <em>where did this come from</em>, <em>when</em>, <em>by what code</em>, and{" "}
            <em>what must I delete if the source record is withdrawn</em>.
          </p>
        </div>

        <div id="p2-reject">
          <h3 data-reveal>4.5 · Rejection rules</h3>
          <p data-reveal>
            A contract execution fails — <strong>loudly, not silently</strong> — when any of these hold.
            Rejected records go to a quarantine store with the failure reason. They do not go into the
            graph.
          </p>
          <Grid cols={2}>
            {REJECTS.map((r, i) => (
              <div className="panel bl-stop" key={i}>
                <p className="small flat">{r}</p>
              </div>
            ))}
            <div className="panel bl-stop span-all">
              <p className="small flat">
                <strong>Payload volume deviates beyond the declared tolerance.</strong> A blown upstream
                join usually shows up as a 50× row count — and by the time anyone notices in the graph,
                the damage has propagated to every consumer.
              </p>
            </div>
          </Grid>
        </div>
      </div>
    </Band>
  );
}
