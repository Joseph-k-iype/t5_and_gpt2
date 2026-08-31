import { Band, SectionHead, Grid } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import {
  CONTRACT_YAML, REL_YAML, D_STRUCTURE, D_COUNTS, D_KEYS,
  D_GOVERNANCE, D_PREFLIGHT, D_HEALTH, D_FRESHNESS,
} from "../lib/samples.js";

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
    <Band id="appendix" aud="eng ops sci">
      <SectionHead
        index="11 · Reference"
        title="Appendices"
        aud={["Data engineering", "Platform ops", "Data science"]}
      >
        Copy these. They are the working artefacts — everything above exists to explain why they look the
        way they do.
      </SectionHead>

      <div className="stack">
        <div id="ax-a">
          <h3 className="mb-s" data-reveal>Appendix A · Ingestion contract template</h3>
          <Code lang="yaml" caption="contracts/ing.kyc.corereg.v2.yaml" code={CONTRACT_YAML} />
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
          <Code lang="yaml" caption="registry/relationships/holds_account.yaml" code={REL_YAML} />
        </div>

        <div id="ax-d">
          <h3 className="mb-s" data-reveal>Appendix D · Introspection and conformance queries</h3>
          <p data-reveal>
            Run per graph. Verify procedure and command availability against your FalkorDB version before
            wiring these into a job.
          </p>
          <div className="stack-s">
            <Code caption="structure" code={D_STRUCTURE} />
            <Code caption="counts by label and type" code={D_COUNTS} />
            <Code caption="which property keys appear on a label (sampled)" code={D_KEYS} />
            <Code caption="governance property completeness" code={D_GOVERNANCE} />
            <Code note caption="pre-flight before any unique constraint — must return zero rows" code={D_PREFLIGHT} />
            <Code caption="identity, orphans and supernodes" code={D_HEALTH} />
            <Code caption="freshness by contract" code={D_FRESHNESS} />
          </div>
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
