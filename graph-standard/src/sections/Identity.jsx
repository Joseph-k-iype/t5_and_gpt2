import { Band, SectionHead, Note, Table, Grid } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import Linter from "../components/Linter.jsx";
import { NATURAL_KEY, MINTED_KEY } from "../lib/samples.js";

const CONVENTIONS = [
  ["Graph key", <code>&lt;env&gt;.&lt;domain&gt;.&lt;usecase&gt;.v&lt;n&gt;</code>, <><code>prd.risk.kyc-network.v1</code><br /><code>dev.sandbox.shared</code></>],
  ["Node label", "PascalCase, singular noun", <><code>Customer</code> · <code>TradeAccount</code> · <code>RiskAssessment</code></>],
  ["Facet label", "PascalCase adjective or state", <><code>Sanctioned</code> · <code>Deprecated</code></>],
  ["Relationship type", "UPPER_SNAKE_CASE, verb phrase", <><code>HOLDS_ACCOUNT</code> · <code>REPORTS_TO</code> · <code>ISSUED_BY</code></>],
  ["Property key", "camelCase", <><code>legalName</code> · <code>openedAt</code> · <code>ownershipPct</code></>],
  ["Governance property", <><code>_camelCase</code> — reserved prefix</>, <><code>_ingestedAt</code> · <code>_contractId</code></>],
  ["Index name", <code>idx_&lt;label&gt;_&lt;property&gt;</code>, <code>idx_customer_entityid</code>],
  ["Contract ID", <code>ing.&lt;domain&gt;.&lt;source&gt;.v&lt;n&gt;</code>, <code>ing.kyc.corereg.v2</code>],
];

const RULES = [
  ["Singular labels always", <><code>Customer</code>, never <code>Customers</code>. The label describes one node, not the set.</>],
  ["No abbreviations", <>Unless the abbreviation <em>is</em> the enterprise-standard term. <code>KYC</code>, <code>LEI</code>, <code>ISIN</code> are fine; <code>Cust</code>, <code>Txn</code>, <code>Acct</code> are not.</>],
  ["Verbs for edges, nouns for nodes", <><code>OWNS</code>, not <code>OWNER</code>. If you want a noun relationship type, you probably need reification.</>],
  ["No type suffixes on keys", <><code>openedAt</code>, not <code>openedAtDate</code>. The registry declares the type; the key names the thing.</>],
  ["Booleans read as assertions", <><code>isActive</code>, <code>hasConsent</code> — not <code>active</code>, <code>consentFlag</code>.</>],
  ["Underscore is reserved", <>The <code>_</code> prefix belongs to the platform. Application properties must never use it.</>],
];

export default function Identity() {
  return (
    <Band id="p3" aud="eng sci ops">
      <SectionHead
        index="05 · Pillar three"
        title="Naming and identity"
        aud={["Data engineering", "Data science", "Platform ops"]}
      >
        Naming is cheap to standardise on day one and effectively impossible to standardise on day four
        hundred. Identity is what makes two graphs describe the same world.
      </SectionHead>

      <div className="stack">
        <div id="p3-naming">
          <h3 data-reveal>5.1 · Conventions</h3>
          <div className="mb-m">
            <Table head={["Element", "Convention", "Examples"]}>
              {CONVENTIONS.map((c, i) => (
                <tr key={i}>
                  <td>{c[0]}</td>
                  <td>{c[1]}</td>
                  <td>{c[2]}</td>
                </tr>
              ))}
            </Table>
          </div>
          <Grid cols={2}>
            {RULES.map(([h, p]) => (
              <div className="panel" key={h}>
                <h4>{h}</h4>
                <p className="small flat">{p}</p>
              </div>
            ))}
          </Grid>
        </div>

        <div id="p3-linter">
          <h3 data-reveal>5.2 · Check a name against the conventions</h3>
          <p data-reveal>
            Every rule above is mechanically checkable, which means it belongs in CI rather than in a
            review comment. This is the same rule set your linter should implement.
          </p>
          <Linter />
        </div>

        <div id="p3-drift">
          <h3 data-reveal>5.3 · Property keys are the real drift risk</h3>
          <p data-reveal>
            Labels and relationship types are visible in every query and get reviewed. Property keys
            proliferate quietly. <code>customerName</code>, <code>custName</code>, <code>name</code>,{" "}
            <code>legalName</code>, <code>full_name</code> across five tenants is the <em>normal</em>{" "}
            outcome of having no standard — not a worst case.
          </p>
          <Grid cols={2}>
            <div className="panel">
              <span className="eyebrow">Control one</span>
              <h4 className="tight">A shared property key dictionary</h4>
              <p className="small flat">
                Held in the catalogue. Before a contract introduces a new key it must check the dictionary
                and either reuse the existing key or justify a new one. The dictionary holds: key,
                canonical definition, datatype, classification, and the labels it appears on.
              </p>
            </div>
            <div className="panel">
              <span className="eyebrow">Control two</span>
              <h4 className="tight">A weekly drift report</h4>
              <p className="small flat">
                Lists property keys observed in graphs but absent from the dictionary, ranked by node
                count. Anything above threshold goes on the steward&rsquo;s queue. Ranking by node count
                is what stops the report becoming noise.
              </p>
            </div>
          </Grid>
        </div>

        <div id="p3-identity">
          <h3 data-reveal>5.4 · Identity strategy</h3>
          <p data-reveal>
            Every registered label declares <strong>exactly one</strong> identity strategy. There are two
            permitted patterns, and the choice between them is about how many systems get to describe the
            entity.
          </p>

          <Grid cols={2} className="mb-m">
            <div className="panel">
              <span className="eyebrow c-accent">Pattern A</span>
              <h4 className="tight">Namespaced natural key</h4>
              <p className="small">Use when a single authoritative source system owns the entity and its key is stable.</p>
              <Code caption="" code={NATURAL_KEY} />
              <p className="small flat mt-s">
                <strong>Cheap, transparent, debuggable</strong>, and requires no crosswalk. It breaks if
                the source re-keys, or if a second source starts describing the same entity.
              </p>
            </div>
            <div className="panel">
              <span className="eyebrow c-accent">Pattern B</span>
              <h4 className="tight">Minted enterprise identifier</h4>
              <p className="small">Use when multiple sources describe the same entity, or when the entity must survive source-system replacement.</p>
              <Code caption="" code={MINTED_KEY} />
              <p className="small flat mt-s">
                The mint is issued once by the identity service and never changes. Every source key that
                resolves to it is recorded in a crosswalk:{" "}
                <code>(:Customer)-[:IDENTIFIED_BY]-&gt;(:Identifier {"{"}system, value, confidence, resolvedAt{"}"})</code>.
              </p>
            </div>
          </Grid>

          <Note eyebrow="Composite keys">
            <p>
              Derive a deterministic hash of the normalised, ordered components rather than concatenating
              raw values — and <strong>record the hash recipe in the registry</strong>, because it is
              impossible to reverse-engineer later and you will need it the first time a key
              doesn&rsquo;t match.
            </p>
          </Note>

          <h4 className="mt-l mb-s" data-reveal>Identity rules</h4>
          <ul data-reveal>
            <li><strong><code>entityId</code> is mandatory on every node</strong> in incubation and production graphs. It is the only property guaranteed present everywhere.</li>
            <li><strong>A <code>UNIQUE</code> constraint on <code>entityId</code> per label</strong> at production tier. Create the supporting range index first, resolve duplicates first, and verify with <code>db.constraints()</code> that it reached <code>OPERATIONAL</code> rather than <code>FAILED</code>.</li>
            <li><strong>Never key on a mutable business attribute.</strong> Not email, not name, not an account number that gets recycled.</li>
            <li><strong>Never key on the internal node ID.</strong> It is not stable across restore, rebuild, or graph copy.</li>
            <li><strong>Entity resolution is a pipeline concern, not a query concern.</strong> Fuzzy matching happens upstream and produces either a mint assignment or an explicit <code>(:Customer)-[:SAME_AS {"{"}confidence, method, resolvedAt{"}"}]-&gt;(:Customer)</code> edge. Never let probabilistic matches silently collapse two nodes into one — the <code>SAME_AS</code> edge preserves the evidence and can be reversed.</li>
            <li><strong>Cross-tenant identity is declared, not enforced.</strong> Since graphs cannot be joined, two tenants using the same <code>entityId</code> for the same real-world entity is a convention held together by the catalogue. Registering that both align to the same canonical concept is what makes future federation possible.</li>
          </ul>
        </div>
      </div>
    </Band>
  );
}
