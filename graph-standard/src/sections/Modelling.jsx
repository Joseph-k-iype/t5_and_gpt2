import { Band, SectionHead, Note, Table, Grid } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import Decider from "../components/Decider.jsx";
import Counter from "../components/Counter.jsx";
import { LABEL_YAML, REIFICATION } from "../lib/samples.js";

const TESTS = [
  ["1", "Does it need its own relationships?", <>A <code>country</code> string cannot participate in a trade-bloc hierarchy; a <code>(:Country)</code> node can.</>],
  ["2", "Does it have an independent lifecycle?", <>If it is created, versioned, expired or approved on a different clock from its parent, it is a node. A risk assessment is superseded; a customer is not.</>],
  ["3", "Is it a shared controlled vocabulary?", <>Reference data — country, currency, product code, business unit — becomes a node so the vocabulary has one authoritative definition and traversals can pivot through it.</>],
  ["4", "Do you need to find all parents by this value, cheaply and often?", <>Repeated global filters are a signal — though a range index is the cheaper first answer, and usually the right one.</>],
  ["5", "Otherwise, it is a property.", <>High-cardinality, single-use, non-relational, descriptive values stay as properties. Do not create a <code>(:Timestamp)</code> node.</>],
];

export default function Modelling() {
  return (
    <Band id="p1" aud="eng sci">
      <SectionHead
        index="03 · Pillar one"
        title="Graph modelling standard"
        aud={["Data engineering", "Data science"]}
      >
        How we shape a graph. In a labelled property graph there is no class hierarchy to formalise — so
        the artefact is not an ontology, it&rsquo;s a registry, and it&rsquo;s concrete.
      </SectionHead>

      <div className="stack">
        <div id="p1-registry">
          <h3 data-reveal>3.1 · The core artefact is a label registry</h3>
          <p data-reveal>Three controlled lists, held in source control, reviewed like code:</p>
          <Grid cols={3} className="mb-m">
            <div className="panel">
              <h4>Node labels</h4>
              <p className="small flat">Each with a definition, an owner, a steward, and a property schema.</p>
            </div>
            <div className="panel">
              <h4>Relationship types</h4>
              <p className="small flat">Each with permitted <code>(source)&nbsp;→&nbsp;(target)</code> label pairs and its own property schema.</p>
            </div>
            <div className="panel">
              <h4>Property keys</h4>
              <p className="small flat">Each with a datatype, a classification, and a canonical definition.</p>
            </div>
          </Grid>
          <p data-reveal>
            <strong>Anything not in the registry is not permitted in an incubation or production graph.</strong>{" "}
            Sandbox graphs are exempt — that is what sandbox is for — but promotion out of sandbox requires
            registration of everything the graph contains.
          </p>
        </div>

        <div id="p1-schema">
          <h3 data-reveal>3.2 · Property schema per label</h3>
          <p data-reveal>
            Every registered label declares its property keys in three tiers. The tiers exist so that
            &ldquo;we don&rsquo;t put risk scores on customers&rdquo; is an enforceable statement rather
            than a preference.
          </p>
          <div className="mb-m">
            <Table head={["Tier", "Meaning", "Enforcement"]}>
              <tr><td>Required</td><td>Must be present on every node carrying this label</td><td><code>MANDATORY</code> constraint</td></tr>
              <tr><td>Optional</td><td>Permitted, defined, may be absent</td><td>Registry only</td></tr>
              <tr><td>Prohibited</td><td>Explicitly disallowed — usually because it belongs on another label, or should be a node</td><td>Conformance check</td></tr>
            </Table>
          </div>
          <Code lang="yaml" caption="registry/labels/customer.yaml" code={LABEL_YAML} />
        </div>

        <div id="p1-decide">
          <h3 data-reveal>3.3 · The property-versus-node decision rule</h3>
          <p data-reveal>
            This is the single most consequential modelling decision, and the one teams get wrong most
            often. Apply the tests in order — <strong>the first &ldquo;yes&rdquo; wins and the attribute
            becomes a node.</strong>
          </p>

          <div className="my-m"><Decider /></div>

          <div className="mb-m">
            <Table head={["#", "Test", "Why it decides"]}>
              {TESTS.map((t) => (
                <tr key={t[0]}>
                  <td className="num">{t[0]}</td>
                  <td>{t[1]}</td>
                  <td>{t[2]}</td>
                </tr>
              ))}
            </Table>
          </div>

          <Note kind="warn" eyebrow="Counter-rule · do not over-nodify">
            <p>
              Every attribute promoted to a node adds a traversal hop to every query that touches it. If
              it fails all four tests, leaving it as a property is the correct answer — not a compromise,
              and not technical debt.
            </p>
          </Note>
        </div>

        <div id="p1-rels">
          <h3 data-reveal>3.4 · Relationship properties and reification</h3>
          <p data-reveal>
            Relationship properties are for <strong>qualifiers of that particular connection</strong>,
            never attributes of either endpoint.
          </p>

          <Grid cols={2} className="mb-m">
            <div className="panel panel-ok">
              <span className="eyebrow c-ok">Permitted on edges</span>
              <ul className="small list-in">
                <li>Temporal validity — <code>validFrom</code>, <code>validTo</code></li>
                <li>Strength or confidence — <code>weight</code>, <code>confidence</code>, <code>matchScore</code></li>
                <li>Provenance — the governance property set</li>
                <li>Role or qualifier — <code>role</code>, <code>ownershipPct</code></li>
              </ul>
            </div>
            <div className="panel panel-stop">
              <span className="eyebrow c-stop">Not permitted on edges</span>
              <ul className="small list-in">
                <li>Attributes of the source or target node — put them on the node</li>
                <li>Anything you will need to traverse <em>from</em> — an edge cannot have edges</li>
              </ul>
            </div>
          </Grid>

          <p data-reveal>
            <strong>The reification rule.</strong> If a relationship needs its own relationships, or needs
            to be referenced by other parts of the graph, it must be reified into an intermediate node —
            typically an event or association node.
          </p>

          <Code caption="reification" code={REIFICATION} />

          <p className="mt-s" data-reveal>
            Prefer reification early for anything that models an <strong>event</strong>, a{" "}
            <strong>transaction</strong>, or a <strong>decision</strong>. Retrofitting it later means
            rewriting every query that touches it.
          </p>
        </div>

        <div id="p1-labels">
          <h3 data-reveal>3.5 · Multi-label discipline</h3>
          <p data-reveal>
            FalkorDB permits multiple labels per node. Uncontrolled, this becomes the primary source of
            model drift — one team&rsquo;s subtype is another team&rsquo;s state flag, and neither is
            written down.
          </p>
          <ul data-reveal>
            <li>Every node carries <strong>exactly one primary type label</strong> — the registered entity type.</li>
            <li>It may carry <strong>zero or more facet labels</strong> drawn from a controlled facet list declared against that primary type: <code>Sanctioned</code>, <code>Deleted</code>, <code>Provisional</code>.</li>
            <li>Facets express <strong>state or classification</strong>, never a subtype with different properties. If the facet brings its own required properties, it is a distinct label or a related node.</li>
            <li>Facets used as soft-delete or lifecycle markers <strong>must</strong> be declared in the registry — because every query author needs to know to exclude them.</li>
          </ul>
        </div>

        <div id="p1-structure">
          <h3 data-reveal>3.6 · Structural rules</h3>
          <Grid cols={2}>
            <div className="panel">
              <h4>Direction is semantic</h4>
              <p className="small flat">Choose the direction that makes the edge read as a sentence, source to target. Never create both directions for the same fact — traversal is bidirectional in Cypher regardless.</p>
            </div>
            <div className="panel">
              <h4>No duplicate parallel edges</h4>
              <p className="small flat">Not of the same type between the same pair, unless they carry distinguishing properties — usually temporal. If they do, document the disambiguating property in the registry.</p>
            </div>
            <div className="panel">
              <h4>No orphan nodes in production</h4>
              <p className="small flat">A node with zero relationships is either reference data (declare it as such) or a defect. It appears in the conformance report either way.</p>
            </div>
            <div className="panel">
              <h4>Supernode policy</h4>
              <p className="small flat">
                Any node exceeding the declared degree threshold — default{" "}
                <Counter value={100000} /> edges — must be reviewed. Remedies: partition by time bucket,
                introduce an intermediate grouping node, or demote the relationship to a property.
              </p>
            </div>
            <div className="panel span-all">
              <h4>Model depth for the query, not the whiteboard</h4>
              <p className="small flat">
                Every additional hop is a cost. Where a common query traverses a fixed three-hop path
                thousands of times a day, a materialised shortcut edge is legitimate — but it must be
                registered, and the pipeline that maintains it is a registered contract like any other. An
                undeclared shortcut edge is indistinguishable from a modelling error.
              </p>
            </div>
          </Grid>
        </div>
      </div>
    </Band>
  );
}
