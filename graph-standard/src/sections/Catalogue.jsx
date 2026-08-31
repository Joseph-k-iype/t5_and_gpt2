import { Band, SectionHead, Note, Grid } from "../components/Primitives.jsx";
import Code from "../components/Code.jsx";
import MetaGraph from "../components/MetaGraph.jsx";
import {
  KEY_SAMPLING, Q_UNREGISTERED, Q_BLAST, Q_CONFIDENTIAL, Q_FEDERATION, Q_OWNERLESS,
} from "../lib/samples.js";

export default function Catalogue() {
  return (
    <Band id="p6" aud="ops eng biz sci">
      <SectionHead
        index="08 · Pillar six"
        title="The knowledge catalogue, as a meta-graph"
        aud={["Everyone"]}
      >
        Build the catalogue of the graph estate <em>in</em> a graph. Every question about the estate is
        inherently relational — which labels does this team own, which contracts break if this source
        retires, which tenants share this concept — and those are exactly the queries a graph answers well.
      </SectionHead>

      <div className="stack">
        <Note eyebrow="Dedicated graph key">
          <p>
            <code>prd.platform.catalog.v1</code> — a production tenant like any other, held to this
            standard like any other. If the standard is not good enough for the catalogue, it is not good
            enough.
          </p>
        </Note>

        <div id="p6-layers">
          <h3 data-reveal>8.1 · Two layers, and the gap between them</h3>
          <p data-reveal>
            Every catalogue node carries an <code>_origin</code> property with one of two values. The
            whole design rests on keeping them separate.
          </p>
          <Grid cols={2} className="mb-m">
            <div className="panel">
              <span className="eyebrow c-accent">_origin: observed</span>
              <h4 className="tight">What actually exists</h4>
              <p className="small flat">
                Discovered automatically by introspecting every graph key. Facts, not intentions.
                Refreshed daily; anything unseen for 30 consecutive days is marked <code>Dormant</code>.
              </p>
            </div>
            <div className="panel">
              <span className="eyebrow c-accent">_origin: declared</span>
              <h4 className="tight">What we said would exist</h4>
              <p className="small flat">
                Registered by humans: definitions, ownership, classification, intent, TTL. Sourced from
                the YAML registry and contract files on merge. <strong>Source control is the system of
                record; the catalogue is the queryable projection.</strong>
              </p>
            </div>
          </Grid>

          <Note kind="ok" eyebrow="Drift is the set difference">
            <p>
              <strong>Observed but not declared</strong> is unregistered data.{" "}
              <strong>Declared but not observed</strong> is either a stale registration or a broken
              pipeline. Both are actionable, and both are trivial Cypher once the two layers coexist in
              one graph.
            </p>
          </Note>

          <p className="mt-m" data-reveal>
            Never edit the declared layer in the graph directly. The moment the catalogue and source
            control diverge, you will trust neither, and the catalogue becomes another thing people work
            around.
          </p>
        </div>

        <div id="p6-model">
          <h3 data-reveal>8.2 · The meta-graph model</h3>
          <p data-reveal>
            The spine runs left to right: <strong>who owns it</strong> → <strong>which tenant</strong> →{" "}
            <strong>what is modelled</strong> → <strong>what it means</strong>. Ingestion contracts enter
            from below, connecting source systems to the labels they populate.
          </p>

          <MetaGraph />

          <p className="small mt-s" data-reveal>
            Also in the catalogue, omitted from the diagram to keep the spine readable:{" "}
            <code>IndexDef</code> and <code>ConstraintDef</code> (attached to <code>LabelDef</code>,
            carrying type, target and current status).
          </p>

          <div className="mt-m">
            <Note eyebrow="The piece that makes federation possible">
              <p>
                <code>CanonicalConcept</code>. Two tenants both have a <code>Customer</code> label; both
                align to <code>concept:party.customer</code>. Now you can answer <em>&ldquo;who else
                models this concept&rdquo;</em> without either team having to know the other exists.
              </p>
            </Note>
          </div>
        </div>

        <div>
          <h3 data-reveal>8.3 · Populating the observed layer</h3>
          <p data-reveal>
            A daily job iterates every graph key, calls the introspection procedures
            (<code>db.labels()</code>, <code>db.relationshipTypes()</code>, <code>db.propertyKeys()</code>,{" "}
            <code>db.indexes()</code>, <code>db.constraints()</code>) plus counts, then upserts the
            definition nodes with <code>_origin: 'observed'</code>, <code>firstSeenAt</code> and{" "}
            <code>lastSeenAt</code>.
          </p>
          <Note kind="warn" eyebrow="A trap worth knowing about">
            <p>
              Label and property-key introspection tells you <em>what exists</em>, not{" "}
              <em>what co-occurs</em>. To learn which property keys appear on which label you must sample
              per label — and a sample is a sample, so record the sample size alongside the result.
            </p>
          </Note>
          <div className="mt-s">
            <Code caption="per-label key sampling" code={KEY_SAMPLING} />
          </div>
        </div>

        <div id="p6-queries">
          <h3 data-reveal>8.4 · What it unlocks</h3>
          <p data-reveal>
            These five queries are the reason the catalogue is a graph and not a spreadsheet. None of them
            is answerable today.
          </p>
          <div className="stack-s">
            <Code caption="unregistered labels, worst offenders first" code={Q_UNREGISTERED} />
            <Code caption="blast radius of decommissioning a source" code={Q_BLAST} />
            <Code caption="everywhere a confidential key is used" code={Q_CONFIDENTIAL} />
            <Code caption="tenants modelling the same concept — federation candidates" code={Q_FEDERATION} />
            <Code caption="ownerless graphs" code={Q_OWNERLESS} />
          </div>
        </div>
      </div>
    </Band>
  );
}
