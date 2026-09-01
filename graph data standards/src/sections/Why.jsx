import { Band, SectionHead } from "../components/Primitives.jsx";

const QUESTIONS = [
  ["What", <>Which labels, relationship types and property keys exist, in which tenant, at what volume.</>],
  ["Where from", <>Which source system, which pipeline, which owner, how fresh, and what to delete if the source record is withdrawn.</>],
  ["Same thing?", <>Is <code>Customer</code> in the KYC graph the same concept, with the same key, as <code>Customer</code> in the servicing graph.</>],
];

const PILLARS = [
  { n: 1, id: "p1", h: "Graph modelling standard", p: "A registry of labels, relationship types and property keys — with rules for when an attribute becomes a node.", q: "How do we shape a graph?" },
  { n: 2, id: "p2", h: "Ingestion contracts", p: "Nothing enters a governed graph without a versioned, reviewed contract that declares what it writes.", q: "What is allowed in, and on whose authority?" },
  { n: 3, id: "p3", h: "Naming and identity", p: "Conventions that survive contact with five teams, and one identity strategy per label.", q: "Do two nodes refer to the same real-world thing?" },
  { n: 4, id: "p4", h: "Tenant lifecycle", p: "Three tiers, explicit promotion gates, and a reaping cycle that keeps the inventory true.", q: "When does a graph get created, promoted, frozen, deleted?" },
  { n: 5, id: "p5", h: "Observability and lineage", p: "Structural, freshness and query metrics, rolled into a single published conformance score.", q: "Is it healthy, fresh, and conformant?" },
  { n: 6, id: "p6", h: "Knowledge catalogue", p: "The estate's own metadata, modelled as a graph, because every question about it is relational.", q: "What exists across the estate, and who owns it?" },
];

export default function Why() {
  return (
    <Band id="why">
      <SectionHead index="01 · The case" title="Three questions we currently cannot answer">
        The infrastructure works. What&rsquo;s missing is a shared way to answer these — and every pillar
        below exists to make one of them answerable.
      </SectionHead>

      <div className="grid g3 full mb-l" data-reveal-group>
        {QUESTIONS.map(([k, v]) => (
          <div className="stat" key={k}>
            <span className="big">{k}</span>
            <span className="cap">{v}</span>
          </div>
        ))}
      </div>

      <h3 className="mb-s" data-reveal>The six pillars</h3>
      <div className="grid g2 full" data-reveal-group>
        {PILLARS.map((p) => (
          <a className="pillar" href={`#${p.id}`} key={p.n}>
            <span className="n">{p.n}</span>
            <div>
              <h4>{p.h}</h4>
              <p>{p.p}</p>
              <span className="q">{p.q}</span>
            </div>
          </a>
        ))}
      </div>
    </Band>
  );
}
