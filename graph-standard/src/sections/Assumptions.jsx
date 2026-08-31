import { Band, SectionHead, Note, Table } from "../components/Primitives.jsx";

const ROWS = [
  [
    "Query language is OpenCypher with FalkorDB extensions",
    <>Every example in this document is Cypher. Procedure availability varies by version.</>,
    <code>CALL dbms.procedures()</code>,
  ],
  [
    "A tenant is a graph key on an instance",
    <>Isolation is at the graph level, not the schema level. <strong>There is no cross-graph query.</strong> Any federation happens in the application layer, and reference data must be replicated rather than joined.</>,
    <code>GRAPH.LIST</code>,
  ],
  [
    "Index types: range, full-text, vector",
    <>Range covers exact match and comparison; full-text brings stemming, stopwords and TF-IDF scoring; vector supports nearest-neighbour on euclidean or cosine.</>,
    <code>CALL db.indexes()</code>,
  ],
  [
    <>Constraint types: <code>UNIQUE</code> and <code>MANDATORY</code>, on nodes or relationships</>,
    <>A unique constraint requires a supporting exact-match index to exist first. Creation is asynchronous: <code>PENDING</code> → <code>UNDER CONSTRUCTION</code> → <code>OPERATIONAL</code> or <code>FAILED</code>.</>,
    <code>CALL db.constraints()</code>,
  ],
  [
    "Property values are scalars plus geospatial and temporal types",
    <>String, boolean, 64-bit integer, 64-bit double, point, Date, Time, DateTime, Duration. <strong><code>null</code> cannot be stored as a property value</strong> — absence is the only way to express it.</>,
    "Write a probe node",
  ],
  [
    "Nested structures are not a design target",
    <>Do not build a model that depends on lists or maps as property values. Validate the behaviour of your version before relying on it.</>,
    "Version release notes",
  ],
];

export default function Assumptions() {
  return (
    <Band id="assumptions" aud="eng ops">
      <SectionHead
        index="02 · Foundations"
        title="Platform assumptions"
        aud={["Data engineering", "Platform ops"]}
      >
        These shape everything below. Confirm each against your deployed FalkorDB version before
        ratifying the standard — a wrong assumption here invalidates rules three sections down.
      </SectionHead>

      <div className="mb-m">
        <Table head={["Assumption", "What it means for the standard", "Verify"]}>
          {ROWS.map((r, i) => (
            <tr key={i}>
              <td>{r[0]}</td>
              <td>{r[1]}</td>
              <td>{r[2]}</td>
            </tr>
          ))}
        </Table>
      </div>

      <Note kind="stop" eyebrow="Operational warning · constraints">
        <p>
          Adding a unique constraint to a graph that already contains duplicate values has, in at least
          some versions, caused instability during enforcement. Always run the duplicate-detection query
          (<a href="#ax-d">Appendix D</a>) and resolve conflicts <em>before</em> issuing{" "}
          <code>GRAPH.CONSTRAINT CREATE</code>, and never introduce a new constraint directly into
          production without exercising it on a restored snapshot first.
        </p>
      </Note>
    </Band>
  );
}
