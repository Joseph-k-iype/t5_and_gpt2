import { Band, SectionHead, Note, Table } from "../components/Primitives.jsx";

const ROWS = [
  [
    "Query language is OpenCypher with FalkorDB extensions",
    <>Every rule here is expressed against a labelled property graph. Which introspection procedures exist varies by version.</>,
    "Read the release notes for your deployed version",
  ],
  [
    "A tenant is a graph key on an instance",
    <>Isolation is at the graph level, not the schema level. <strong>There is no cross-graph query.</strong> Any federation happens in the application layer, and reference data must be replicated rather than joined.</>,
    "List the graphs on the instance and confirm none can reference another",
  ],
  [
    "Index types: range, full-text, vector",
    <>Range covers exact match and comparison; full-text brings stemming, stopwords and relevance scoring; vector supports nearest-neighbour on euclidean or cosine.</>,
    "Read the index catalogue for a known graph",
  ],
  [
    <>Constraint types: <strong>unique</strong> and <strong>mandatory</strong>, on nodes or relationships</>,
    <>A unique constraint requires a supporting exact-match index to exist first. Creation is asynchronous and moves through <em>pending</em> and <em>under construction</em> before reaching <em>operational</em> — or <em>failed</em>, if existing data conflicts.</>,
    "Read the constraint catalogue and confirm the status values it reports",
  ],
  [
    "Property values are scalars plus geospatial and temporal types",
    <>String, boolean, 64-bit integer, 64-bit double, point, date, time, datetime, duration. <strong>Null cannot be stored as a property value</strong> — absence is the only way to express it.</>,
    "Write a probe node carrying one property of each type",
  ],
  [
    "Nested structures are not a design target",
    <>Do not build a model that depends on lists or maps as property values. Validate the behaviour of your version before relying on it.</>,
    "Version release notes",
  ],
];

export default function Assumptions() {
  return (
    <Band id="assumptions">
      <SectionHead index="02 · Foundations" title="Platform assumptions">
        These shape everything below. Confirm each against your deployed FalkorDB version before
        ratifying the standard — a wrong assumption here invalidates rules three sections down.
      </SectionHead>

      <div className="mb-m">
        <Table head={["Assumption", "What it means for the standard", "How to confirm it"]}>
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
          some versions, caused instability during enforcement. Always find and resolve the duplicates
          first (<a href="#ax-d">Appendix D</a> specifies the check), and never introduce a new constraint
          directly into production without exercising it on a restored snapshot.
        </p>
      </Note>
    </Band>
  );
}
