import { useState } from "react";
import { Tool } from "./Primitives.jsx";

const TESTS = [
  {
    q: "Does it need its own relationships?",
    h: "Will you ever want to connect it to something else — a hierarchy, a mapping, another entity?",
    yes: "It has to be able to sit at the end of an edge. A string cannot; a node can.",
  },
  {
    q: "Does it have an independent lifecycle?",
    h: "Is it created, versioned, expired or approved on a different clock from its parent?",
    yes: "It changes on its own schedule, so it needs its own identity and its own history.",
  },
  {
    q: "Is it a shared controlled vocabulary reused across entities?",
    h: "Country, currency, product code, business unit — reference data used by more than one label.",
    yes: "One authoritative definition, and traversals can pivot through it to find everything that shares the value.",
  },
  {
    q: "Do you need to find all parents by this value, cheaply and often?",
    h: "A repeated global filter across the whole graph, many times a day.",
    yes: "Incoming edges beat a scan — though check first whether a range index solves it more cheaply.",
  },
];

export default function Decider() {
  const [attr, setAttr] = useState("country");
  const [step, setStep] = useState(0);
  const [trace, setTrace] = useState([]);
  const [verdict, setVerdict] = useState(null);

  const name = attr.trim() || "this attribute";

  const restart = () => { setStep(0); setTrace([]); setVerdict(null); };

  const answer = (yes) => {
    const t = TESTS[step];
    const line = `Test ${step + 1}: ${yes ? "yes" : "no"} — ${t.q.toLowerCase()}`;
    setTrace((prev) => [...prev, line]);
    if (yes) {
      setVerdict({ kind: "node", why: t.yes });
    } else if (step + 1 >= TESTS.length) {
      setVerdict({ kind: "prop" });
    } else {
      setStep(step + 1);
    }
  };

  return (
    <Tool title="Decide: property or node?" hint="Answer for one attribute at a time" id="decider">
      <div className="attr-row">
        <label className="fld fld-attr">
          <span className="k">Attribute you&rsquo;re deciding about</span>
          <input
            type="text"
            value={attr}
            spellCheck="false"
            aria-label="Attribute name"
            onChange={(e) => { setAttr(e.target.value); if (verdict) restart(); }}
          />
        </label>
        <div className="end">
          <button type="button" className="btn btn-ghost" onClick={restart}>Restart</button>
        </div>
      </div>

      {verdict ? (
        <div className={"dec-verdict " + verdict.kind}>
          {verdict.kind === "node" ? (
            <>
              <h4>{name} is a node</h4>
              <p>
                {verdict.why} Model it as its own label with a registered <code>entityId</code>, and
                connect it with a relationship type registered against both endpoint labels.
              </p>
            </>
          ) : (
            <>
              <h4>{name} stays a property</h4>
              <p>
                It failed all four tests: high-cardinality, single-use, non-relational, descriptive.
                Leaving it as a property is the correct answer, not a compromise — every promotion to a
                node adds a traversal hop to every query that touches it. If you filter on it often, add
                a range index.
              </p>
            </>
          )}
        </div>
      ) : (
        <div className="dec-q">
          <span className="step">Test {step + 1} of {TESTS.length}</span>
          <h4>{TESTS[step].q}</h4>
          <p>{TESTS[step].h}</p>
          <div className="dec-actions">
            <button type="button" className="yes" onClick={() => answer(true)}>Yes</button>
            <button type="button" className="no" onClick={() => answer(false)}>No</button>
          </div>
        </div>
      )}

      <ol id="dec-trace" className="small">
        {trace.map((t) => <li key={t}>{t}</li>)}
      </ol>
    </Tool>
  );
}
