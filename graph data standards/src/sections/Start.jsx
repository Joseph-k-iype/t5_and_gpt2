import { Band, SectionHead } from "../components/Primitives.jsx";
import { useSetLens, useLens } from "../lib/LensContext.jsx";

const PATHS = [
  {
    aud: "biz",
    eyebrow: "If you run a business area",
    title: "You are accountable for what's in your graphs",
    body: "You will be named on a conformance report. What you need to know is which graphs belong to your domain, what data they hold, at what sensitivity, and whether the pipelines feeding them are governed or improvised.",
    steps: [
      <>Read <a href="#why">the three questions</a> — that&rsquo;s the whole case for the standard.</>,
      <>Read <a href="#p4-tiers">the three tiers</a> — it tells you what your team is signing up for.</>,
      <>Skim <a href="#roles">roles</a> and <a href="#ladder">the conformance ladder</a>.</>,
    ],
  },
  {
    aud: "eng",
    eyebrow: "If you build pipelines",
    title: "Your work is the ingestion contract",
    body: "Nearly every rule that will change your day-to-day is in pillars 2 and 3. The contract is a YAML file in source control; the governance properties are non-negotiable; idempotency is what makes recovery possible.",
    steps: [
      <><a href="#p2">Ingestion contracts</a> end to end, then <a href="#ax-a">Appendix A</a>.</>,
      <><a href="#p3-identity">Identity strategy</a> — pick A or B per label, and commit.</>,
      <><a href="#p2-gov">Governance properties</a> — every write, no exceptions.</>,
    ],
  },
  {
    aud: "sci",
    eyebrow: "If you model or analyse",
    title: "The model decides what you can ask",
    body: "The property-versus-node rule and the reification rule are the two that will constrain or liberate your work two years from now. If you are doing GraphRAG or embeddings, read the open decision on vectors before you build.",
    steps: [
      <><a href="#p1-decide">Property vs. node</a> — walk the interactive test.</>,
      <><a href="#p1-rels">Reification</a> — events, transactions and decisions are nodes.</>,
      <><a href="#p3-identity">Entity resolution</a> stays a pipeline concern, not a query concern.</>,
    ],
  },
  {
    aud: "ops",
    eyebrow: "If you keep it running",
    title: "You own the measurement, not the policy",
    body: "The catalogue's observed layer is the first thing to build, before any policy is ratified. You cannot govern an estate you have not measured, and the first inventory usually reframes everyone's priorities.",
    steps: [
      <><a href="#p6">The catalogue meta-graph</a> and its observed layer.</>,
      <><a href="#ax-d">Appendix D</a> — the measurements the daily job must produce.</>,
      <><a href="#p4-reap">Expiry and reaping</a> — the cure for sandbox sprawl.</>,
    ],
  },
];

function PathCard({ path }) {
  const setLens = useSetLens();
  const lens = useLens();
  const active = lens === path.aud;

  return (
    <div className="panel">
      <span className="eyebrow">{path.eyebrow}</span>
      <h3 className="tight">{path.title}</h3>
      <p className="small">{path.body}</p>
      <ol className="small">
        {path.steps.map((s, i) => <li key={i}>{s}</li>)}
      </ol>
      <button
        type="button"
        className={"btn btn-sm " + (active ? "" : "btn-ghost")}
        onClick={() => setLens(active ? "all" : path.aud)}
      >
        {active ? "Showing only these sections — undo" : "Hide everything else"}
      </button>
    </div>
  );
}

export default function Start() {
  return (
    <Band id="start" tight>
      <SectionHead index="00 · Orientation" title="Where to start">
        This is a long document because it has to be. Nobody needs to read all of it. Pick the path that
        matches your job, and use <strong>Hide everything else</strong> to cut the document down to the
        sections written for you — the filter at the top of the page does the same thing, and is always
        one click from putting everything back.
      </SectionHead>

      <div className="grid g2 full" data-reveal-group>
        {PATHS.map((p) => <PathCard key={p.aud} path={p} />)}
      </div>
    </Band>
  );
}
