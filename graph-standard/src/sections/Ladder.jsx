import { Band, SectionHead } from "../components/Primitives.jsx";

const RUNGS = [
  {
    lv: "L0", h: "Inventoried", req: "Sandbox minimum",
    p: "The graph exists in the catalogue with a named owner and a TTL. Nothing else. From a standing start this alone is a large improvement — and it is all the shared sandbox ever needs.",
  },
  {
    lv: "L1", h: "Described", req: "",
    p: <>Labels, relationship types and property keys registered; classification agreed. You can now answer <em>what is in it</em> without opening a query console.</>,
  },
  {
    lv: "L2", h: "Governed", req: "Incubation minimum",
    p: "Every source has an ingestion contract; governance properties present on every node and edge; identity strategy declared per label.",
  },
  {
    lv: "L3", h: "Enforced", req: "Production, within grace period",
    p: "Constraints operational; conformance score above threshold; freshness monitored; retention implemented and exercised.",
  },
];

const PHASES = [
  ["Weeks 1–3", "Measure before you govern", <>Stand up the catalogue graph and the observed-layer introspection job. Do this <em>before</em> writing any policy — you cannot govern an estate you have not measured, and the first inventory usually reframes the priorities.</>],
  ["Weeks 3–6", "Ratify naming and the modelling rule", "Agree the conventions and the property-versus-node rule. Publish the property key dictionary seeded from what is already observed, so it starts as a description rather than a wish."],
  ["Weeks 5–10", "Build the templates, pilot with two teams", "Registry and contract templates. Onboard two willing teams and fix the templates based on what they actually hit."],
  ["Weeks 8–14", "Backfill and publish", "Backfill registrations for existing production tenants. Publish the first conformance report — visibility, not enforcement."],
  ["Weeks 12–20", "Turn on enforcement", "Constraints, rejection rules, lifecycle automation. Enforce on new tenants first, then migrate existing ones on a published schedule."],
];

export default function Ladder() {
  return (
    <Band id="ladder" aud="biz ops eng">
      <SectionHead
        index="10 · Adoption"
        title="The conformance ladder"
        aud={["Business & governance", "Platform ops", "Data engineering"]}
      >
        Adopt in stages rather than declaring everything non-compliant on day one. A standard that makes
        every existing team a violator on publication day gets ignored on publication day.
      </SectionHead>

      <div className="ladder full" data-reveal-group>
        {RUNGS.map((r) => (
          <div className="rung" key={r.lv}>
            <span className="lv">{r.lv}</span>
            <div>
              <h4>{r.h}</h4>
              <p className="small flat mt-s">{r.p}</p>
            </div>
            <span className="req">{r.req}</span>
          </div>
        ))}
      </div>

      <h3 className="mt-l mb-s" data-reveal>Suggested sequence</h3>
      <p data-reveal>
        Five overlapping phases. The ordering matters more than the dates: measurement precedes policy,
        pilots precede rollout, and enforcement comes last.
      </p>

      <ol className="phases full" data-reveal-group>
        {PHASES.map(([wk, h, p]) => (
          <li key={wk}>
            <span className="wk">{wk}</span>
            <div>
              <h4>{h}</h4>
              <p className="small flat mt-s">{p}</p>
            </div>
          </li>
        ))}
      </ol>
    </Band>
  );
}
