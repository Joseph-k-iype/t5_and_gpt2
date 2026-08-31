import { useRef } from "react";
import FieldCanvas from "../shaders/FieldCanvas.jsx";
import { useHeroTimeline } from "../lib/motion.js";

const META = [
  ["Scope", "Sandbox · Incubation · Production"],
  ["Model", "Labelled Property Graph"],
  ["Language", "OpenCypher"],
  ["Isolation", "One graph key per tenant"],
  ["Status", "Awaiting architecture forum"],
];

export default function Masthead() {
  const ref = useRef(null);
  useHeroTimeline(ref);

  return (
    <header className="masthead" id="top" ref={ref}>
      <FieldCanvas intensity={1} />

      <div className="mh-inner">
        <span className="mh-tag">Draft v0.1 · for review</span>

        <h1 className="mh-title">
          <span className="mh-line"><span>Graph</span></span>
          <span className="mh-line"><span>Engineering</span></span>
          <span className="mh-line"><span>Standard</span></span>
          <span className="mh-line mh-line--sm"><span>for the FalkorDB estate</span></span>
        </h1>

        <p className="lede mh-lede">
          Today the platform is infrastructure: any team can point any source at any graph and model it
          however they like. This standard turns that into a governed estate — six pillars, a conformance
          ladder, and a catalogue that knows what exists.
        </p>

        <div className="mh-meta">
          <span className="mh-rule" aria-hidden="true" />
          {META.map(([k, v]) => (
            <div key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ))}
        </div>

        <div className="cue">
          <span className="track"><b /></span>
          Scroll · 11 sections · about 40 minutes
        </div>
      </div>
    </header>
  );
}
