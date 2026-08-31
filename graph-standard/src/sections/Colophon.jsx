import { useRef } from "react";
import { useMagnetic } from "../lib/motion.js";

export default function Colophon() {
  const btn = useRef(null);
  useMagnetic(btn, 0.24);

  return (
    <footer className="band band-tight no-b" id="colophon">
      <div className="grid g2 full items-end" data-reveal-group>
        <div>
          <span className="eyebrow">Next step</span>
          <h3 className="tight">Take this to the architecture forum with Appendix E answered</h3>
          <p className="small measure-s">
            Seven decisions, one hour. Everything else in this document can be adopted incrementally —
            those seven cannot, because each one is baked into the first registration that follows it.
          </p>
        </div>
        <div className="row-end">
          <button type="button" className="btn magnetic" ref={btn} onClick={() => window.print()}>
            Save as PDF
          </button>
          <a className="btn btn-ghost" href="#top">Back to top</a>
        </div>
      </div>
      <hr />
      <p className="small c-ink3">
        Graph Engineering Standard · FalkorDB estate · Draft v0.1 · Labelled property graph, OpenCypher ·
        Verify every platform assumption in §2 against your deployed version before ratification.
      </p>
    </footer>
  );
}
