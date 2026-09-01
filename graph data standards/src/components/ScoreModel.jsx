import { useEffect, useRef, useState } from "react";
import { Tool } from "./Primitives.jsx";
import { gsap, prefersReduced } from "../lib/motion.js";

const ITEMS = [
  { k: "gov",  w: 20, label: "Nodes with the complete governance property set" },
  { k: "reg",  w: 20, label: "Labels registered in the catalogue" },
  { k: "dict", w: 15, label: "Property keys present in the dictionary" },
  { k: "uniq", w: 15, label: <>Labels with an <b>entityId</b> unique constraint operational</> },
  { k: "ends", w: 10, label: "Relationship types with declared endpoint pairs" },
  { k: "dup",  w: 10, label: "Freedom from duplicate entityId within label" },
  { k: "orph", w: 5,  label: "Freedom from orphan nodes" },
  { k: "sup",  w: 5,  label: "Freedom from supernodes over threshold" },
];

const CIRC = 2 * Math.PI * 52;
const START = Object.fromEntries(ITEMS.map((i) => [i.k, 100]));

function verdictFor(s) {
  if (s >= 90) return { c: "#1E6A4B", t: <><b>L3 — Enforced.</b> Production-ready on this measure. Recertify annually.</> };
  if (s >= 70) return { c: "#23348F", t: <><b>L2 — Governed.</b> Acceptable at incubation. Close the gaps before a production promotion.</> };
  if (s >= 50) return { c: "#8F5D00", t: <><b>L1 — Described.</b> The graph is understood but not yet governed. Contracts are the next step.</> };
  return { c: "#A32B1D", t: <><b>L0 — Inventoried.</b> Remediation plan required, with a named owner and a date.</> };
}

export default function ScoreModel() {
  const [vals, setVals] = useState(START);
  const arcRef = useRef(null);
  const numRef = useRef(null);
  const shown = useRef(100);

  const score = Math.round(
    ITEMS.reduce((sum, it) => sum + (vals[it.k] / 100) * it.w, 0)
  );
  const verdict = verdictFor(score);

  /* The dial eases to its new value rather than snapping, and the number
     counts with it — the same easing vocabulary as the rest of the page. */
  useEffect(() => {
    const arc = arcRef.current;
    const num = numRef.current;
    if (!arc || !num) return;
    if (prefersReduced()) {
      arc.setAttribute("stroke-dashoffset", String(CIRC * (1 - score / 100)));
      num.textContent = String(score);
      shown.current = score;
      return;
    }
    const obj = { v: shown.current };
    const tween = gsap.to(obj, {
      v: score,
      duration: 0.6,
      ease: "power2.out",
      onUpdate: () => {
        num.textContent = String(Math.round(obj.v));
        arc.setAttribute("stroke-dashoffset", String(CIRC * (1 - obj.v / 100)));
      },
      onComplete: () => { shown.current = score; },
    });
    return () => tween.kill();
  }, [score]);

  useEffect(() => {
    if (arcRef.current) arcRef.current.setAttribute("stroke", verdict.c);
  }, [verdict.c]);

  return (
    <Tool title="Conformance score model" hint="Weighted to 100">
      <div className="score-layout">
        <div id="score-inputs" className="stack-s">
          {ITEMS.map((it) => (
            <div className="score-row" key={it.k}>
              <div className="lab">
                <span className="txt">
                  {it.label}
                  <span className="w">w {it.w}</span>
                </span>
                <span className="val">{vals[it.k]}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={vals[it.k]}
                aria-label={typeof it.label === "string" ? it.label : it.k}
                onChange={(e) =>
                  setVals((p) => ({ ...p, [it.k]: Number(e.target.value) }))
                }
              />
            </div>
          ))}
        </div>

        <div className="score-readout">
          <div className="score-dial">
            <svg viewBox="0 0 120 120" width="150" height="150" aria-hidden="true">
              <circle cx="60" cy="60" r="52" fill="none" stroke="#E4E6DF" strokeWidth="9" />
              <circle
                ref={arcRef}
                cx="60" cy="60" r="52" fill="none" stroke="#23348F" strokeWidth="9"
                strokeLinecap="round"
                strokeDasharray={CIRC}
                strokeDashoffset="0"
                transform="rotate(-90 60 60)"
              />
            </svg>
            <div className="score-num">
              <span ref={numRef} className="num">100</span>
            </div>
          </div>
          <div>
            <span className="eyebrow">Verdict</span>
            <p className="small mt-s flat" role="status">{verdict.t}</p>
          </div>
          <button type="button" className="btn btn-ghost" onClick={() => setVals(START)}>
            Reset to 100
          </button>
        </div>
      </div>
    </Tool>
  );
}
