import { useRef } from "react";
import { LENSES } from "../lib/nav.js";
import { useMagnetic } from "../lib/motion.js";

export default function Toolbar({ lens, setLens, onJump }) {
  const printRef = useRef(null);
  useMagnetic(printRef, 0.22);

  return (
    <div className="toolbar noprint" role="toolbar" aria-label="Reading controls">
      <div className="lens-row">
        <span className="tb-label">Read as</span>
        {LENSES.map((l) => (
          <button
            key={l.key}
            type="button"
            className="lens"
            data-lens={l.key}
            aria-pressed={lens === l.key}
            onClick={() => setLens(l.key)}
          >
            {l.label}
          </button>
        ))}
      </div>
      <div className="tb-actions">
        <button type="button" className="btn btn-ghost" onClick={onJump}>
          Jump to…
        </button>
        <button
          type="button"
          className="btn magnetic"
          ref={printRef}
          onClick={() => window.print()}
        >
          Save as PDF
        </button>
      </div>
    </div>
  );
}
