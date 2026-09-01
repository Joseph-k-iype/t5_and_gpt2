import { useRef } from "react";
import { AUDIENCES } from "../lib/nav.js";
import { useMagnetic } from "../lib/motion.js";

/**
 * One sticky bar: the audience filter (a labelled segmented control, so it
 * reads as a control rather than as decoration) and the two actions. Exposed
 * as a radiogroup because exactly one audience applies at a time.
 * "You are here" lives in the rail, next to the contents it refers to.
 */
export default function Toolbar({ lens, setLens, onJump }) {
  const printRef = useRef(null);
  useMagnetic(printRef, 0.2);

  return (
    <div className="toolbar noprint">
      <div className="filter" role="radiogroup" aria-label="Filter the standard by audience">
        <span className="tb-label" aria-hidden="true">Reading for</span>
        {AUDIENCES.map((a) => (
          <button
            key={a.key}
            type="button"
            role="radio"
            className="lens"
            data-lens={a.key}
            aria-checked={lens === a.key}
            aria-label={`Read the sections written for ${a.label}`}
            title={a.label}
            onClick={() => setLens(a.key)}
          >
            {a.short}
          </button>
        ))}
      </div>

      {/* Below the chip group's usable width the same control becomes a native
          select: one line, always shows the current value, and opens the whole
          list on tap instead of hiding four options off the edge. */}
      <label className="filter-select">
        <span className="tb-label">Reading for</span>
        <select value={lens} onChange={(e) => setLens(e.target.value)} aria-label="Filter the standard by audience">
          {AUDIENCES.map((a) => (
            <option key={a.key} value={a.key}>{a.label}</option>
          ))}
        </select>
      </label>

      <span className="tb-spacer" />

      <div className="tb-actions">
        <button type="button" className="btn btn-ghost" onClick={onJump}>
          Jump to&hellip;
        </button>
        <button type="button" className="btn magnetic print-btn" ref={printRef} onClick={() => window.print()}>
          Save as PDF
        </button>
      </div>
    </div>
  );
}
