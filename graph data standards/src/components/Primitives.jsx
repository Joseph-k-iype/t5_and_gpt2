/* Small shared building blocks. Each one exists because it appears in at
   least three sections; anything used once stays inline where it is read. */

import { useVisibleSection } from "../lib/LensContext.jsx";

export function Band({ id, tight, children, last }) {
  /* The audience a section is written for lives in nav.js, so the rail, the
     counter and the section itself can never disagree about it. */
  if (!useVisibleSection(id)) return null;
  return (
    <section
      id={id}
      className={"band" + (tight ? " band-tight" : "") + (last ? " no-b" : "")}
    >
      <span className="band-mark" aria-hidden="true"><b /></span>
      {children}
    </section>
  );
}

export function SectionHead({ index, title, children }) {
  return (
    <div className="section-head" data-reveal>
      <span className="eyebrow">{index}</span>
      <h2>{title}</h2>
      {children && <p className="lede">{children}</p>}
    </div>
  );
}

export function Note({ kind = "", eyebrow, children }) {
  return (
    <div className={"note " + kind} data-reveal>
      <span className="eyebrow">{eyebrow}</span>
      {children}
    </div>
  );
}

export function Panel({ className = "", children, ...rest }) {
  return (
    <div className={"panel " + className} {...rest}>
      {children}
    </div>
  );
}

export function Table({ head, children, className = "" }) {
  return (
    <div className="tw" data-reveal data-reveal-rows>
      <table className={className}>
        <thead>
          <tr>
            {head.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Grid({ cols = 2, className = "", children, ...rest }) {
  return (
    <div className={`grid g${cols} full ${className}`.trim()} data-reveal-group {...rest}>
      {children}
    </div>
  );
}

export function Tool({ title, hint, children, ...rest }) {
  return (
    <div className="tool" data-reveal {...rest}>
      <div className="tool-head">
        <h4>{title}</h4>
        {hint && <span className="hint">{hint}</span>}
      </div>
      <div className="tool-body">{children}</div>
    </div>
  );
}

/**
 * The shape an identifier must take, with the variable parts marked.
 * Deliberately not a code block: nobody executes this, they pattern-match it.
 */
export function Specimen({ parts, example }) {
  return (
    <div className="specimen" data-reveal>
      <div>
        {parts.map((p, i) =>
          typeof p === "string" ? (
            <span key={i}>{p}</span>
          ) : (
            <span key={i} className="var">{p.var}</span>
          )
        )}
      </div>
      {example && <span className="eg">e.g. {example}</span>}
    </div>
  );
}
