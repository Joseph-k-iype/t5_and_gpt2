/* Small shared building blocks. Each one exists because it appears in at
   least three sections; anything used once stays inline where it is read. */

import { useDim } from "../lib/LensContext.jsx";

export function Band({ id, aud, tight, children, last }) {
  const dim = useDim(aud);
  return (
    <section
      id={id}
      className={"band" + (tight ? " band-tight" : "") + (last ? " no-b" : "") + (dim ? " " + dim : "")}
      data-aud={aud}
    >
      <span className="band-mark" aria-hidden="true"><b /></span>
      {children}
    </section>
  );
}

export function SectionHead({ index, title, aud, children }) {
  return (
    <div className="section-head" data-reveal>
      {aud && (
        <span className="aud">
          {aud.map((a) => (
            <b key={a}>{a}</b>
          ))}
        </span>
      )}
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
    <div className={`grid g${cols} full ${className}`} data-reveal-group {...rest}>
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
