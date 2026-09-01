import { NAV, isVisible } from "../lib/nav.js";
import { useLens } from "../lib/LensContext.jsx";

export default function Rail({ active, position }) {
  const lens = useLens();
  const sections = NAV.filter((s) => isVisible(s.id, lens));
  const openId = sections.find(
    (s) => s.id === active || (s.sub || []).some((x) => x.id === active)
  )?.id;

  return (
    <nav className="rail" aria-label="Contents">
      <div className="rail-mark">
        <svg width="26" height="26" viewBox="0 0 26 26" aria-hidden="true">
          <line x1="5" y1="6" x2="20" y2="6" stroke="#B9C1E4" strokeWidth="1" />
          <line x1="5" y1="6" x2="5" y2="20" stroke="#B9C1E4" strokeWidth="1" />
          <line x1="5" y1="20" x2="20" y2="6" stroke="#B9C1E4" strokeWidth="1" />
          <line x1="20" y1="6" x2="20" y2="20" stroke="#B9C1E4" strokeWidth="1" />
          <line x1="5" y1="20" x2="20" y2="20" stroke="#B9C1E4" strokeWidth="1" />
          <circle cx="5" cy="6" r="2.6" fill="#23348F" />
          <circle cx="20" cy="6" r="2.6" fill="#15181A" />
          <circle cx="5" cy="20" r="2.6" fill="#15181A" />
          <circle cx="20" cy="20" r="2.6" fill="#23348F" />
        </svg>
        <div>
          <b>Graph Standard</b>
          <span>FalkorDB estate</span>
        </div>
      </div>

      {position && (
        <div className="rail-pos" aria-live="polite">
          <span className="label">You are here</span>
          <span className="v">
            <b className="num">{position.num}</b>
            <span className="of">of {position.total}</span>
            {position.label}
          </span>
        </div>
      )}

      <div className="toc">
        <ol>
          {sections.map((s) => (
            <li key={s.id} className={openId === s.id ? "open" : undefined}>
              <a href={`#${s.id}`} className={active === s.id ? "on" : undefined}>
                <i>{s.num}</i>
                <span>{s.label}</span>
              </a>
              {s.sub && (
                <ol className="sub">
                  {s.sub.map((x) => (
                    <li key={x.id}>
                      <a href={`#${x.id}`} className={active === x.id ? "on" : undefined}>
                        {x.label}
                      </a>
                    </li>
                  ))}
                </ol>
              )}
            </li>
          ))}
        </ol>
      </div>

      <div className="rail-foot">
        <kbd>/</kbd> jump to section &nbsp;·&nbsp; <kbd>P</kbd> save as PDF
      </div>
    </nav>
  );
}
