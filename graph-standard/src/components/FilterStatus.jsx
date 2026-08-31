import { AUDIENCE_LABEL } from "../lib/nav.js";

/**
 * Nielsen's first heuristic: when a filter is on, say exactly what it did and
 * put the way out in the same sentence.
 *
 * The "nothing hidden" case is real and has to be handled honestly — data
 * engineering is the widest audience this standard has, and every section is
 * written for it. Reporting "12 of 12 shown, the rest are hidden" would be a
 * lie the reader can see through, which costs more trust than the filter buys.
 */
export default function FilterStatus({ lens, shown, total, onReset }) {
  if (lens === "all") return null;
  const hidden = total - shown;
  const who = AUDIENCE_LABEL[lens];

  return (
    <div className="filter-status noprint" role="status">
      <span>
        {hidden === 0 ? (
          <>
            Every section of this standard is written for <b>{who}</b> — nothing is hidden. It is the
            widest audience the document has.
          </>
        ) : (
          <>
            <b className="num">{hidden}</b> of <b className="num">{total}</b> sections hidden. Showing the{" "}
            <b className="num">{shown}</b> written for <b>{who}</b>; the rest are addressed to other
            readers, not deleted.
          </>
        )}
      </span>
      <button type="button" className="btn btn-ghost btn-sm" onClick={onReset}>
        {hidden === 0 ? "Clear filter" : "Show everything"}
      </button>
    </div>
  );
}
