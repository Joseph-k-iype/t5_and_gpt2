import { useEffect, useMemo, useRef, useState } from "react";
import { FLAT_NAV } from "../lib/nav.js";

export default function Palette({ open, onClose }) {
  const [q, setQ] = useState("");
  const [sel, setSel] = useState(0);
  const inputRef = useRef(null);
  const listRef = useRef(null);

  const results = useMemo(() => {
    const needle = q.trim().toLowerCase();
    if (!needle) return FLAT_NAV;
    return FLAT_NAV.filter((e) => e.label.toLowerCase().includes(needle));
  }, [q]);

  useEffect(() => {
    if (open) {
      setQ("");
      setSel(0);
      const t = setTimeout(() => inputRef.current?.focus(), 20);
      return () => clearTimeout(t);
    }
  }, [open]);

  useEffect(() => {
    listRef.current?.querySelector(".sel")?.scrollIntoView({ block: "nearest" });
  }, [sel, results]);

  if (!open) return null;

  const go = (entry) => {
    onClose();
    const target = document.getElementById(entry.id);
    if (target) target.scrollIntoView({ behavior: "smooth", block: "start" });
    history.replaceState(null, "", `#${entry.id}`);
  };

  const onKeyDown = (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setSel((s) => Math.min(s + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSel((s) => Math.max(s - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (results[sel]) go(results[sel]);
    } else if (e.key === "Escape") {
      onClose();
    }
  };

  return (
    <div
      id="palette"
      className="noprint"
      onMouseDown={(e) => { if (e.target.id === "palette") onClose(); }}
    >
      <div className="pal-box" role="dialog" aria-modal="true" aria-label="Jump to section">
        <input
          id="pal-input"
          ref={inputRef}
          type="text"
          value={q}
          placeholder="Jump to a section…"
          autoComplete="off"
          spellCheck="false"
          onChange={(e) => { setQ(e.target.value); setSel(0); }}
          onKeyDown={onKeyDown}
        />
        <ul id="pal-list" ref={listRef} role="listbox">
          {results.map((e, i) => (
            <li
              key={e.id}
              role="option"
              aria-selected={i === sel}
              className={i === sel ? "sel" : undefined}
              onMouseEnter={() => setSel(i)}
              onClick={() => go(e)}
            >
              <i>{e.num}</i>
              <span>{e.label}</span>
            </li>
          ))}
          {!results.length && (
            <li aria-disabled="true"><span>No section matches that.</span></li>
          )}
        </ul>
      </div>
    </div>
  );
}
