import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import Rail from "./components/Rail.jsx";
import Toolbar from "./components/Toolbar.jsx";
import FilterStatus from "./components/FilterStatus.jsx";
import Palette from "./components/Palette.jsx";

import Masthead from "./sections/Masthead.jsx";
import Start from "./sections/Start.jsx";
import Why from "./sections/Why.jsx";
import Assumptions from "./sections/Assumptions.jsx";
import Modelling from "./sections/Modelling.jsx";
import Ingestion from "./sections/Ingestion.jsx";
import Identity from "./sections/Identity.jsx";
import Lifecycle from "./sections/Lifecycle.jsx";
import Observability from "./sections/Observability.jsx";
import Catalogue from "./sections/Catalogue.jsx";
import Roles from "./sections/Roles.jsx";
import Ladder from "./sections/Ladder.jsx";
import Appendix from "./sections/Appendix.jsx";
import Colophon from "./sections/Colophon.jsx";

import { LensContext, LensSetContext } from "./lib/LensContext.jsx";
import { NAV, FLAT_NAV, countVisible, isVisible } from "./lib/nav.js";
import useLocalState from "./lib/useLocalState.js";
import { useReveal, useProgress, useBandMarks, useScrollRefresh, ScrollTrigger } from "./lib/motion.js";

function useActiveSection(lens) {
  const [active, setActive] = useState("start");

  useEffect(() => {
    const targets = FLAT_NAV
      .map((e) => document.getElementById(e.id))
      .filter(Boolean);
    if (!targets.length) return;

    const io = new IntersectionObserver(
      (entries) => {
        const hit = entries.find((e) => e.isIntersecting);
        if (hit) setActive(hit.target.id);
      },
      { rootMargin: "-15% 0px -70% 0px", threshold: 0 }
    );
    targets.forEach((t) => io.observe(t));
    return () => io.disconnect();
  }, [lens]); // sections mount and unmount with the filter

  return active;
}

export default function App() {
  const [lens, setLens] = useLocalState("gs.lens", "all");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const [showTop, setShowTop] = useState(false);
  const barRef = useRef(null);
  const mainRef = useRef(null);
  const active = useActiveSection(lens);

  useProgress(barRef);
  useReveal(mainRef, lens);
  useBandMarks(mainRef, lens);
  useScrollRefresh();

  const { shown, total } = countVisible(lens);

  /* "You are here" — serial position and progress both need a location, not
     just a percentage bar. */
  const position = useMemo(() => {
    const visible = NAV.filter((s) => isVisible(s.id, lens));
    const idx = visible.findIndex(
      (s) => s.id === active || (s.sub || []).some((x) => x.id === active)
    );
    if (idx < 0) return null;
    return { num: idx + 1, total: visible.length, label: visible[idx].label };
  }, [active, lens]);

  const openPalette = useCallback(() => setPaletteOpen(true), []);
  const closePalette = useCallback(() => setPaletteOpen(false), []);

  const changeLens = useCallback((next) => {
    setLens(next);
    /* Layout changes underneath the reader, so put them somewhere sensible
       rather than wherever the old scroll offset happens to land. */
    requestAnimationFrame(() => {
      ScrollTrigger.refresh();
      const anchor = document.getElementById(next === "all" ? "top" : "start");
      anchor?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }, [setLens]);

  useEffect(() => {
    const onScroll = () => setShowTop(window.scrollY > window.innerHeight * 1.5);
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    const typing = (el) =>
      el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT" || el.isContentEditable);

    const onKey = (e) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      if (typing(document.activeElement)) return;
      if (e.key === "/") { e.preventDefault(); openPalette(); }
      else if (e.key === "p" || e.key === "P") { e.preventDefault(); window.print(); }
      else if (e.key === "Escape") closePalette();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [openPalette, closePalette]);

  return (
    <LensContext.Provider value={lens}>
    <LensSetContext.Provider value={changeLens}>
      <a className="skip" href="#start">Skip to the document</a>
      <div id="progress" ref={barRef} aria-hidden="true" />

      <div className="shell">
        <Rail active={active} position={position} />

        <main className="main" ref={mainRef}>
          <Toolbar lens={lens} setLens={changeLens} onJump={openPalette} />
          <FilterStatus lens={lens} shown={shown} total={total} onReset={() => changeLens("all")} />

          <Masthead />
          <Start />
          <Why />
          <Assumptions />
          <Modelling />
          <Ingestion />
          <Identity />
          <Lifecycle />
          <Observability />
          <Catalogue />
          <Roles />
          <Ladder />
          <Appendix />
          <Colophon />
        </main>
      </div>

      <Palette open={paletteOpen} onClose={closePalette} />

      {showTop && (
        <button
          type="button"
          className="to-top noprint"
          aria-label="Back to top"
          onClick={() => window.scrollTo({ top: 0, behavior: "smooth" })}
        >
          <svg width="16" height="16" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M8 13V3M3.5 7.5L8 3l4.5 4.5" fill="none" stroke="currentColor"
                  strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </button>
      )}
    </LensSetContext.Provider>
    </LensContext.Provider>
  );
}
