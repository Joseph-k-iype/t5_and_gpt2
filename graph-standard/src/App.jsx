import { useCallback, useEffect, useRef, useState } from "react";

import Rail from "./components/Rail.jsx";
import Toolbar from "./components/Toolbar.jsx";
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

import { LensContext } from "./lib/LensContext.jsx";
import { FLAT_NAV } from "./lib/nav.js";
import useLocalState from "./lib/useLocalState.js";
import { useReveal, useProgress, useBandMarks, useScrollRefresh } from "./lib/motion.js";

function useActiveSection() {
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
  }, []);

  return active;
}

export default function App() {
  const [lens, setLens] = useLocalState("gs.lens", "all");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const barRef = useRef(null);
  const mainRef = useRef(null);
  const active = useActiveSection();

  useProgress(barRef);
  useReveal(mainRef);
  useBandMarks(mainRef);
  useScrollRefresh();

  const openPalette = useCallback(() => setPaletteOpen(true), []);
  const closePalette = useCallback(() => setPaletteOpen(false), []);

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
      <div id="progress" ref={barRef} aria-hidden="true" />

      <div className="shell">
        <Rail active={active} />

        <main className="main" ref={mainRef}>
          <Toolbar lens={lens} setLens={setLens} onJump={openPalette} />
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
    </LensContext.Provider>
  );
}
