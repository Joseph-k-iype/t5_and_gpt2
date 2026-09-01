import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";

gsap.registerPlugin(ScrollTrigger);

export const prefersReduced = () =>
  typeof window !== "undefined" &&
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* A single easing vocabulary keeps the whole page feeling like one object. */
export const EASE = {
  out: "power3.out",
  inOut: "power2.inOut",
  mask: "expo.out",
};

gsap.defaults({ ease: EASE.out, duration: 0.9 });

/**
 * Masthead entrance. Lines rise out of their own overflow masks, then the
 * supporting matter settles behind them. Runs once, on load.
 */
export function useHeroTimeline(scopeRef) {
  useEffect(() => {
    const el = scopeRef.current;
    if (!el) return;
    const ctx = gsap.context(() => {
      if (prefersReduced()) {
        gsap.set("[data-hero]", { opacity: 1, y: 0 });
        gsap.set(".mh-line > span", { yPercent: 0 });
        return;
      }
      const tl = gsap.timeline({ defaults: { ease: EASE.mask } });
      tl.from(".mh-tag", { opacity: 0, y: 12, duration: 0.7, ease: EASE.out })
        .from(".mh-line > span", { yPercent: 108, duration: 1.15, stagger: 0.085 }, "-=0.35")
        .from(".mh-lede", { opacity: 0, y: 16, duration: 0.8, ease: EASE.out }, "-=0.75")
        .from(".mh-meta > div", { opacity: 0, y: 14, duration: 0.7, stagger: 0.06, ease: EASE.out }, "-=0.6")
        .from(".mh-rule", { scaleX: 0, transformOrigin: "left center", duration: 1.0 }, "-=0.9")
        .from(".cue", { opacity: 0, duration: 0.6 }, "-=0.4");

      gsap.to(".cue .track b", {
        xPercent: 190, duration: 1.6, repeat: -1, ease: "power1.inOut", yoyo: true,
      });
    }, el);
    return () => ctx.revert();
  }, [scopeRef]);
}

/**
 * Scroll reveals. Anything marked data-reveal rises into place once; groups
 * marked data-reveal-group stagger their direct children.
 * Elements are only hidden from JS, so the page reads fine without it.
 */
export function useReveal(scopeRef, dep) {
  useEffect(() => {
    const root = scopeRef?.current || document.body;
    const ctx = gsap.context(() => {
      if (prefersReduced()) return;

      gsap.utils.toArray("[data-reveal]").forEach((node) => {
        gsap.from(node, {
          opacity: 0, y: 24, duration: 0.85,
          scrollTrigger: { trigger: node, start: "top 88%", once: true },
        });
      });

      gsap.utils.toArray("[data-reveal-group]").forEach((group) => {
        const kids = gsap.utils.toArray(group.children);
        if (!kids.length) return;
        gsap.from(kids, {
          opacity: 0, y: 26, duration: 0.8, stagger: 0.07,
          scrollTrigger: { trigger: group, start: "top 86%", once: true },
        });
      });

      gsap.utils.toArray("[data-reveal-rows] tbody tr").forEach((row, i) => {
        gsap.from(row, {
          opacity: 0, duration: 0.5, delay: Math.min(i, 8) * 0.03,
          scrollTrigger: { trigger: row, start: "top 96%", once: true },
        });
      });
    }, root);
    return () => ctx.revert();
  }, [scopeRef, dep]);
}

/** Reading progress, driven by ScrollTrigger rather than a scroll listener. */
export function useProgress(barRef) {
  useEffect(() => {
    const bar = barRef.current;
    if (!bar) return;
    const st = ScrollTrigger.create({
      start: 0,
      end: "max",
      onUpdate: (self) => { bar.style.width = (self.progress * 100).toFixed(2) + "%"; },
    });
    return () => st.kill();
  }, [barRef]);
}

/** Each band carries a hairline that fills as you read through it. */
export function useBandMarks(scopeRef, dep) {
  useEffect(() => {
    const root = scopeRef?.current || document.body;
    const ctx = gsap.context(() => {
      gsap.utils.toArray(".band-mark b").forEach((mark) => {
        gsap.to(mark, {
          scaleY: 1, ease: "none",
          scrollTrigger: {
            trigger: mark.closest(".band"),
            start: "top 70%", end: "bottom 70%", scrub: 0.4,
          },
        });
      });
    }, root);
    return () => ctx.revert();
  }, [scopeRef, dep]);
}

/** Pointer magnetism on the primary actions — small, and only on fine pointers. */
export function useMagnetic(ref, strength = 0.28) {
  useEffect(() => {
    const el = ref.current;
    if (!el || prefersReduced()) return;
    if (!window.matchMedia("(pointer: fine)").matches) return;

    const xTo = gsap.quickTo(el, "x", { duration: 0.5, ease: EASE.out });
    const yTo = gsap.quickTo(el, "y", { duration: 0.5, ease: EASE.out });

    const move = (e) => {
      const r = el.getBoundingClientRect();
      xTo((e.clientX - (r.left + r.width / 2)) * strength);
      yTo((e.clientY - (r.top + r.height / 2)) * strength);
    };
    const leave = () => { xTo(0); yTo(0); };

    el.addEventListener("pointermove", move);
    el.addEventListener("pointerleave", leave);
    return () => {
      el.removeEventListener("pointermove", move);
      el.removeEventListener("pointerleave", leave);
    };
  }, [ref, strength]);
}

/** Count a number up when it scrolls into view. */
export function useCountUp(ref, value, { suffix = "", duration = 1.4 } = {}) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    if (prefersReduced()) { el.textContent = value.toLocaleString() + suffix; return; }
    const obj = { v: 0 };
    const tween = gsap.to(obj, {
      v: value, duration, ease: "power2.out",
      onUpdate: () => { el.textContent = Math.round(obj.v).toLocaleString() + suffix; },
      scrollTrigger: { trigger: el, start: "top 92%", once: true },
    });
    return () => { tween.scrollTrigger?.kill(); tween.kill(); };
  }, [ref, value, suffix, duration]);
}

/** The catalogue meta-graph draws itself: edges stroke on, then the nodes land. */
export function useDiagramDraw(ref) {
  useEffect(() => {
    const svg = ref.current;
    if (!svg) return;
    const ctx = gsap.context(() => {
      const edges = gsap.utils.toArray(".dg-edges > *", svg);
      edges.forEach((e) => {
        const len = typeof e.getTotalLength === "function" ? e.getTotalLength() : 400;
        gsap.set(e, { strokeDasharray: len, strokeDashoffset: len });
      });
      if (prefersReduced()) {
        gsap.set(edges, { strokeDashoffset: 0 });
        return;
      }
      /* Arrowheads sit at the end of each path and would otherwise float in
         mid-air while the line is still drawing, so they stay off until the
         edges land. */
      svg.classList.add("drawing");
      const tl = gsap.timeline({
        scrollTrigger: { trigger: svg, start: "top 78%", once: true },
        onComplete: () => svg.classList.remove("drawing"),
      });
      tl.to(edges, { strokeDashoffset: 0, duration: 0.9, stagger: 0.028, ease: "power2.inOut" })
        .add(() => svg.classList.remove("drawing"), "-=0.15")
        .from(".dg-nodes > g", { opacity: 0, scale: 0.94, transformOrigin: "center", duration: 0.45, stagger: 0.035 }, "-=0.75")
        .from(".dg-lbls text", { opacity: 0, duration: 0.35, stagger: 0.018 }, "-=0.45");
    }, svg);
    return () => ctx.revert();
  }, [ref]);
}

/** Refresh ScrollTrigger after late layout shifts (webfonts, images). */
export function useScrollRefresh() {
  useEffect(() => {
    const refresh = () => ScrollTrigger.refresh();
    if (document.fonts?.ready) document.fonts.ready.then(refresh);
    const t = setTimeout(refresh, 900);
    window.addEventListener("load", refresh);
    return () => { clearTimeout(t); window.removeEventListener("load", refresh); };
  }, []);
}

export { gsap, ScrollTrigger };
