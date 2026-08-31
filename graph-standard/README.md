# Graph Engineering Standard — FalkorDB estate

A single-page reference for the graph and data engineering standards applied to
the FalkorDB estate: six pillars (modelling, ingestion contracts, naming and
identity, tenant lifecycle, observability, and the catalogue meta-graph), a
four-rung conformance ladder, and the working templates in the appendices.

Written for four audiences at once. The **Read as** filter at the top of the
page fades out everything that isn't relevant to business and governance, data
engineering, data science, or platform ops.

## Interactive parts

Everything below is a decision aid, not decoration — each one implements a rule
stated in the surrounding text, so the rule can be checked rather than debated.

| Tool | Section | What it does |
|---|---|---|
| Property-vs-node decider | §3.3 | Walks the four tests in order; the first "yes" makes the attribute a node |
| Convention checker | §5.2 | Validates a label, facet, relationship type, property key, graph key, index name or contract ID against §5.1 — the same rule set your CI linter should implement |
| Promotion checklists | §6.2 | Sandbox → incubation and incubation → production gates, remembered per browser |
| Conformance score model | §7.2 | The weighted score from §7.4, with the L0–L3 verdict |
| Meta-graph diagram | §8.2 | The catalogue model, drawn edge by edge |
| Jump palette | anywhere | <kbd>/</kbd> to open, <kbd>P</kbd> to print |

## Stack

- **React 18 + Vite 5** — one page, no router, no UI framework.
- **GSAP + ScrollTrigger** — the masthead entrance, scroll reveals, section
  progress spines, the diagram draw-in, magnetic buttons and the score dial.
- **Raw WebGL** (`src/shaders/`) — the masthead field is a domain-warped fbm
  resolved into survey contours and a graph lattice. No shader library; the
  palette is passed in as uniforms so it matches the CSS tokens exactly.
- **No CSS framework.** The design system lives in `src/styles/global.css` as
  custom properties: a cool paper ground, ink, one ultramarine accent, and
  semantic ink reserved for status.

Every animation checks `prefers-reduced-motion` and lands in its final state
instead of running. Nothing is hidden by CSS, so the page reads correctly if the
motion layer never executes.

## Commands

```bash
npm install
npm run dev            # local dev server
npm run build          # normal build → dist/
npm run build:single   # inlined build → dist-single/index.html
                       # plus dist-single/artifact.html, a body-fragment
                       # variant for hosts that supply their own <head>
```

`dist-single/index.html` is a standalone file: open it from disk, email it, or
serve it from anywhere. Use **Save as PDF** in the toolbar (or <kbd>P</kbd>) for
a print-formatted copy — the print stylesheet drops the navigation, un-dims the
audience filter and keeps panels from breaking across pages.

## Status

Draft v0.1, for the architecture forum. Appendix E lists the seven decisions the
draft deliberately does not make; each one is cheap to settle now and expensive
to change once anything is registered against it.

Verify every platform assumption in §2 against your deployed FalkorDB version
before ratifying — several rules downstream depend on constraint semantics and
index availability that vary by release.
