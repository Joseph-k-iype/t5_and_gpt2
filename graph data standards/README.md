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
| Audience filter | toolbar | Hides the sections written for other readers, and says how many it hid. Chips on desktop, a native select below 720px |
| Property-vs-node decider | §3.3 | Walks the four tests in order; the first "yes" makes the attribute a node |
| Reification diagram | §3.4 | Why an edge with properties is a dead end, and what a first-class event node buys |
| Convention checker | §5.2 | Validates a label, facet, relationship type, property key, graph key, index name or contract ID against §5.1 — and offers the conforming form of whatever you typed |
| Promotion checklists | §6.3 | Sandbox → incubation and incubation → production gates, with progress, remembered per browser |
| Conformance score model | §7.2 | The weighted score from §7.4, with the L0–L3 verdict |
| Meta-graph diagram | §8.2 | The catalogue model, drawn edge by edge |
| Jump palette | anywhere | <kbd>/</kbd> to open, <kbd>P</kbd> to print |

## What this document does not contain

**No queries.** A standard states what must be true, what must be registered and
what must be measured. *How* each check is computed belongs to whoever builds
the catalogue job, against the platform version they actually have — and it
changes when that version does. Pinning implementations into the standard makes
the standard wrong at the next upgrade, and makes it unreadable for the
non-engineers who have to agree to it.

So §8.4 is a table of questions and the walks that answer them, Appendix D is a
measurement specification with cadences and alert conditions, and §4.3 is an
ordered write sequence. The YAML in the appendices stays: those are the
artefacts teams actually author and commit.

## Stack

- **React 18 + Vite 5** — one page, no router, no UI framework.
- **GSAP + ScrollTrigger** — the masthead entrance, scroll reveals, section
  progress spines, the diagram draw-in, magnetic buttons and the score dial.
- **Raw WebGL** (`src/shaders/`) — the masthead field is a domain-warped fbm
  resolved into survey contours and a graph lattice. No shader library; the
  palette is passed in as uniforms so it matches the CSS tokens exactly.
- **No CSS framework.** The design system lives in `src/styles/global.css` as
  custom properties: a warm-neutral paper ground, ink, one signal-red accent,
  and separate status inks that are never the only carrier of meaning.

### Design constraints the stylesheet holds to

- **One type scale.** Nine size tokens from a 17px body at a 1.25 ratio.
  Nothing sets a raw `font-size`; the only relative sizes are inline code
  (`0.84em`, so it tracks whatever it sits in) and the masthead's accent line.
  Every uppercase label in the document resolves to one treatment.
- **Contrast measured, not guessed.** Against the paper ground: ink 16.1:1,
  secondary 7.1:1, tertiary 5.0:1, accent 6.2:1, white-on-accent 6.9:1. The
  one token below 3:1 is used for borders and marks and never for text.
- **Nothing interactive under 40px**, and no fixed column counts that reflow a
  four-card group into three-plus-one.
- **Filtering hides, it does not fade.** Faded text is unreadable text, and a
  half-visible section reads as broken rather than as filtered out.

Every animation checks `prefers-reduced-motion` and lands in its final state
instead of running. Nothing is hidden by CSS, so the page reads correctly if the
motion layer never executes.

## Commands

The folder name contains spaces, so quote it when changing into it:
`cd "graph data standards"`.

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
