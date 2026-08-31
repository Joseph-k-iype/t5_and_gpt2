/* One source of truth for the document's structure and its audiences.
   The rail, the jump palette, the audience filter and the section counter all
   read from here, so they can never disagree about what exists. */

/* `short` is what the chip shows; `label` is what the reader is told
   everywhere it matters — the status strip, the role cards, the accessible
   name. The full names do not fit the bar without turning it into a
   horizontal scroller, and a control you have to scroll is a control most
   people never finish reading. */
export const AUDIENCES = [
  { key: "all", label: "Everything", short: "Everything" },
  { key: "biz", label: "Business & governance", short: "Business" },
  { key: "eng", label: "Data engineering", short: "Engineering" },
  { key: "sci", label: "Data science", short: "Science" },
  { key: "ops", label: "Platform ops", short: "Ops" },
];

export const AUDIENCE_LABEL = Object.fromEntries(
  AUDIENCES.map((a) => [a.key, a.label])
);

/* aud: null means the section is written for everyone and never filters out. */
export const NAV = [
  { id: "start", num: "00", label: "Where to start", aud: null },
  { id: "why", num: "01", label: "The three questions", aud: null },
  { id: "assumptions", num: "02", label: "Platform assumptions", aud: "eng ops" },
  {
    id: "p1", num: "03", label: "Graph modelling", aud: "eng sci",
    sub: [
      { id: "p1-registry", label: "Label registry" },
      { id: "p1-schema", label: "Property schema" },
      { id: "p1-decide", label: "Property vs. node" },
      { id: "p1-rels", label: "Relationships & reification" },
      { id: "p1-labels", label: "Multi-label discipline" },
      { id: "p1-structure", label: "Structural rules" },
      { id: "p1-change", label: "Changing a live model" },
    ],
  },
  {
    id: "p2", num: "04", label: "Ingestion contracts", aud: "eng biz ops",
    sub: [
      { id: "p2-gate", label: "The gate" },
      { id: "p2-fields", label: "Contract fields" },
      { id: "p2-write", label: "Write-path rules" },
      { id: "p2-gov", label: "Governance properties" },
      { id: "p2-reject", label: "Rejection rules" },
    ],
  },
  {
    id: "p3", num: "05", label: "Naming & identity", aud: "eng sci ops",
    sub: [
      { id: "p3-naming", label: "Conventions" },
      { id: "p3-linter", label: "Convention checker" },
      { id: "p3-drift", label: "Property key drift" },
      { id: "p3-identity", label: "Identity strategy" },
    ],
  },
  {
    id: "p4", num: "06", label: "Tenant lifecycle", aud: "biz ops eng",
    sub: [
      { id: "p4-tiers", label: "Three tiers" },
      { id: "p4-access", label: "Access & sensitivity" },
      { id: "p4-promote", label: "Promotion gates" },
      { id: "p4-reap", label: "Expiry & reaping" },
      { id: "p4-sandbox", label: "Shared sandbox" },
    ],
  },
  {
    id: "p5", num: "07", label: "Observability", aud: "ops eng biz",
    sub: [
      { id: "p5-metrics", label: "What we measure" },
      { id: "p5-score", label: "Conformance score" },
      { id: "p5-lineage", label: "Lineage answers" },
    ],
  },
  {
    id: "p6", num: "08", label: "Knowledge catalogue", aud: null,
    sub: [
      { id: "p6-layers", label: "Observed vs. declared" },
      { id: "p6-model", label: "Meta-graph model" },
      { id: "p6-queries", label: "What it answers" },
    ],
  },
  { id: "roles", num: "09", label: "Roles & accountability", aud: null },
  { id: "ladder", num: "10", label: "Conformance ladder", aud: "biz ops eng" },
  {
    id: "appendix", num: "11", label: "Appendices", aud: "eng ops sci",
    sub: [
      { id: "ax-a", label: "A · Contract template" },
      { id: "ax-b", label: "B · Label template" },
      { id: "ax-c", label: "C · Relationship template" },
      { id: "ax-d", label: "D · Measurement spec" },
      { id: "ax-e", label: "E · Open decisions" },
    ],
  },
];

export const SECTION_AUD = Object.fromEntries(NAV.map((s) => [s.id, s.aud]));

export function isVisible(sectionId, lens) {
  if (lens === "all") return true;
  const aud = SECTION_AUD[sectionId];
  if (!aud) return true; // written for everyone
  return aud.split(/\s+/).includes(lens);
}

export function countVisible(lens) {
  const shown = NAV.filter((s) => isVisible(s.id, lens)).length;
  return { shown, total: NAV.length };
}

export const FLAT_NAV = NAV.flatMap((s) => [
  { id: s.id, num: s.num, label: s.label, parent: s.id },
  ...(s.sub || []).map((x) => ({ id: x.id, num: "·", label: x.label, parent: s.id })),
]);
