export const NAV = [
  { id: "start", num: "00", label: "Where to start" },
  { id: "why", num: "01", label: "The three questions" },
  { id: "assumptions", num: "02", label: "Platform assumptions" },
  {
    id: "p1", num: "03", label: "Graph modelling",
    sub: [
      { id: "p1-registry", label: "Label registry" },
      { id: "p1-schema", label: "Property schema" },
      { id: "p1-decide", label: "Property vs. node" },
      { id: "p1-rels", label: "Relationship properties" },
      { id: "p1-labels", label: "Multi-label discipline" },
      { id: "p1-structure", label: "Structural rules" },
    ],
  },
  {
    id: "p2", num: "04", label: "Ingestion contracts",
    sub: [
      { id: "p2-gate", label: "The gate" },
      { id: "p2-fields", label: "Contract fields" },
      { id: "p2-write", label: "Write-path rules" },
      { id: "p2-gov", label: "Governance properties" },
      { id: "p2-reject", label: "Rejection rules" },
    ],
  },
  {
    id: "p3", num: "05", label: "Naming & identity",
    sub: [
      { id: "p3-naming", label: "Conventions" },
      { id: "p3-linter", label: "Convention checker" },
      { id: "p3-drift", label: "Property key drift" },
      { id: "p3-identity", label: "Identity strategy" },
    ],
  },
  {
    id: "p4", num: "06", label: "Tenant lifecycle",
    sub: [
      { id: "p4-tiers", label: "Three tiers" },
      { id: "p4-promote", label: "Promotion gates" },
      { id: "p4-reap", label: "Expiry & reaping" },
      { id: "p4-sandbox", label: "Shared sandbox" },
    ],
  },
  {
    id: "p5", num: "07", label: "Observability",
    sub: [
      { id: "p5-metrics", label: "What we measure" },
      { id: "p5-score", label: "Conformance score" },
      { id: "p5-lineage", label: "Lineage answers" },
    ],
  },
  {
    id: "p6", num: "08", label: "Knowledge catalogue",
    sub: [
      { id: "p6-layers", label: "Observed vs. declared" },
      { id: "p6-model", label: "Meta-graph model" },
      { id: "p6-queries", label: "What it unlocks" },
    ],
  },
  { id: "roles", num: "09", label: "Roles & accountability" },
  { id: "ladder", num: "10", label: "Conformance ladder" },
  {
    id: "appendix", num: "11", label: "Appendices",
    sub: [
      { id: "ax-a", label: "A · Contract template" },
      { id: "ax-b", label: "B · Label template" },
      { id: "ax-c", label: "C · Relationship template" },
      { id: "ax-d", label: "D · Introspection queries" },
      { id: "ax-e", label: "E · Open decisions" },
    ],
  },
];

export const LENSES = [
  { key: "all", label: "Everything" },
  { key: "biz", label: "Business & governance" },
  { key: "eng", label: "Data engineering" },
  { key: "sci", label: "Data science" },
  { key: "ops", label: "Platform ops" },
];

export const FLAT_NAV = NAV.flatMap((s) => [
  { id: s.id, num: s.num, label: s.label },
  ...(s.sub || []).map((x) => ({ id: x.id, num: "·", label: x.label })),
]);
