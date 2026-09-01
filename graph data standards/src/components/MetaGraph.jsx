import { useRef } from "react";
import { useDiagramDraw } from "../lib/motion.js";

const NODES = [
  { x: 30,  y: 24,  label: "Person" },
  { x: 30,  y: 120, label: "Team" },
  { x: 30,  y: 216, label: "Domain" },
  { x: 30,  y: 330, label: "Snapshot" },
  { x: 30,  y: 470, label: "SourceSystem" },
  { x: 310, y: 146, label: "Tenant", key: true },
  { x: 310, y: 470, label: "IngestionContract", key: true },
  { x: 600, y: 120, label: "LabelDef", key: true },
  { x: 600, y: 330, label: "RelTypeDef", key: true },
  { x: 880, y: 40,  label: "CanonicalConcept" },
  { x: 880, y: 225, label: "PropertyKeyDef", key: true },
  { x: 880, y: 400, label: "Classification" },
];

const LABELS = [
  [264, 152, "OWNS"],
  [268, 206, "BELONGS_TO"],
  [390, 42,  "STEWARDS"],
  [549, 150, "DECLARES"],
  [536, 268, "DECLARES"],
  [242, 286, "MEASURED_AS"],
  [405, 336, "WRITES_TO"],
  [265, 486, "READS_FROM"],
  [536, 330, "POPULATES"],
  [562, 436, "POPULATES"],
  [660, 252, "CONNECTS ▸ source"],
  [732, 292, "CONNECTS ▸ target"],
  [838, 196, "HAS_PROPERTY"],
  [838, 316, "HAS_PROPERTY"],
  [842, 100, "ALIGNS_TO"],
  [975, 344, "CLASSIFIED_AS"],
  [405, 568, "DEPENDS_ON"],
];

export default function MetaGraph() {
  const ref = useRef(null);
  useDiagramDraw(ref);

  return (
    <div className="diagram-wrap" data-reveal>
      <svg
        ref={ref}
        viewBox="0 0 1140 600"
        className="diagram"
        role="img"
        aria-label="Catalogue meta-graph. Team owns Tenant; Tenant belongs to Domain, declares LabelDef and RelTypeDef, and is measured as Snapshot. Person stewards LabelDef. IngestionContract reads from SourceSystem, writes to Tenant, and populates LabelDef and RelTypeDef, and depends on other contracts. RelTypeDef connects LabelDef as source and as target. LabelDef and RelTypeDef have PropertyKeyDef properties. LabelDef aligns to CanonicalConcept. PropertyKeyDef is classified as Classification."
      >
        <defs>
          <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#8B918C" />
          </marker>
        </defs>

        <g className="dg-edges">
          <line x1="220" y1="152" x2="308" y2="168" />
          <line x1="310" y1="190" x2="222" y2="240" />
          <polyline points="220,50 560,50 560,140 598,140" fill="none" />
          <line x1="500" y1="166" x2="598" y2="150" />
          <line x1="500" y1="190" x2="598" y2="344" />
          <line x1="312" y1="192" x2="222" y2="348" />
          <line x1="405" y1="468" x2="405" y2="200" />
          <line x1="308" y1="496" x2="222" y2="496" />
          <line x1="500" y1="478" x2="598" y2="176" />
          <line x1="500" y1="492" x2="598" y2="374" />
          <path d="M 660 328 L 660 174" fill="none" />
          <path d="M 732 328 L 732 174" fill="none" />
          <line x1="790" y1="160" x2="878" y2="238" />
          <line x1="790" y1="342" x2="878" y2="272" />
          <line x1="790" y1="132" x2="878" y2="82" />
          <line x1="975" y1="279" x2="975" y2="398" />
          <path d="M 360 522 C 330 570, 480 570, 450 524" fill="none" />
        </g>

        <g className="dg-lbls">
          {LABELS.map(([x, y, t]) => (
            <text key={t + x} x={x} y={y}>{t}</text>
          ))}
        </g>

        <g className="dg-nodes">
          {NODES.map((n) => (
            <g key={n.label} className={"dg-n" + (n.key ? " dg-key" : "")}>
              <rect x={n.x} y={n.y} width="190" height="52" rx="6" />
              <text x={n.x + 95} y={n.y + 31}>{n.label}</text>
            </g>
          ))}
        </g>
      </svg>
    </div>
  );
}
