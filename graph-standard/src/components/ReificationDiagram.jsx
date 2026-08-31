import { useRef } from "react";
import { useDiagramDraw } from "../lib/motion.js";

/**
 * Why an edge with properties is a dead end, and what reifying it buys.
 * This replaces the pair of code samples that used to make the same point —
 * the failure is structural, so it should be shown structurally.
 */
export default function ReificationDiagram() {
  const ref = useRef(null);
  useDiagramDraw(ref);

  return (
    <div className="diagram-wrap" data-reveal>
      <svg
        ref={ref}
        viewBox="0 0 1140 360"
        className="diagram diagram--wide"
        role="img"
        aria-label="Before: a Customer node joined to a Counterparty node by a TRANSACTED_WITH edge carrying amount and date. A rule, an analyst and evidence sit below with no way to attach, because an edge cannot have edges. After: the transaction is its own node — Customer initiated Transaction, Transaction settled with Counterparty, and the Transaction is flagged by a Rule and reviewed by an Analyst."
      >
        <defs>
          <marker id="ah2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="#9A928F" />
          </marker>
        </defs>

        <text className="dg-cap" x="30" y="26">Before · the edge is a dead end</text>
        <text className="dg-cap" x="600" y="26">After · the transaction is a node</text>
        <line x1="545" y1="14" x2="545" y2="346" stroke="#DDD8D5" strokeWidth="1" />

        <g className="dg-edges" markerEnd="url(#ah2)">
          <line x1="182" y1="83" x2="326" y2="83" />
          <line x1="752" y1="94" x2="786" y2="152" />
          <line x1="952" y1="152" x2="996" y2="96" />
          <line x1="838" y1="198" x2="790" y2="276" />
          <line x1="890" y1="198" x2="912" y2="276" />
        </g>

        {/* the attachments that have nowhere to go */}
        <g stroke="#DFB6B6" strokeWidth="1.2" strokeDasharray="4 4" fill="none">
          <path d="M 85 276 L 210 196" />
          <path d="M 225 276 L 250 196" />
          <path d="M 370 276 L 296 196" />
        </g>
        <g stroke="#8E1017" strokeWidth="2" strokeLinecap="round">
          <line x1="244" y1="150" x2="264" y2="170" />
          <line x1="264" y1="150" x2="244" y2="170" />
        </g>
        <text className="dg-lbls" x="254" y="196" textAnchor="middle"
              fontFamily="Archivo, sans-serif" fontSize="11" fontWeight="600"
              letterSpacing="0.06em" fill="#8E1017">NOWHERE TO ATTACH</text>

        <g className="dg-lbls">
          <text x="254" y="68">TRANSACTED_WITH</text>
          <text x="254" y="106">amount, executedAt</text>
          <text x="742" y="132">INITIATED</text>
          <text x="1000" y="132">SETTLED_WITH</text>
          <text x="770" y="248">FLAGGED_BY</text>
          <text x="938" y="248">REVIEWED_BY</text>
        </g>

        <g className="dg-nodes">
          <g className="dg-n"><rect x="30" y="60" width="150" height="46" rx="6" /><text x="105" y="88">Customer</text></g>
          <g className="dg-n"><rect x="328" y="60" width="160" height="46" rx="6" /><text x="408" y="88">Counterparty</text></g>

          <g className="dg-n dg-bad"><rect x="30" y="276" width="110" height="42" rx="6" /><text x="85" y="302">Rule</text></g>
          <g className="dg-n dg-bad"><rect x="170" y="276" width="110" height="42" rx="6" /><text x="225" y="302">Analyst</text></g>
          <g className="dg-n dg-bad"><rect x="310" y="276" width="120" height="42" rx="6" /><text x="370" y="302">Evidence</text></g>

          <g className="dg-n"><rect x="600" y="60" width="150" height="46" rx="6" /><text x="675" y="88">Customer</text></g>
          <g className="dg-n dg-key"><rect x="788" y="152" width="160" height="46" rx="6" /><text x="868" y="180">Transaction</text></g>
          <g className="dg-n"><rect x="998" y="60" width="132" height="46" rx="6" /><text x="1064" y="88">Counterparty</text></g>
          <g className="dg-n"><rect x="700" y="276" width="110" height="42" rx="6" /><text x="755" y="302">Rule</text></g>
          <g className="dg-n"><rect x="860" y="276" width="120" height="42" rx="6" /><text x="920" y="302">Analyst</text></g>
        </g>
      </svg>
    </div>
  );
}
