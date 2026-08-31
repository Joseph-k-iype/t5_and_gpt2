import { useState } from "react";
import { Tool } from "./Primitives.jsx";

const BANNED = ["cust","txn","acct","addr","amt","qty","nbr","num","dt","ts","org","cd","desc","mgr","dept"];
const PLURAL_OK = ["status","address","analysis","basis","business","access","process","series","class"];

const words = (s) => s.replace(/([a-z0-9])([A-Z])/g, "$1 $2").split(/[\s_]+/).filter(Boolean);
const abbrevs = (s) => words(s).filter((w) => BANNED.includes(w.toLowerCase()));

const KINDS = [
  { key: "label", label: "Node label", sample: "Customers" },
  { key: "facet", label: "Facet label", sample: "Sanctioned" },
  { key: "rel", label: "Relationship type", sample: "ACCOUNT_OWNER" },
  { key: "prop", label: "Property key", sample: "customer_name" },
  { key: "graph", label: "Graph key", sample: "prd.risk.kyc-network" },
  { key: "index", label: "Index name", sample: "idx_customer_entityid" },
  { key: "contract", label: "Contract ID", sample: "ing.kyc.corereg" },
];

const CHECKS = {
  label(v) {
    const w = words(v);
    const last = (w[w.length - 1] || "").toLowerCase();
    const ab = abbrevs(v);
    return [
      { ok: /^[A-Z][A-Za-z0-9]*$/.test(v), t: "PascalCase, letters and digits only", d: <>e.g. <code>TradeAccount</code></> },
      { ok: !v.includes("_"), t: "No underscores in node labels", d: "underscores are for relationship types" },
      { ok: !(last.endsWith("s") && !PLURAL_OK.includes(last)), t: "Singular noun", d: "the label describes one node, not the set" },
      { ok: ab.length === 0, t: "No non-standard abbreviations", d: ab.length ? <>found <code>{ab.join(", ")}</code></> : <><code>KYC</code>, <code>LEI</code>, <code>ISIN</code> are permitted</> },
      { ok: v[0] !== "_", t: "No reserved underscore prefix", d: <><code>_</code> belongs to the platform</> },
    ];
  },
  facet(v) {
    return [
      { ok: /^[A-Z][A-Za-z0-9]*$/.test(v), t: "PascalCase adjective or state", d: <>e.g. <code>Sanctioned</code>, <code>Deprecated</code></> },
      { ok: !v.includes("_"), t: "No underscores", d: "" },
      { ok: true, t: "Declared against a primary label in the registry", d: "facets carry no required properties of their own — check this by hand" },
    ];
  },
  rel(v) {
    const segs = v.split("_");
    const nounish = /(ER|OR|TION|MENT|ITY)$/.test(segs[segs.length - 1] || "");
    return [
      { ok: /^[A-Z][A-Z0-9]*(_[A-Z0-9]+)*$/.test(v), t: "UPPER_SNAKE_CASE", d: <>e.g. <code>HOLDS_ACCOUNT</code></> },
      { ok: !nounish, t: "Verb phrase, not a noun", d: nounish ? <><code>{v}</code> reads as a noun — you probably need reification into a node</> : <><code>OWNS</code>, not <code>OWNER</code></> },
      { ok: abbrevs(v).length === 0, t: "No non-standard abbreviations", d: "" },
      { ok: true, t: "Endpoint label pairs declared in the registry", d: "every permitted (source) → (target) pair — check this by hand" },
    ];
  },
  prop(v) {
    const reserved = v[0] === "_";
    const suffix = /(Date|Time|Timestamp|Str|String|Num|Int|Flag)$/.test(v) && !/^(is|has)/.test(v);
    const bool = /^(is|has|can|should)/.test(v);
    return [
      { ok: /^_?[a-z][A-Za-z0-9]*$/.test(v), t: "camelCase, letters and digits only", d: <>e.g. <code>legalName</code>, <code>ownershipPct</code></> },
      { ok: v.indexOf("_") <= 0, t: "No snake_case", d: "the estate standard is camelCase" },
      { ok: !reserved, t: "Not in the reserved platform namespace", d: reserved ? <><code>_</code>-prefixed keys are written by the platform, never by application logic</> : "" },
      { ok: !suffix, t: "No type suffix", d: suffix ? "the registry declares the type — drop the suffix" : <><code>openedAt</code>, not <code>openedAtDate</code></> },
      { ok: true, t: bool ? "Reads as an assertion — good for a boolean" : "Checked against the property key dictionary", d: bool ? "" : "reuse an existing key, or justify a new one" },
    ];
  },
  graph(v) {
    const p = v.split(".");
    return [
      { ok: ["dev", "tst", "uat", "prd"].includes(p[0]), t: "Starts with a known environment", d: <><code>dev</code> · <code>tst</code> · <code>uat</code> · <code>prd</code></> },
      { ok: p.length >= 3, t: "Carries a domain and a use case", d: <><code>&lt;env&gt;.&lt;domain&gt;.&lt;usecase&gt;</code></> },
      { ok: /^[a-z0-9.-]+$/.test(v), t: "Lowercase, dots and hyphens only", d: "" },
      { ok: p[0] !== "prd" || /^v\d+$/.test(p[p.length - 1]), t: "Production graph keys are versioned", d: <>append <code>.v1</code> — you will need <code>.v2</code> one day</> },
    ];
  },
  index(v) {
    return [
      { ok: /^idx_[a-z0-9]+_[a-z0-9_]+$/.test(v), t: <>Matches <code>idx_&lt;label&gt;_&lt;property&gt;</code></>, d: <>e.g. <code>idx_customer_entityid</code></> },
      { ok: v === v.toLowerCase(), t: "Lowercase throughout", d: "" },
    ];
  },
  contract(v) {
    return [
      { ok: /^ing\./.test(v), t: <>Starts with <code>ing.</code></>, d: "" },
      { ok: /^ing\.[a-z0-9-]+\.[a-z0-9-]+\.v\d+$/.test(v), t: <>Matches <code>ing.&lt;domain&gt;.&lt;source&gt;.v&lt;n&gt;</code></>, d: <>e.g. <code>ing.kyc.corereg.v2</code></> },
      { ok: /\.v\d+$/.test(v), t: "Versioned", d: "a contract change that alters semantics is a new version" },
    ];
  },
};

export default function Linter() {
  const [kind, setKind] = useState("label");
  const [value, setValue] = useState("Customers");

  const v = value.trim();
  const rules = v ? CHECKS[kind](v) : [];
  const failed = rules.filter((r) => !r.ok);

  return (
    <Tool title="Convention checker" hint="Type a name — rules evaluate as you go">
      <div className="lint-fields">
        <label className="fld">
          <span className="k">Element type</span>
          <select
            value={kind}
            onChange={(e) => {
              const k = e.target.value;
              setKind(k);
              setValue(KINDS.find((x) => x.key === k)?.sample || "");
            }}
          >
            {KINDS.map((k) => (
              <option key={k.key} value={k.key}>{k.label}</option>
            ))}
          </select>
        </label>
        <label className="fld">
          <span className="k">Name</span>
          <input
            type="text"
            value={value}
            spellCheck="false"
            autoComplete="off"
            onChange={(e) => setValue(e.target.value)}
          />
        </label>
      </div>

      <div id="lint-verdict" className={!v ? "" : failed.length ? "fail" : "pass"}>
        {!v ? (
          "Type a name to check it."
        ) : failed.length ? (
          <><code>{v}</code> breaks {failed.length} rule{failed.length > 1 ? "s" : ""}</>
        ) : (
          <><code>{v}</code> conforms</>
        )}
      </div>

      <ul id="lint-rules">
        {rules.map((r, i) => (
          <li key={i} className={r.ok ? "ok" : "no"}>
            <span className="mk">{r.ok ? "✓" : "✕"}</span>
            <span>
              {r.t}
              {r.d ? <em> — {r.d}</em> : null}
            </span>
          </li>
        ))}
      </ul>
    </Tool>
  );
}
