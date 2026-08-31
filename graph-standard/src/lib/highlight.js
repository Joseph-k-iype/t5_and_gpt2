/* A deliberately small highlighter: two languages, no dependency, and it
   never sees anything but literal strings authored in this repo. */

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const CYPHER =
  /(\/\/[^\n]*)|('(?:[^'\\]|\\.)*'|"(?:[^"\\]|\\.)*")|(\$[A-Za-z_]\w*)|(:[A-Z_][A-Za-z0-9_]*)|\b(MATCH|OPTIONAL|RETURN|WHERE|WITH|UNWIND|MERGE|CREATE|SET|DELETE|DETACH|CALL|ORDER|BY|LIMIT|SKIP|AS|AND|OR|NOT|NULL|IS|EXISTS|DISTINCT|CASE|WHEN|THEN|ELSE|END|ON|YIELD)\b|(\b\d+(?:\.\d+)?\b)/g;

const YAML =
  /(#[^\n]*)|("(?:[^"\\]|\\.)*")|(^[ \t]*(?:- )?[A-Za-z_][\w.\-]*(?=:))|(\b(?:true|false|null)\b)|(\b\d+(?:\.\d+)?\b)/gm;

const span = (cls, text) => `<span class="${cls}">${text}</span>`;

export function highlight(source, lang = "cypher") {
  const src = esc(source);
  if (lang === "yaml") {
    return src.replace(YAML, (m, com, str, key, bool, num) => {
      if (com) return span("t-com", com);
      if (str) return span("t-str", str);
      if (key) return span("t-key", key);
      if (bool) return span("t-kw", bool);
      if (num) return span("t-num", num);
      return m;
    });
  }
  return src.replace(CYPHER, (m, com, str, par, lbl, kw, num) => {
    if (com) return span("t-com", com);
    if (str) return span("t-str", str);
    if (par) return span("t-num", par);
    if (lbl) return span("t-lbl", lbl);
    if (kw) return span("t-kw", kw);
    if (num) return span("t-num", num);
    return m;
  });
}
