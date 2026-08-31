/* The standard states models, rules and templates — never queries — so the
   only thing that needs highlighting is the YAML of the registry and contract
   artefacts. Input is always a literal string authored in this repo. */

const esc = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");

const YAML =
  /(#[^\n]*)|("(?:[^"\\]|\\.)*")|(^[ \t]*(?:- )?[A-Za-z_][\w.\-]*(?=:))|(\b(?:true|false|null)\b)|(\b\d+(?:\.\d+)?\b)/gm;

const span = (cls, text) => `<span class="${cls}">${text}</span>`;

export function highlightYaml(source) {
  return esc(source).replace(YAML, (m, com, str, key, bool, num) => {
    if (com) return span("t-com", com);
    if (str) return span("t-str", str);
    if (key) return span("t-key", key);
    if (bool) return span("t-kw", bool);
    if (num) return span("t-num", num);
    return m;
  });
}
