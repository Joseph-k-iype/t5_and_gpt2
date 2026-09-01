/* Rewrites the single-file Vite build into a body-fragment page.
   The publishing host supplies its own <!doctype>/<head>/<body> skeleton, so
   the fragment carries only: title, font links, styles, the mount point, and
   the inlined module script — in that order. */

import { readFileSync, writeFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const src = resolve(here, "../dist-single/index.html");
const out = resolve(here, "../dist-single/artifact.html");

const html = readFileSync(src, "utf8");

const pick = (re, what) => {
  const m = html.match(re);
  if (!m) throw new Error(`make-artifact: could not find ${what} in ${src}`);
  return m[0];
};

const title = pick(/<title>[\s\S]*?<\/title>/i, "<title>");
const preconnects = html.match(/<link rel="preconnect"[^>]*>/gi) || [];
const fonts = pick(/<link\s+[\s\S]*?fonts\.googleapis\.com[\s\S]*?>/i, "the font stylesheet link");
const styles = html.match(/<style[\s\S]*?<\/style>/gi) || [];
const script = pick(/<script type="module"[\s\S]*?<\/script>/i, "the inlined module script");

const fragment = [
  title,
  ...preconnects,
  fonts.replace(/\s+/g, " "),
  ...styles,
  '<div id="root"></div>',
  script,
].join("\n");

writeFileSync(out, fragment + "\n", "utf8");
console.log(
  `make-artifact: wrote ${out} (${(fragment.length / 1024).toFixed(0)} kB)`
);
