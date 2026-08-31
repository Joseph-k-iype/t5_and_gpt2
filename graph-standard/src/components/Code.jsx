import { useRef, useState } from "react";
import { highlight } from "../lib/highlight.js";

export default function Code({ lang = "cypher", caption, note, code }) {
  const [label, setLabel] = useState("Copy");
  const raw = useRef(code);
  raw.current = code;

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(raw.current);
      setLabel("Copied");
      setTimeout(() => setLabel("Copy"), 1600);
    } catch {
      setLabel("Select manually");
      setTimeout(() => setLabel("Copy"), 2400);
    }
  };

  return (
    <figure className="code" data-reveal>
      <figcaption>
        <span className="lang">{lang === "yaml" ? "YAML" : "Cypher"}</span>
        <span className={note ? "c-stop" : undefined}>{caption}</span>
        <span className="sp" />
        <button
          type="button"
          className={"copy" + (label === "Copied" ? " done" : "")}
          onClick={copy}
        >
          {label}
        </button>
      </figcaption>
      <pre>
        <code dangerouslySetInnerHTML={{ __html: highlight(code, lang) }} />
      </pre>
    </figure>
  );
}
