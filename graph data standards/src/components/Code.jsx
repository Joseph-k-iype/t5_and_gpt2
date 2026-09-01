import { useRef, useState } from "react";
import { highlightYaml } from "../lib/highlight.js";

/** A registry or contract artefact, copyable as written. */
export default function Code({ caption, code }) {
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
        <span className="lang">YAML</span>
        <span>{caption}</span>
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
        <code dangerouslySetInnerHTML={{ __html: highlightYaml(code) }} />
      </pre>
    </figure>
  );
}
