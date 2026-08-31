import useLocalState from "../lib/useLocalState.js";
import { Tool } from "./Primitives.jsx";

export default function GateList({ title, storageKey, intro, items }) {
  const [checked, setChecked] = useLocalState(storageKey, {});
  const done = items.filter((it) => checked[it.k]).length;

  return (
    <Tool title={title} hint={<><span className="gate-count">{done}</span>/{items.length}</>}>
      {intro && <p className="small mt-0">{intro}</p>}
      <ul className="gate-list">
        {items.map((it) => (
          <li key={it.k}>
            <label>
              <input
                type="checkbox"
                checked={!!checked[it.k]}
                onChange={(e) =>
                  setChecked((prev) => ({ ...prev, [it.k]: e.target.checked }))
                }
              />
              <span>{it.text}</span>
            </label>
          </li>
        ))}
      </ul>
    </Tool>
  );
}
