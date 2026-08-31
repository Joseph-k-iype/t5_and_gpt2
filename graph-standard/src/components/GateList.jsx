import useLocalState from "../lib/useLocalState.js";
import { Tool } from "./Primitives.jsx";

/**
 * A promotion gate. The bar is the point: goal-gradient says people push
 * harder the closer the end looks, and a bare "3/9" hides that entirely.
 */
export default function GateList({ title, storageKey, intro, items }) {
  const [checked, setChecked] = useLocalState(storageKey, {});
  const done = items.filter((it) => checked[it.k]).length;
  const pct = Math.round((done / items.length) * 100);
  const complete = done === items.length;

  return (
    <Tool
      title={title}
      hint={complete ? "Gate met" : `${items.length - done} left`}
    >
      <div
        className="gate-progress"
        role="progressbar"
        aria-valuenow={done}
        aria-valuemin={0}
        aria-valuemax={items.length}
        aria-label={`${title}: ${done} of ${items.length} complete`}
      >
        <b className={complete ? "done" : undefined} style={{ width: `${pct}%` }} />
      </div>

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
