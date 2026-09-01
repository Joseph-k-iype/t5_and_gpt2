import { useCallback, useEffect, useState } from "react";

/* localStorage can throw outright in private windows and preview contexts,
   so every read and write is guarded and the page renders fine without it. */
function read(key, fallback) {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch {
    return fallback;
  }
}

export default function useLocalState(key, fallback) {
  const [value, setValue] = useState(() => read(key, fallback));

  useEffect(() => {
    try {
      window.localStorage.setItem(key, JSON.stringify(value));
    } catch {
      /* storage unavailable — the page still works, it just forgets */
    }
  }, [key, value]);

  const reset = useCallback(() => setValue(fallback), [fallback]);
  return [value, setValue, reset];
}
