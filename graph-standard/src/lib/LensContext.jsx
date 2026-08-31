import { createContext, useContext } from "react";

export const LensContext = createContext("all");

/** Returns the class list for a block tagged with an audience. */
export function useDim(aud) {
  const lens = useContext(LensContext);
  if (!aud || lens === "all") return "";
  return aud.split(/\s+/).includes(lens) ? "" : "dim";
}
