import { createContext, useContext } from "react";
import { isVisible } from "./nav.js";

export const LensContext = createContext("all");

export const useLens = () => useContext(LensContext);

/**
 * Whether a section survives the current audience filter.
 * Filtering hides rather than fades: faded text is unreadable text, and a
 * half-visible section reads as broken rather than as filtered out.
 */
export function useVisibleSection(sectionId) {
  return isVisible(sectionId, useContext(LensContext));
}

/* Setting the lens from inside the document — the toolbar is not the only
   place a reader should be able to choose their path. */
export const LensSetContext = createContext(() => {});
export const useSetLens = () => useContext(LensSetContext);
