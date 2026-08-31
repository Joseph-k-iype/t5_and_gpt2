import { useRef } from "react";
import { useCountUp } from "../lib/motion.js";

/** A number that counts up the first time it is read. */
export default function Counter({ value, suffix = "" }) {
  const ref = useRef(null);
  useCountUp(ref, value, { suffix });
  return <span className="num counter" ref={ref}>0{suffix}</span>;
}
