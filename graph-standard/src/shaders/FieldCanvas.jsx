import { useEffect, useRef } from "react";
import { VERT, FRAG } from "./field.js";

const PAPER = [0.945, 0.949, 0.933];
const INK   = [0.137, 0.204, 0.561];
const LINE  = [0.725, 0.757, 0.894];

function compile(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    console.warn("shader:", gl.getShaderInfoLog(sh));
    gl.deleteShader(sh);
    return null;
  }
  return sh;
}

/**
 * Full-bleed WebGL field behind the masthead.
 * Degrades to a CSS dot lattice when WebGL is unavailable, and holds a
 * single still frame when the viewer asks for reduced motion.
 */
export default function FieldCanvas({ intensity = 1 }) {
  const hostRef = useRef(null);
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const host = hostRef.current;
    if (!canvas || !host) return;

    const gl =
      canvas.getContext("webgl", { antialias: false, alpha: false, powerPreference: "low-power" }) ||
      canvas.getContext("experimental-webgl");
    if (!gl) {
      host.classList.add("no-webgl");
      return;
    }
    gl.getExtension("OES_standard_derivatives");

    const prog = gl.createProgram();
    const vs = compile(gl, gl.VERTEX_SHADER, VERT);
    const fs = compile(
      gl,
      gl.FRAGMENT_SHADER,
      gl.getExtension("OES_standard_derivatives")
        ? "#extension GL_OES_standard_derivatives : enable\n" + FRAG
        : FRAG.replace("fwidth(bands)", "0.006")
    );
    if (!vs || !fs) { host.classList.add("no-webgl"); return; }
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.bindAttribLocation(prog, 0, "a_pos");
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) { host.classList.add("no-webgl"); return; }
    gl.useProgram(prog);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const U = (n) => gl.getUniformLocation(prog, n);
    const uRes = U("u_res"), uTime = U("u_time"), uMouse = U("u_mouse");
    const uScroll = U("u_scroll"), uInt = U("u_intensity");
    gl.uniform3fv(U("u_paper"), PAPER);
    gl.uniform3fv(U("u_ink"), INK);
    gl.uniform3fv(U("u_line"), LINE);
    gl.uniform1f(uInt, intensity);

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const state = { mx: 0, my: 0, tx: 0, ty: 0, scroll: 0, visible: true };
    let raf = 0, t0 = performance.now();

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const w = Math.max(1, Math.round(host.clientWidth * dpr * 0.85));
      const h = Math.max(1, Math.round(host.clientHeight * dpr * 0.85));
      if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w; canvas.height = h;
        gl.viewport(0, 0, w, h);
        gl.uniform2f(uRes, w, h);
      }
    };

    const onMove = (e) => {
      const r = host.getBoundingClientRect();
      state.tx = ((e.clientX - r.left) / r.width - 0.5) * 2;
      state.ty = ((e.clientY - r.top) / r.height - 0.5) * 2;
    };
    const onScroll = () => {
      state.scroll = window.scrollY / Math.max(1, window.innerHeight);
    };

    const frame = (now) => {
      state.mx += (state.tx - state.mx) * 0.045;
      state.my += (state.ty - state.my) * 0.045;
      gl.uniform1f(uTime, (now - t0) / 1000);
      gl.uniform2f(uMouse, state.mx, state.my);
      gl.uniform1f(uScroll, state.scroll);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
      if (!reduced && state.visible) raf = requestAnimationFrame(frame);
    };

    const io = new IntersectionObserver(
      ([en]) => {
        state.visible = en.isIntersecting;
        if (state.visible && !reduced && !raf) raf = requestAnimationFrame(frame);
        if (!state.visible && raf) { cancelAnimationFrame(raf); raf = 0; }
      },
      { threshold: 0 }
    );

    resize();
    frame(performance.now());
    io.observe(host);
    window.addEventListener("resize", resize);
    window.addEventListener("scroll", onScroll, { passive: true });
    if (!reduced) window.addEventListener("pointermove", onMove, { passive: true });

    return () => {
      if (raf) cancelAnimationFrame(raf);
      io.disconnect();
      window.removeEventListener("resize", resize);
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("pointermove", onMove);
      gl.deleteProgram(prog); gl.deleteShader(vs); gl.deleteShader(fs); gl.deleteBuffer(buf);
    };
  }, [intensity]);

  return (
    <div className="field-stage" ref={hostRef} aria-hidden="true">
      <div className="field-fallback" />
      <canvas ref={canvasRef} />
    </div>
  );
}
