/* Fragment shader for the masthead field.
   A domain-warped fbm resolved into contour lines and a dot lattice —
   a plotted survey drawing that never quite settles. Light-theme palette
   is baked in as uniforms so the shader matches the CSS tokens exactly. */

export const VERT = `
attribute vec2 a_pos;
void main(){ gl_Position = vec4(a_pos, 0.0, 1.0); }
`;

export const FRAG = `
precision highp float;

uniform vec2  u_res;
uniform float u_time;
uniform vec2  u_mouse;
uniform float u_scroll;
uniform float u_intensity;
uniform vec3  u_paper;
uniform vec3  u_ink;
uniform vec3  u_line;

float hash(vec2 p){
  return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}

float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
}

float fbm(vec2 p){
  float v = 0.0, a = 0.5;
  for(int i = 0; i < 5; i++){ v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
}

void main(){
  vec2 uv = gl_FragCoord.xy / u_res.xy;
  vec2 p  = vec2(uv.x * (u_res.x / u_res.y), uv.y);

  float t = u_time * 0.035;
  vec2 warp = vec2(u_scroll * 0.30) + u_mouse * 0.10;

  vec2 q = vec2(fbm(p * 1.55 + vec2(0.0, t)),
                fbm(p * 1.55 + vec2(5.2, 1.3) - t * 0.62));
  vec2 r = vec2(fbm(p * 2.05 + 3.2 * q + vec2(1.7, 9.2) + t * 0.40),
                fbm(p * 2.05 + 3.2 * q + vec2(8.3, 2.8) - t * 0.28));
  float f = fbm(p * 1.35 + 2.5 * r + warp);

  /* contour lines — the plotter pass */
  float bands = fract(f * 7.5);
  float w = fwidth(bands) * 1.4 + 0.004;
  float edge = smoothstep(0.0, w, bands) * smoothstep(2.0 * w + 0.030, w, bands);

  /* survey lattice — one dot per 44 device px */
  vec2 g = fract(gl_FragCoord.xy / 44.0) - 0.5;
  float dots = smoothstep(0.10, 0.045, length(g));

  /* where the field runs high, lattice points resolve into graph nodes */
  float d    = length(g);
  float ring = smoothstep(0.175, 0.125, d) - smoothstep(0.115, 0.075, d);
  float live = smoothstep(0.58, 0.78, f);

  vec3 col = u_paper;
  /* Red carries far more visual weight than the blue this replaced, so every
     pass is dialled back — the field should register at the edge of vision. */
  col = mix(col, u_line, edge * 0.42 * u_intensity);
  col = mix(col, u_ink,  dots * 0.10 * u_intensity);
  col = mix(col, u_ink,  ring * live * 0.26 * u_intensity);
  col = mix(col, mix(u_paper, u_ink, 0.04), smoothstep(0.56, 0.96, f) * u_intensity);

  /* keep the type side clean: fade in from the left, ease out at the base */
  float fade = smoothstep(0.06, 0.70, uv.x);
  col = mix(u_paper, col, fade);
  col = mix(col, u_paper, smoothstep(0.62, 1.0, 1.0 - uv.y) * 0.40);

  gl_FragColor = vec4(col, 1.0);
}
`;
