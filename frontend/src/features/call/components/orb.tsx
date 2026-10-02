/**
 * ZEN's presence on a call: the ensō from the logo, drawn as a living ink
 * brushstroke. A fragment shader renders a heavy start, a long dry taper and
 * the open gap; JS eases a handful of stroke parameters toward the target for
 * each state, so every transition is continuous rather than a swap.
 *
 *   idle       slow breath
 *   listening  steady, faint response to the room
 *   hearing    the stroke swells with the user's voice
 *   thinking   the stroke shortens and chases its gap around the circle
 *   speaking   thickness and radius follow ZEN's own voice
 */
import { useEffect, useLayoutEffect, useRef } from "react";

import { cn } from "@/lib/utils";

import type { ZenState } from "../call-store";

export type OrbState = ZenState | "idle";

type Params = { radius: number; width: number; coverage: number; wobble: number; spin: number };

const TAU = Math.PI * 2;
/** Where the stroke starts; puts the gap at about one o'clock, as in the logo. */
const START_ANGLE = 0.35;

function target(state: OrbState, level: number, time: number): Params {
  switch (state) {
    case "idle":
      return { radius: 0.62 + Math.sin(time * 0.9) * 0.012, width: 0.2, coverage: 0.9, wobble: 0.02, spin: 0 };
    case "listening":
      return { radius: 0.61 + level * 0.02, width: 0.19 + level * 0.06, coverage: 0.9, wobble: 0.02 + level * 0.03, spin: 0 };
    case "hearing":
      return { radius: 0.6 + level * 0.05, width: 0.2 + level * 0.2, coverage: 0.92, wobble: 0.03 + level * 0.08, spin: 0 };
    case "thinking":
      return { radius: 0.6, width: 0.15, coverage: 0.66 + Math.sin(time * 2.4) * 0.1, wobble: 0.025, spin: 3.4 };
    case "speaking":
      return { radius: 0.62 + level * 0.05, width: 0.19 + level * 0.24, coverage: 0.9, wobble: 0.03 + level * 0.1, spin: 0 };
  }
}

const VERTEX = `#version 300 es
in vec2 a_pos;
void main() { gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FRAGMENT = `#version 300 es
precision highp float;
uniform vec2 u_res;
uniform float u_time, u_radius, u_width, u_coverage, u_rotation, u_wobble;
uniform vec3 u_color;
out vec4 outColor;

const float TAU = 6.28318530718;

float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float noise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * noise(p); p *= 2.03; a *= 0.5; }
  return v;
}

void main() {
  vec2 p = (gl_FragCoord.xy - 0.5 * u_res) / (0.5 * min(u_res.x, u_res.y));
  float r = length(p);
  float a = atan(p.y, p.x);
  float t = fract((u_rotation - a) / TAU);            // 0 at the stroke start, clockwise

  float R = u_radius + (fbm(vec2(a * 1.5, u_time * 0.35)) - 0.5) * u_wobble;
  float along = clamp(t / u_coverage, 0.0, 1.0);
  float w = u_width * mix(1.0, 0.16, pow(along, 1.5));

  float edge = (fbm(vec2(t * 26.0, r * 6.0 + u_time * 0.15)) - 0.5) * w * 0.5;
  float d = abs(r - R) - (w * 0.5 + edge);
  float aa = fwidth(r) * 1.25;
  float ink = 1.0 - smoothstep(-aa, aa, d);

  float tail = 1.0 - smoothstep(u_coverage - 0.07, u_coverage, t);
  float across = (r - R) / max(w, 1e-3);
  float bristle = fbm(vec2(across * 9.0, t * 3.0));
  float dry = smoothstep(0.2 + 0.5 * along, 0.85, bristle);
  ink *= tail * mix(1.0, dry, 0.12 + 0.68 * along);

  // Where the brush first lands: a rounded, slightly heavier blot.
  vec2 start = u_radius * vec2(cos(u_rotation), sin(u_rotation));
  float blot = length(p - start) - u_width * (0.5 + (fbm(p * 9.0) - 0.5) * 0.12);
  ink = max(ink, 1.0 - smoothstep(-aa, aa, blot));

  float alpha = clamp(ink, 0.0, 1.0);
  outColor = vec4(u_color * alpha, alpha);
}`;

const hexToRgb = (hex: string): [number, number, number] => {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex.trim());
  const n = m ? parseInt(m[1]!, 16) : 0xd01e1e;
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
};

type Renderer = { draw: (p: Params, rotation: number, time: number) => void; dispose: () => void };

function webglRenderer(canvas: HTMLCanvasElement, color: [number, number, number]): Renderer | null {
  const gl = canvas.getContext("webgl2", { premultipliedAlpha: true, antialias: false, alpha: true });
  if (!gl) return null;

  const compile = (type: number, src: string) => {
    const shader = gl.createShader(type)!;
    gl.shaderSource(shader, src);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) ?? "shader");
    return shader;
  };
  const program = gl.createProgram();
  try {
    gl.attachShader(program, compile(gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(program, compile(gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error("link");
  } catch (err) {
    console.warn("Orb shader unavailable", err);
    return null;
  }

  const buffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(program, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.useProgram(program);

  const u = (name: string) => gl.getUniformLocation(program, name);
  const uniforms = {
    res: u("u_res"), time: u("u_time"), radius: u("u_radius"), width: u("u_width"),
    coverage: u("u_coverage"), rotation: u("u_rotation"), wobble: u("u_wobble"), color: u("u_color"),
  };
  gl.uniform3f(uniforms.color, ...color);

  return {
    draw(p, rotation, time) {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform2f(uniforms.res, canvas.width, canvas.height);
      gl.uniform1f(uniforms.time, time);
      gl.uniform1f(uniforms.radius, p.radius);
      gl.uniform1f(uniforms.width, p.width);
      gl.uniform1f(uniforms.coverage, p.coverage);
      gl.uniform1f(uniforms.rotation, rotation);
      gl.uniform1f(uniforms.wobble, p.wobble);
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    },
    dispose() {
      gl.deleteBuffer(buffer);
      gl.deleteProgram(program);
    },
  };
}

/** Plain arc for browsers without WebGL2. */
function canvasRenderer(canvas: HTMLCanvasElement, color: [number, number, number]): Renderer {
  const ctx = canvas.getContext("2d")!;
  const css = `rgb(${color.map((c) => Math.round(c * 255)).join(",")})`;
  return {
    draw(p, rotation) {
      const s = Math.min(canvas.width, canvas.height) / 2;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = css;
      ctx.lineCap = "round";
      ctx.lineWidth = p.width * s * 0.8;
      ctx.beginPath();
      // canvas y points down, so the clockwise stroke maps to a positive sweep
      ctx.arc(canvas.width / 2, canvas.height / 2, p.radius * s, -rotation, -rotation + p.coverage * TAU);
      ctx.stroke();
    },
    dispose() {},
  };
}

export function Orb({
  state,
  getLevel,
  color = "#d01e1e",
  className,
}: {
  state: OrbState;
  getLevel?: () => number;
  color?: string;
  className?: string;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef(state);
  const levelRef = useRef(getLevel);
  useLayoutEffect(() => {
    stateRef.current = state;
    levelRef.current = getLevel;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rgb = hexToRgb(color);
    const renderer = webglRenderer(canvas, rgb) ?? canvasRenderer(canvas, rgb);
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.max(1, Math.round(canvas.clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    };
    resize();
    const observer = new ResizeObserver(resize);
    observer.observe(canvas);

    let params = target("idle", 0, 0);
    let level = 0;
    let spinOffset = 0;
    let last = performance.now();
    let frame = 0;

    const tick = (now: number) => {
      const dt = Math.min(0.05, (now - last) / 1000);
      last = now;
      const time = now / 1000;
      const still = reduceMotion.matches;

      const raw = still ? 0 : Math.max(0, Math.min(1, levelRef.current?.() ?? 0));
      // Fast attack, slow release: syllables register, silence settles gently.
      level += (raw - level) * (raw > level ? 0.45 : 0.12);

      const goal = target(stateRef.current, level, still ? 0 : time);
      if (still) goal.wobble = 0;
      const k = 1 - Math.exp(-dt * 7);
      params = {
        radius: params.radius + (goal.radius - params.radius) * k,
        width: params.width + (goal.width - params.width) * k,
        coverage: params.coverage + (goal.coverage - params.coverage) * k,
        wobble: params.wobble + (goal.wobble - params.wobble) * k,
        spin: params.spin + (goal.spin - params.spin) * k,
      };

      if (!still && params.spin > 0.05) {
        spinOffset = (spinOffset + params.spin * dt) % TAU;
      } else {
        // Settle back to the logo's orientation along the shortest way round.
        const back = spinOffset > Math.PI ? TAU - spinOffset : -spinOffset;
        spinOffset = (spinOffset + back * k + TAU) % TAU;
      }

      renderer.draw(params, START_ANGLE - spinOffset, time);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.dispose();
    };
  }, [color]);

  return <canvas ref={canvasRef} className={cn("block aspect-square", className)} aria-hidden />;
}
