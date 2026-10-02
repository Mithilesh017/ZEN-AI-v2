/**
 * Draws tracked objects over the camera as hairline corner brackets with a
 * small label. The tracker updates ~12 times a second; this layer eases each
 * box toward its latest position every display frame so motion stays fluid.
 */
import { useEffect, useLayoutEffect, useRef } from "react";

import { coverTransform, type Box, type Track } from "../vision/tracker";

const FONT = "500 11px ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, monospace";

export function DetectionOverlay({
  video,
  tracks,
  mirrored,
}: {
  video: HTMLVideoElement | null;
  tracks: () => Track[];
  mirrored: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mirroredRef = useRef(mirrored);
  useLayoutEffect(() => {
    mirroredRef.current = mirrored;
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !video) return;

    const shown = new Map<number, Box>();
    let frame = 0;

    const draw = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const W = canvas.clientWidth;
      const H = canvas.clientHeight;
      if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
        canvas.width = Math.round(W * dpr);
        canvas.height = Math.round(H * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, W, H);

      const current = tracks();
      const live = new Set(current.map((t) => t.id));
      for (const id of shown.keys()) if (!live.has(id)) shown.delete(id);

      if (video.videoWidth && video.videoHeight) {
        const dims = { width: video.videoWidth, height: video.videoHeight };
        for (const track of current) {
          const goal = coverTransform(track.box, dims, { width: W, height: H }, mirroredRef.current);
          const prev = shown.get(track.id) ?? goal;
          const box = {
            x: prev.x + (goal.x - prev.x) * 0.3,
            y: prev.y + (goal.y - prev.y) * 0.3,
            w: prev.w + (goal.w - prev.w) * 0.3,
            h: prev.h + (goal.h - prev.h) * 0.3,
          };
          shown.set(track.id, box);
          drawTrack(ctx, box, track, W);
        }
      }
      frame = requestAnimationFrame(draw);
    };
    frame = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(frame);
  }, [video, tracks]);

  return <canvas ref={canvasRef} className="pointer-events-none absolute inset-0 size-full" aria-hidden />;
}

function drawTrack(ctx: CanvasRenderingContext2D, b: Box, track: Track, width: number) {
  const alpha = track.opacity;
  if (alpha <= 0.01) return;
  const arm = Math.max(8, Math.min(22, Math.min(b.w, b.h) * 0.18));

  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "rgba(255,255,255,0.92)";
  ctx.lineWidth = 1.5;
  ctx.lineCap = "round";
  ctx.beginPath();
  // four corner brackets
  for (const [x, y, dx, dy] of [
    [b.x, b.y, 1, 1],
    [b.x + b.w, b.y, -1, 1],
    [b.x, b.y + b.h, 1, -1],
    [b.x + b.w, b.y + b.h, -1, -1],
  ] as const) {
    ctx.moveTo(x, y + dy * arm);
    ctx.lineTo(x, y);
    ctx.lineTo(x + dx * arm, y);
  }
  ctx.stroke();

  const text = `${track.label}  ${Math.round(track.score * 100)}`;
  ctx.font = FONT;
  const tw = ctx.measureText(text).width;
  const lx = Math.min(Math.max(4, b.x), width - tw - 16);
  const ly = b.y > 26 ? b.y - 24 : b.y + 6;
  ctx.fillStyle = "rgba(0,0,0,0.55)";
  ctx.beginPath();
  ctx.roundRect(lx, ly, tw + 12, 18, 4);
  ctx.fill();
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.textBaseline = "middle";
  ctx.fillText(text, lx + 6, ly + 9.5);
  ctx.globalAlpha = 1;
}
