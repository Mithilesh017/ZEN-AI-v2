/**
 * Turns raw per-frame detections into stable on-screen objects.
 *
 * A detector's raw output flickers: boxes jump a few pixels every frame and
 * objects blink in and out at the score threshold. Matching detections to
 * existing tracks (same label, best IoU), easing the box towards each new
 * measurement, requiring a few consecutive hits before showing a track, and
 * letting it linger briefly when missed is what makes the overlay feel solid.
 */

export type Box = { x: number; y: number; w: number; h: number }; // normalised 0–1

export type RawDetection = { label: string; score: number; box: Box };

export type Track = {
  id: number;
  label: string;
  score: number;
  box: Box;
  hits: number;
  lastSeen: number;
  visible: boolean;
  /** 0–1, for fading in and out */
  opacity: number;
};

export type TrackerOptions = {
  minHits: number;
  lingerMs: number;
  minIou: number;
  smoothing: number; // weight of the new measurement, 0–1
};

const DEFAULTS: TrackerOptions = { minHits: 3, lingerMs: 300, minIou: 0.3, smoothing: 0.4 };

export function iou(a: Box, b: Box): number {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

const lerp = (a: number, b: number, t: number) => a + (b - a) * t;

export class Tracker {
  private tracks: Track[] = [];
  private nextId = 1;
  private readonly opts: TrackerOptions;
  /** Called once per track when it first becomes visible. */
  onAppear?: (track: Track) => void;

  constructor(options: Partial<TrackerOptions> = {}) {
    this.opts = { ...DEFAULTS, ...options };
  }

  update(detections: RawDetection[], now: number): Track[] {
    const { minHits, lingerMs, minIou, smoothing } = this.opts;
    const unmatched = new Set(this.tracks);

    for (const det of [...detections].sort((a, b) => b.score - a.score)) {
      let best: Track | undefined;
      let bestIou = minIou;
      for (const track of unmatched) {
        if (track.label !== det.label) continue;
        const overlap = iou(track.box, det.box);
        if (overlap >= bestIou) {
          best = track;
          bestIou = overlap;
        }
      }

      if (best) {
        unmatched.delete(best);
        best.box = {
          x: lerp(best.box.x, det.box.x, smoothing),
          y: lerp(best.box.y, det.box.y, smoothing),
          w: lerp(best.box.w, det.box.w, smoothing),
          h: lerp(best.box.h, det.box.h, smoothing),
        };
        best.score = lerp(best.score, det.score, smoothing);
        best.hits += 1;
        best.lastSeen = now;
        if (!best.visible && best.hits >= minHits) {
          best.visible = true;
          this.onAppear?.(best);
        }
      } else {
        this.tracks.push({
          id: this.nextId++,
          label: det.label,
          score: det.score,
          box: det.box,
          hits: 1,
          lastSeen: now,
          visible: minHits <= 1,
          opacity: 0,
        });
        if (minHits <= 1) this.onAppear?.(this.tracks.at(-1)!);
      }
    }

    // A missed track must be re-confirmed if it was never shown.
    for (const track of unmatched) if (!track.visible) track.hits = 0;

    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= lingerMs && (t.visible || t.hits > 0));
    for (const t of this.tracks) {
      const fresh = now - t.lastSeen;
      t.opacity = t.visible ? Math.max(0, 1 - fresh / lingerMs) : 0;
    }
    return this.tracks;
  }

  visible(): Track[] {
    return this.tracks.filter((t) => t.visible);
  }

  reset(): void {
    this.tracks = [];
  }
}

/**
 * Where a normalised box lands inside an element showing the video with
 * `object-fit: cover` (centre-cropped), optionally mirrored (front camera).
 */
export function coverTransform(
  box: Box,
  video: { width: number; height: number },
  element: { width: number; height: number },
  mirrored: boolean,
): Box {
  const scale = Math.max(element.width / video.width, element.height / video.height);
  const offsetX = (element.width - video.width * scale) / 2;
  const offsetY = (element.height - video.height * scale) / 2;
  const w = box.w * video.width * scale;
  const h = box.h * video.height * scale;
  let x = box.x * video.width * scale + offsetX;
  const y = box.y * video.height * scale + offsetY;
  if (mirrored) x = element.width - x - w;
  return { x, y, w, h };
}
