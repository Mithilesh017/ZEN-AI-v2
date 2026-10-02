/**
 * On-device object detection: MediaPipe EfficientDet-Lite0 (COCO, int8).
 *
 * Runs on the GPU delegate where available and is paced to the video's own
 * frames, capped at a target rate. If frames take too long (older phones,
 * thermal throttling) the rate backs off rather than stealing time from
 * audio and rendering.
 */
import { FilesetResolver, ObjectDetector } from "@mediapipe/tasks-vision";

import type { RawDetection } from "./tracker";

const WASM_PATH = `${import.meta.env.BASE_URL}call-assets/mediapipe`;
const MODEL_PATH = `${import.meta.env.BASE_URL}models/efficientdet_lite0.tflite`;

const TARGET_FPS = 12;
const MIN_FPS = 4;
const SLOW_FRAME_MS = 70;

type VideoWithCallback = HTMLVideoElement & {
  requestVideoFrameCallback?: (cb: () => void) => number;
  cancelVideoFrameCallback?: (id: number) => void;
};

export class Detector {
  private readonly detector: ObjectDetector;
  private video: VideoWithCallback | null = null;
  private handle: number | null = null;
  private fps = TARGET_FPS;
  private lastRun = 0;
  private onResult: ((detections: RawDetection[], now: number) => void) | null = null;

  private constructor(detector: ObjectDetector) {
    this.detector = detector;
  }

  static async create(): Promise<Detector> {
    const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
    const options = (delegate: "GPU" | "CPU") => ({
      baseOptions: { modelAssetPath: MODEL_PATH, delegate },
      runningMode: "VIDEO" as const,
      scoreThreshold: 0.45,
      maxResults: 8,
    });
    try {
      return new Detector(await ObjectDetector.createFromOptions(fileset, options("GPU")));
    } catch {
      return new Detector(await ObjectDetector.createFromOptions(fileset, options("CPU")));
    }
  }

  start(video: HTMLVideoElement, onResult: (detections: RawDetection[], now: number) => void): void {
    this.stop();
    this.video = video;
    this.onResult = onResult;
    this.next();
  }

  stop(): void {
    if (this.handle !== null && this.video) {
      if (this.video.cancelVideoFrameCallback) this.video.cancelVideoFrameCallback(this.handle);
      else cancelAnimationFrame(this.handle);
    }
    this.handle = null;
    this.video = null;
  }

  close(): void {
    this.stop();
    this.detector.close();
  }

  private next(): void {
    const video = this.video;
    if (!video) return;
    const tick = () => this.tick();
    this.handle = video.requestVideoFrameCallback
      ? video.requestVideoFrameCallback(tick)
      : requestAnimationFrame(tick);
  }

  private tick(): void {
    const video = this.video;
    if (!video) return;
    const now = performance.now();
    if (now - this.lastRun >= 1000 / this.fps && video.readyState >= 2 && !document.hidden) {
      this.lastRun = now;
      const result = this.detector.detectForVideo(video, now);
      const took = performance.now() - now;
      this.fps = took > SLOW_FRAME_MS ? Math.max(MIN_FPS, this.fps - 1) : Math.min(TARGET_FPS, this.fps + 0.25);

      const { videoWidth: w, videoHeight: h } = video;
      const detections: RawDetection[] = result.detections.flatMap((d) => {
        const category = d.categories[0];
        const b = d.boundingBox;
        if (!category || !b || !w || !h) return [];
        return [{
          label: category.categoryName,
          score: category.score,
          box: { x: b.originX / w, y: b.originY / h, w: b.width / w, h: b.height / h },
        }];
      });
      this.onResult?.(detections, now);
    }
    this.next();
  }
}
