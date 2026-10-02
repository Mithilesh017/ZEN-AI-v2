/**
 * Camera ownership for a call: open, flip, turn off, and capture a still.
 *
 * Flipping stops the current track before opening the other camera, because
 * many Android devices cannot hold both open at once. Turning the camera off
 * stops the track so the hardware indicator light goes out too.
 */
import type { Facing } from "../transport/types";

const MAX_FRAME_SIDE = 768;
const FRAME_QUALITY = 0.72;

export class CameraError extends Error {
  readonly kind: "denied" | "missing" | "busy";

  constructor(kind: "denied" | "missing" | "busy", message: string) {
    super(message);
    this.kind = kind;
  }
}

export const mediaErrorKind = (err: unknown): "denied" | "missing" | "busy" => {
  const name = err instanceof DOMException ? err.name : "";
  if (name === "NotAllowedError" || name === "SecurityError") return "denied";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "missing";
  return "busy";
};

export class Camera {
  stream: MediaStream | null = null;
  facing: Facing = "user";

  async open(facing: Facing): Promise<MediaStream> {
    this.close();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: facing },
          width: { ideal: 1280 },
          height: { ideal: 720 },
          frameRate: { ideal: 30, max: 30 },
        },
        audio: false,
      });
    } catch (err) {
      const kind = mediaErrorKind(err);
      throw new CameraError(kind, err instanceof Error ? err.message : String(err));
    }
    // Laptops report no facingMode; their only camera faces the user.
    const actual = this.stream.getVideoTracks()[0]?.getSettings().facingMode;
    this.facing = actual === "environment" ? "environment" : actual === "user" ? "user" : facing;
    return this.stream;
  }

  close(): void {
    this.stream?.getTracks().forEach((t) => t.stop());
    this.stream = null;
  }

  static async hasMultiple(): Promise<boolean> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.filter((d) => d.kind === "videoinput").length > 1;
    } catch {
      return false;
    }
  }
}

/** The current video frame as a JPEG, longest side ≤ 768 px, never mirrored. */
export async function captureFrame(video: HTMLVideoElement): Promise<ArrayBuffer | null> {
  const { videoWidth: w, videoHeight: h } = video;
  if (!w || !h || video.readyState < 2) return null;
  const scale = Math.min(1, MAX_FRAME_SIDE / Math.max(w, h));
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(w * scale);
  canvas.height = Math.round(h * scale);
  canvas.getContext("2d")?.drawImage(video, 0, 0, canvas.width, canvas.height);
  const blob = await new Promise<Blob | null>((resolve) =>
    canvas.toBlob(resolve, "image/jpeg", FRAME_QUALITY),
  );
  return blob ? blob.arrayBuffer() : null;
}
