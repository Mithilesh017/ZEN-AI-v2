import { describe, expect, it } from "vitest";

import { encodeWav, loudness } from "./media/wav";
import { decodeSpeech, encodeUtterance, KIND_SPEECH, KIND_UTTERANCE } from "./transport/codec";
import { coverTransform, iou, Tracker, type RawDetection } from "./vision/tracker";

describe("wav encoding", () => {
  it("writes a valid 16 kHz mono PCM16 header and clamps samples", () => {
    const wav = encodeWav(new Float32Array([0, 1, -1, 2]));
    const view = new DataView(wav);
    const text = (o: number, n: number) => String.fromCharCode(...new Uint8Array(wav, o, n));
    expect(text(0, 4)).toBe("RIFF");
    expect(text(8, 4)).toBe("WAVE");
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint16(34, true)).toBe(16);
    expect(view.getUint32(40, true)).toBe(8);
    expect([0, 1, 2, 3].map((i) => view.getInt16(44 + i * 2, true))).toEqual([0, 32767, -32768, 32767]);
  });

  it("maps silence to 0 and a loud signal near 1", () => {
    expect(loudness(new Float32Array(512))).toBe(0);
    expect(loudness(new Float32Array(512).fill(0.5))).toBeGreaterThan(0.9);
  });
});

describe("wire codec", () => {
  it("frames an utterance as kind | header length | header | wav | jpeg", () => {
    const wav = new Uint8Array([82, 73, 70, 70, 1, 2]).buffer; // "RIFF.."
    const jpeg = new Uint8Array([0xff, 0xd8, 9]).buffer;
    const data = encodeUtterance({
      turn: 3, wav, image: jpeg, camera: true, facing: "environment",
      detections: [{ label: "cup", score: 0.9 }],
    });
    const view = new DataView(data);
    expect(view.getUint8(0)).toBe(KIND_UTTERANCE);
    const headerLength = view.getUint32(1);
    const header = JSON.parse(new TextDecoder().decode(new Uint8Array(data, 5, headerLength)));
    expect(header).toEqual({
      turn: 3, audio_bytes: 6, camera: true, facing: "environment", detections: [{ label: "cup", score: 0.9 }],
    });
    expect([...new Uint8Array(data, 5 + headerLength)]).toEqual([82, 73, 70, 70, 1, 2, 0xff, 0xd8, 9]);
  });

  it("decodes speech frames and rejects anything else", () => {
    const data = new Uint8Array([KIND_SPEECH, 0, 0, 0, 7, 0, 2, 0xaa, 0xbb]).buffer;
    const chunk = decodeSpeech(data)!;
    expect([chunk.turn, chunk.seq, [...new Uint8Array(chunk.wav)]]).toEqual([7, 2, [0xaa, 0xbb]]);
    expect(decodeSpeech(new Uint8Array([1, 0, 0, 0, 7, 0, 2]).buffer)).toBeNull();
    expect(decodeSpeech(new Uint8Array([KIND_SPEECH]).buffer)).toBeNull();
  });
});

describe("tracker", () => {
  const det = (label: string, x: number, score = 0.8): RawDetection => ({
    label, score, box: { x, y: 0.2, w: 0.2, h: 0.2 },
  });

  it("computes intersection over union", () => {
    const a = { x: 0, y: 0, w: 1, h: 1 };
    expect(iou(a, a)).toBe(1);
    expect(iou(a, { x: 2, y: 2, w: 1, h: 1 })).toBe(0);
    expect(iou(a, { x: 0.5, y: 0, w: 1, h: 1 })).toBeCloseTo(1 / 3);
  });

  it("only shows an object after consecutive sightings, and reports it once", () => {
    const appeared: string[] = [];
    const tracker = new Tracker({ minHits: 3 });
    tracker.onAppear = (t) => appeared.push(t.label);
    tracker.update([det("cup", 0.1)], 0);
    tracker.update([det("cup", 0.11)], 80);
    expect(tracker.visible()).toHaveLength(0);
    tracker.update([det("cup", 0.12)], 160);
    tracker.update([det("cup", 0.12)], 240);
    expect(tracker.visible().map((t) => t.label)).toEqual(["cup"]);
    expect(appeared).toEqual(["cup"]);
  });

  it("smooths box movement instead of jumping", () => {
    const tracker = new Tracker({ minHits: 1, smoothing: 0.5 });
    tracker.update([det("cup", 0.1)], 0);
    const [track] = tracker.update([det("cup", 0.2)], 80);
    expect(track!.box.x).toBeCloseTo(0.15);
  });

  it("keeps a missed object briefly, fading out, then drops it", () => {
    const tracker = new Tracker({ minHits: 1, lingerMs: 300 });
    tracker.update([det("dog", 0.4)], 0);
    const [fading] = tracker.update([], 150);
    expect(fading!.opacity).toBeCloseTo(0.5);
    expect(tracker.update([], 400)).toHaveLength(0);
  });

  it("does not match objects of different labels", () => {
    const tracker = new Tracker({ minHits: 1 });
    tracker.update([det("cup", 0.1)], 0);
    expect(tracker.update([det("bowl", 0.1)], 50).map((t) => t.label).sort()).toEqual(["bowl", "cup"]);
  });
});

describe("cover transform", () => {
  const video = { width: 1280, height: 720 };

  it("maps a box through a centre crop", () => {
    // 16:9 video in a 9:16 phone screen is scaled to fill the height and cropped at the sides.
    const box = coverTransform({ x: 0.5, y: 0, w: 0.1, h: 0.5 }, video, { width: 405, height: 720 }, false);
    expect(box.x).toBeCloseTo(640 - (1280 - 405) / 2);
    expect(box.y).toBe(0);
    expect(box.w).toBeCloseTo(128);
    expect(box.h).toBeCloseTo(360);
  });

  it("mirrors for the front camera", () => {
    const element = { width: 1280, height: 720 };
    const box = coverTransform({ x: 0, y: 0, w: 0.25, h: 0.5 }, video, element, true);
    expect(box.x).toBeCloseTo(960);
  });
});
