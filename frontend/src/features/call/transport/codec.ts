/**
 * Binary framing shared with models/vision/protocol.py. Integers are big-endian.
 *
 *   utterance  u8 0x01 | u32 header length | header JSON | WAV | JPEG
 *   speech     u8 0x10 | u32 turn | u16 seq | WAV
 */
import type { SpeechChunk, Utterance } from "./types";

export const KIND_UTTERANCE = 0x01;
export const KIND_SPEECH = 0x10;

const encoder = new TextEncoder();

export function encodeUtterance(u: Utterance): ArrayBuffer {
  const header = encoder.encode(
    JSON.stringify({
      turn: u.turn,
      audio_bytes: u.wav.byteLength,
      camera: u.camera,
      facing: u.facing,
      detections: u.detections,
    }),
  );
  const image = u.image?.byteLength ?? 0;
  const out = new Uint8Array(5 + header.byteLength + u.wav.byteLength + image);
  const view = new DataView(out.buffer);
  view.setUint8(0, KIND_UTTERANCE);
  view.setUint32(1, header.byteLength);
  out.set(header, 5);
  out.set(new Uint8Array(u.wav), 5 + header.byteLength);
  if (u.image) out.set(new Uint8Array(u.image), 5 + header.byteLength + u.wav.byteLength);
  return out.buffer;
}

export function decodeSpeech(data: ArrayBuffer): SpeechChunk | null {
  if (data.byteLength < 7) return null;
  const view = new DataView(data);
  if (view.getUint8(0) !== KIND_SPEECH) return null;
  return { turn: view.getUint32(1), seq: view.getUint16(5), wav: data.slice(7) };
}
