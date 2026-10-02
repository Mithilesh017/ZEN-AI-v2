/**
 * On-device voice activity detection (Silero VAD v5 in WebAssembly).
 *
 * Running turn detection on the phone is what makes the call feel live:
 * ZEN knows the user stopped talking without a server round trip, and can
 * be interrupted the instant they start. All model and runtime files are
 * served from our own origin (copied at build time), never a CDN.
 */
import { MicVAD } from "@ricky0123/vad-web";

import { loudness } from "./wav";

export const VAD_ASSETS = `${import.meta.env.BASE_URL}call-assets/`;

export type VadCallbacks = {
  /** Confirmed speech (not a cough or a door): safe to treat as barge-in. */
  onSpeech: () => void;
  /** A complete utterance, 16 kHz mono. */
  onUtterance: (audio: Float32Array) => void;
  onMisfire: () => void;
  onLevel: (level: number) => void;
};

/** While ZEN is talking, demand clearer speech so its own voice can't trigger a barge-in. */
const THRESHOLDS = {
  normal: { positiveSpeechThreshold: 0.5, negativeSpeechThreshold: 0.35 },
  guarded: { positiveSpeechThreshold: 0.82, negativeSpeechThreshold: 0.6 },
};

export class VoiceDetector {
  private readonly vad: MicVAD;

  private constructor(vad: MicVAD) {
    this.vad = vad;
  }

  static async create(
    mic: MediaStream,
    audioContext: AudioContext,
    callbacks: VadCallbacks,
  ): Promise<VoiceDetector> {
    const vad = await MicVAD.new({
      model: "v5",
      baseAssetPath: VAD_ASSETS,
      onnxWASMBasePath: VAD_ASSETS,
      audioContext,
      // The call owns the microphone; muting pauses detection without
      // stopping the track (which would re-prompt on some browsers).
      getStream: async () => mic,
      pauseStream: async () => {},
      resumeStream: async () => mic,
      startOnLoad: false,
      ortConfig: (ort) => {
        ort.env.logLevel = "error";
        ort.env.wasm.numThreads = 1; // no SharedArrayBuffer without cross-origin isolation
      },
      ...THRESHOLDS.normal,
      redemptionMs: 650,
      preSpeechPadMs: 320,
      minSpeechMs: 260,
      onSpeechStart: () => {},
      onSpeechRealStart: callbacks.onSpeech,
      onSpeechEnd: callbacks.onUtterance,
      onVADMisfire: callbacks.onMisfire,
      onFrameProcessed: (_probs, frame) => callbacks.onLevel(loudness(frame)),
    });
    return new VoiceDetector(vad);
  }

  start(): Promise<void> {
    return this.vad.start();
  }

  pause(): Promise<void> {
    return this.vad.pause();
  }

  setGuarded(guarded: boolean): void {
    this.vad.setOptions(guarded ? THRESHOLDS.guarded : THRESHOLDS.normal);
  }

  destroy(): Promise<void> {
    return this.vad.destroy();
  }
}
