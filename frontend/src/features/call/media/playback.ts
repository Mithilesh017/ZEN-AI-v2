/**
 * Gapless playback of ZEN's reply, one segment per sentence.
 *
 * Segments are decoded as they arrive but always scheduled in order on the
 * AudioContext clock, back to back. Each segment's caption is revealed the
 * moment its audio starts, so text never runs ahead of the voice. Segments
 * without audio (no server voice available) are spoken by the phone's own
 * voice when it has one, or held on screen for a reading-speed estimate.
 *
 * stop() is the barge-in path: it silences everything immediately and
 * reports how many segments of the turn the user actually heard.
 */

import type { DeviceVoice } from "./device-voice";

export type Segment = { turn: number; seq: number; text: string };

type Scheduled = Segment & {
  source: AudioBufferSourceNode | null;
  start: number;
  end: number;
  /** spoken by the phone's voice; true while it is actually talking */
  device?: { speaking: boolean };
};

const READ_SECONDS_PER_WORD = 0.32;
const LOOKAHEAD = 0.05; // seconds of scheduling headroom

export class SpeechPlayer {
  onSegmentStart?: (segment: Segment) => void;
  onIdle?: () => void;

  private readonly ctx: AudioContext;
  private readonly deviceVoice: DeviceVoice | null;
  private readonly analyser: AnalyserNode;
  private readonly levels: Float32Array<ArrayBuffer>;
  private cursor = 0;
  private generation = 0;
  private chain: Promise<void> = Promise.resolve();
  private scheduled: Scheduled[] = [];
  private timers = new Set<ReturnType<typeof setTimeout>>();
  private started = new Map<number, number>(); // turn -> segments started
  private decoding = 0;

  constructor(ctx: AudioContext, deviceVoice: DeviceVoice | null = null) {
    this.ctx = ctx;
    this.deviceVoice = deviceVoice;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 512;
    this.analyser.smoothingTimeConstant = 0.6;
    this.analyser.connect(ctx.destination);
    this.levels = new Float32Array(this.analyser.fftSize);
  }

  /** Queue a segment; `wav` null means caption only. Order of calls is play order. */
  enqueue(segment: Segment, wav: ArrayBuffer | null): void {
    const generation = this.generation;
    this.decoding += 1;
    this.chain = this.chain.then(async () => {
      let buffer: AudioBuffer | null = null;
      if (wav) {
        try {
          buffer = await this.ctx.decodeAudioData(wav.slice(0));
        } catch {
          buffer = null; // undecodable audio still shows its caption
        }
      }
      if (generation !== this.generation) return;
      this.decoding -= 1;
      this.schedule(segment, buffer);
    });
  }

  /** True while anything is queued, decoding or audible. */
  get playing(): boolean {
    return this.scheduled.length > 0 || this.decoding > 0;
  }

  /** Segments of `turn` whose audio (or caption) has started. */
  heard(turn: number): number {
    return this.started.get(turn) ?? 0;
  }

  /** Current output loudness, 0–1, for the orb. */
  level(): number {
    if (this.scheduled.length === 0) return 0;
    if (this.scheduled.some((s) => s.device?.speaking)) {
      // The phone's voice can't be measured; give the orb a speech-like pulse.
      const t = performance.now() / 1000;
      return 0.35 + 0.2 * Math.sin(t * 11) * Math.sin(t * 3.7);
    }
    this.analyser.getFloatTimeDomainData(this.levels);
    let sum = 0;
    for (const v of this.levels) sum += v * v;
    return Math.min(1, Math.sqrt(sum / this.levels.length) * 4);
  }

  /** Silence everything now. Returns the turn that was cut off, if any. */
  stop(): { turn: number; played: number } | null {
    const current = this.scheduled[0];
    this.generation += 1;
    this.decoding = 0;
    this.chain = Promise.resolve();
    for (const s of this.scheduled) {
      try {
        s.source?.stop();
      } catch {
        // not started yet
      }
    }
    this.scheduled = [];
    this.deviceVoice?.cancel();
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.cursor = 0;
    return current ? { turn: current.turn, played: this.heard(current.turn) } : null;
  }

  private schedule(segment: Segment, buffer: AudioBuffer | null): void {
    const now = this.ctx.currentTime;
    const start = Math.max(now + LOOKAHEAD, this.cursor);
    const duration = buffer
      ? buffer.duration
      : Math.max(1.2, segment.text.split(/\s+/).length * READ_SECONDS_PER_WORD);
    const entry: Scheduled = { ...segment, source: null, start, end: start + duration };
    this.cursor = entry.end;

    if (buffer) {
      const source = this.ctx.createBufferSource();
      source.buffer = buffer;
      source.connect(this.analyser);
      source.start(start);
      entry.source = source;
    } else if (this.deviceVoice?.ready) {
      this.scheduled.push(entry);
      this.speakOnDevice(entry);
      return;
    }
    this.scheduled.push(entry);

    this.at(start, () => {
      this.started.set(segment.turn, Math.max(this.heard(segment.turn), segment.seq));
      this.onSegmentStart?.(segment);
    });
    this.at(entry.end, () => {
      this.scheduled = this.scheduled.filter((s) => s !== entry);
      if (!this.playing) this.onIdle?.();
    });
  }

  /**
   * The phone's voice runs on its own clock: the caption switches when it
   * really starts talking, and the segment ends when it really stops. The
   * estimated duration only spaces out any server audio queued after it.
   */
  private speakOnDevice(entry: Scheduled): void {
    const generation = this.generation;
    entry.device = { speaking: false };
    this.at(entry.start, () => {
      this.deviceVoice!.speak(entry.text, {
        onStart: () => {
          if (generation !== this.generation) return;
          entry.device!.speaking = true;
          this.started.set(entry.turn, Math.max(this.heard(entry.turn), entry.seq));
          this.onSegmentStart?.(entry);
        },
        onEnd: () => {
          if (generation !== this.generation) return;
          entry.device!.speaking = false;
          this.scheduled = this.scheduled.filter((s) => s !== entry);
          if (!this.playing) this.onIdle?.();
        },
      });
    });
  }

  private at(time: number, fn: () => void): void {
    const delay = Math.max(0, (time - this.ctx.currentTime) * 1000);
    const timer = setTimeout(() => {
      this.timers.delete(timer);
      fn();
    }, delay);
    this.timers.add(timer);
  }
}
