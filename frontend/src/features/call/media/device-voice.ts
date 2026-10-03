/**
 * The phone's own text-to-speech (Web Speech API), the last voice ZEN falls
 * back to when no server voice is available. Free, unlimited and instant, but
 * it plays outside Web Audio, so its loudness can't be measured for the orb.
 *
 * Some browsers ship without voices (or hide them for privacy, as Brave can);
 * then `ready` stays false and the call keeps to captions.
 */

const PREFERRED = [/natural/i, /google.*(us|uk) english/i, /samantha/i, /daniel/i, /google/i];

function pickVoice(voices: SpeechSynthesisVoice[]): SpeechSynthesisVoice | null {
  const english = voices.filter((v) => v.lang.toLowerCase().startsWith("en"));
  if (english.length === 0) return null;
  for (const pattern of PREFERRED) {
    const match = english.find((v) => pattern.test(v.name));
    if (match) return match;
  }
  return english.find((v) => v.localService) ?? english[0]!;
}

export class DeviceVoice {
  private voice: SpeechSynthesisVoice | null = null;
  private readonly synth: SpeechSynthesis | null;

  constructor() {
    this.synth = typeof window !== "undefined" && "speechSynthesis" in window ? window.speechSynthesis : null;
    if (!this.synth) return;
    const load = () => {
      this.voice = pickVoice(this.synth!.getVoices());
    };
    load();
    this.synth.addEventListener?.("voiceschanged", load);
  }

  get ready(): boolean {
    return this.voice !== null;
  }

  /** Call inside the user's tap: iOS only lets speech start from a gesture once. */
  prime(): void {
    if (!this.synth) return;
    const silent = new SpeechSynthesisUtterance(" ");
    silent.volume = 0;
    this.synth.speak(silent);
  }

  speak(text: string, events: { onStart: () => void; onEnd: () => void }): void {
    if (!this.synth || !this.voice) {
      events.onEnd();
      return;
    }
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.voice;
    utterance.lang = this.voice.lang;
    utterance.rate = 1.03;
    let ended = false;
    const end = () => {
      if (ended) return;
      ended = true;
      events.onEnd();
    };
    utterance.onstart = events.onStart;
    utterance.onend = end;
    utterance.onerror = end;
    this.synth.speak(utterance);
  }

  cancel(): void {
    this.synth?.cancel();
  }
}
