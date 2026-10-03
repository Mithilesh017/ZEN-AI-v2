/**
 * Orchestrates one call: microphone + VAD, camera + detector, the transport
 * and speech playback. React components render from the call store and call
 * the methods here; nothing in this file touches the DOM beyond the <video>
 * element it is handed.
 *
 * Turn lifecycle
 *   VAD speech ─▶ "hearing" ─▶ utterance + frame sent ─▶ "thinking"
 *   ─▶ first sentence starts playing ─▶ "speaking" ─▶ playback drained ─▶ "listening"
 * Speech detected while ZEN is thinking or speaking cancels that turn
 * (barge-in) and tells the server how much of it was heard.
 */
import { appendLine, initialCallState, noteSeen, setCall, useCallStore, type CallErrorKind } from "./call-store";
import { Camera, CameraError, captureFrame, mediaErrorKind } from "./media/camera";
import { DeviceVoice } from "./media/device-voice";
import { SpeechPlayer } from "./media/playback";
import { VoiceDetector } from "./media/vad";
import { encodeWav } from "./media/wav";
import { CallFailure, GatewayTransport } from "./transport/gateway";
import type { CallTransport, EndReason, LinkStatus, ServerEvent, SpeechChunk } from "./transport/types";
import type { Detector } from "./vision/detector";
import { Tracker, type Track } from "./vision/tracker";

const NOTICE_MS = 4500;
const CAPTION_LINGER_MS = 3500;

const haptic = (ms = 8) => {
  try {
    navigator.vibrate?.(ms);
  } catch {
    // unsupported
  }
};

export class CallController {
  private transport: CallTransport = new GatewayTransport();
  private camera = new Camera();
  private tracker = new Tracker();
  private ctx: AudioContext | null = null;
  private mic: MediaStream | null = null;
  private vad: VoiceDetector | null = null;
  private player: SpeechPlayer | null = null;
  private deviceVoice = new DeviceVoice();
  private detector: Detector | null = null;
  private detectorLoading: Promise<void> | null = null;
  private video: HTMLVideoElement | null = null;
  private wakeLock: WakeLockSentinel | null = null;

  private turnSeq = 0;
  private currentTurn = 0;
  private serverDone = 0;
  private saysThisTurn = 0;
  private pendingSay = new Map<string, string>();
  private micLevel = 0;
  private cameraSuspended = false;
  private disposed = false;
  private timers = new Map<string, ReturnType<typeof setTimeout>>();

  constructor() {
    this.tracker.onAppear = (track) => noteSeen(track.label);
  }

  // ── lifecycle ───────────────────────────────────────────

  async start(): Promise<void> {
    const { cameraOn, facing } = useCallStore.getState();
    if (!window.isSecureContext) return this.fail("insecure", "Calls need a secure (HTTPS) connection.");
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext || !window.WebSocket) {
      return this.fail("unsupported", "This browser can't make calls. Try the latest Chrome or Safari.");
    }

    setCall({
      ...initialCallState,
      phase: "connecting",
      cameraOn,
      facing,
      detectionsOn: useCallStore.getState().detectionsOn,
    });

    // Created inside the user's tap so mobile browsers allow audio output.
    this.ctx = new AudioContext();
    void this.ctx.resume();
    this.deviceVoice.prime();

    try {
      this.mic = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      const kind = mediaErrorKind(err);
      return this.fail(kind === "busy" ? "missing" : kind, "ZEN needs your microphone to hear you.");
    }
    if (this.disposed) return this.teardown();

    if (cameraOn) {
      try {
        await this.openCamera(facing);
      } catch (err) {
        // A call without video is still useful; carry on voice-only.
        setCall({ cameraOn: false });
        this.notify(
          err instanceof CameraError && err.kind === "denied"
            ? "Camera is blocked, so this call is voice only."
            : "Couldn't start the camera, so this call is voice only.",
        );
      }
    }
    void Camera.hasMultiple().then((canFlip) => setCall({ canFlip }));

    this.player = new SpeechPlayer(this.ctx, this.deviceVoice);
    this.player.onSegmentStart = (segment) => {
      if (segment.turn !== this.currentTurn) return;
      this.clearTimer("caption");
      this.vad?.setGuarded(true);
      setCall({ zen: "speaking", zenCaption: segment.text });
      appendLine({ turn: segment.turn, role: "zen", text: segment.text });
    };
    this.player.onIdle = () => {
      this.vad?.setGuarded(false);
      if (this.serverDone === this.currentTurn && useCallStore.getState().zen === "speaking") {
        setCall({ zen: "listening" });
      }
      this.later("caption", CAPTION_LINGER_MS, () => setCall({ zenCaption: "" }));
    };

    try {
      const [vad, ready] = await Promise.all([
        VoiceDetector.create(this.mic, this.ctx, {
          onSpeech: () => this.onUserSpeech(),
          onUtterance: (audio) => void this.onUtterance(audio),
          onMisfire: () => {},
          onLevel: (level) => {
            this.micLevel = level;
          },
        }),
        this.transport.connect({
          onEvent: (e) => this.onServerEvent(e),
          onSpeech: (c) => this.onSpeech(c),
          onStatus: (s) => this.onLinkStatus(s),
        }),
      ]);
      this.vad = vad;
      if (this.disposed) return this.teardown();
      await vad.start();
      setCall({ phase: "live", startedAt: Date.now(), maxSeconds: ready.max_seconds, voice: ready.voice });
      if (!ready.voice) this.voiceFallbackNotice("ZEN's voice is resting for now, so replies will be in captions.");
    } catch (err) {
      if (err instanceof CallFailure) return this.fail(err.kind, err.message);
      console.error("Call setup failed", err);
      return this.fail("unsupported", "Voice detection couldn't start on this device.");
    }

    void this.acquireWakeLock();
    document.addEventListener("visibilitychange", this.onVisibility);
    window.addEventListener("pagehide", this.onPageHide);
    this.syncDetector();
  }

  /** The user pressed "end call". */
  end(): void {
    if (useCallStore.getState().phase !== "live") return this.dispose();
    haptic(20);
    this.transport.hangUp();
    this.finish("hangup");
  }

  /** Release everything; safe to call more than once. */
  dispose(): void {
    this.disposed = true;
    this.transport.close();
    this.teardown();
  }

  // ── controls ────────────────────────────────────────────

  async toggleMute(): Promise<void> {
    const muted = !useCallStore.getState().muted;
    haptic();
    setCall({ muted });
    this.mic?.getAudioTracks().forEach((t) => (t.enabled = !muted));
    if (muted) {
      this.micLevel = 0;
      await this.vad?.pause();
    } else {
      await this.vad?.start();
    }
  }

  async toggleCamera(): Promise<void> {
    const { cameraOn, facing } = useCallStore.getState();
    haptic();
    if (cameraOn) {
      this.camera.close();
      if (this.video) this.video.srcObject = null;
      setCall({ cameraOn: false });
    } else {
      try {
        await this.openCamera(facing);
        setCall({ cameraOn: true });
      } catch {
        this.notify("Couldn't turn the camera on.");
      }
    }
    this.syncDetector();
  }

  async flip(): Promise<void> {
    const { cameraOn, facing } = useCallStore.getState();
    if (!cameraOn) return;
    haptic();
    try {
      await this.openCamera(facing === "user" ? "environment" : "user");
    } catch {
      await this.openCamera(facing).catch(() => setCall({ cameraOn: false }));
      this.notify("Couldn't switch cameras.");
    }
    this.tracker.reset();
    this.syncDetector();
  }

  toggleDetections(): void {
    haptic();
    setCall({ detectionsOn: !useCallStore.getState().detectionsOn });
    this.syncDetector();
  }

  attachVideo(video: HTMLVideoElement | null): void {
    this.video = video;
    if (video) video.srcObject = this.camera.stream;
    this.syncDetector();
  }

  // ── read by the view's animation loops ─────────────────

  orbLevel(): number {
    return useCallStore.getState().zen === "speaking" ? (this.player?.level() ?? 0) : this.micLevel;
  }

  tracks(): Track[] {
    const { cameraOn, detectionsOn } = useCallStore.getState();
    return cameraOn && detectionsOn ? this.tracker.visible() : [];
  }

  // ── turn handling ───────────────────────────────────────

  private onUserSpeech(): void {
    const { zen } = useCallStore.getState();
    if (zen === "thinking" || zen === "speaking" || this.player?.playing) {
      // Barge-in: stop talking now, and tell the server how much was heard.
      this.player?.stop();
      this.transport.interrupt(this.currentTurn, this.player?.heard(this.currentTurn) ?? 0);
      this.vad?.setGuarded(false);
    }
    this.currentTurn = 0; // ignore anything still arriving for the old turn
    this.pendingSay.clear();
    this.clearTimer("caption");
    setCall({ zen: "hearing", zenCaption: "" });
  }

  private async onUtterance(audio: Float32Array): Promise<void> {
    const state = useCallStore.getState();
    if (state.muted || state.phase !== "live") return;
    const turn = ++this.turnSeq;
    this.currentTurn = turn;
    this.saysThisTurn = 0;
    setCall({ zen: "thinking", userCaption: "" });

    const image = state.cameraOn && this.video ? await captureFrame(this.video) : null;
    this.transport.sendUtterance({
      turn,
      wav: encodeWav(audio),
      image,
      camera: image !== null,
      facing: state.facing,
      detections: this.tracker
        .visible()
        .slice(0, 8)
        .map((t) => ({ label: t.label, score: Math.round(t.score * 100) / 100 })),
    });
  }

  private onServerEvent(event: ServerEvent): void {
    switch (event.type) {
      case "transcript":
        if (event.turn !== this.currentTurn) return;
        setCall({ userCaption: event.text });
        appendLine({ turn: event.turn, role: "user", text: event.text });
        return;

      case "say":
        if (event.turn !== this.currentTurn) return;
        this.saysThisTurn += 1;
        // Audio again after a quota pause means the voice is back.
        if (event.audio && !useCallStore.getState().voice) setCall({ voice: true });
        if (event.audio) this.pendingSay.set(`${event.turn}:${event.seq}`, event.text);
        else this.player?.enqueue({ turn: event.turn, seq: event.seq, text: event.text }, null);
        return;

      case "turn_done":
        if (event.turn !== this.currentTurn) return;
        this.serverDone = event.turn;
        if (event.timings) setCall({ timings: event.timings });
        if (event.skipped === "error") this.notify("I didn't catch that. Try again?");
        if (event.skipped || this.saysThisTurn === 0 || !this.player?.playing) {
          setCall({ zen: "listening", ...(event.skipped ? { userCaption: "" } : {}) });
        }
        return;

      case "notice":
        if (event.code === "voice_unavailable" || event.code === "voice_limited") {
          setCall({ voice: false });
          this.voiceFallbackNotice(event.message);
          return;
        }
        this.notify(event.message);
        return;

      case "error":
        this.notify(event.message);
        return;

      case "ended":
        this.finish(event.reason);
        return;

      case "ready":
        return;
    }
  }

  private onSpeech(chunk: SpeechChunk): void {
    if (chunk.turn !== this.currentTurn) return;
    const key = `${chunk.turn}:${chunk.seq}`;
    const text = this.pendingSay.get(key) ?? "";
    this.pendingSay.delete(key);
    this.player?.enqueue({ turn: chunk.turn, seq: chunk.seq, text }, chunk.wav);
  }

  private onLinkStatus(status: LinkStatus): void {
    if (status.state === "reconnecting") {
      this.player?.stop();
      this.currentTurn = 0;
      setCall({ link: "reconnecting", zen: "listening", zenCaption: "" });
    } else if (status.state === "open") {
      setCall({ link: "open" });
    } else {
      this.finish("connection_lost");
    }
  }

  // ── media plumbing ──────────────────────────────────────

  private async openCamera(facing: "user" | "environment"): Promise<void> {
    const stream = await this.camera.open(facing);
    if (this.video) this.video.srcObject = stream;
    setCall({ facing: this.camera.facing });
  }

  private syncDetector(): void {
    const { phase, cameraOn, detectionsOn } = useCallStore.getState();
    const wanted = phase === "live" && cameraOn && detectionsOn && this.video !== null;
    if (!wanted) {
      this.detector?.stop();
      this.tracker.reset();
      return;
    }
    if (this.detector) {
      this.detector.start(this.video!, (detections, now) => this.tracker.update(detections, now));
      return;
    }
    this.detectorLoading ??= import("./vision/detector")
      .then(({ Detector }) => Detector.create())
      .then((detector) => {
        if (this.disposed) return detector.close();
        this.detector = detector;
        setCall({ detectorReady: true });
        this.syncDetector();
      })
      .catch((err: unknown) => {
        console.warn("Object detection unavailable", err);
        setCall({ detectionsOn: false });
        this.notify("Object detection isn't available on this device.");
      });
  }

  private onVisibility = (): void => {
    // Release the camera while the app is in the background, and bring it back after.
    if (document.hidden && useCallStore.getState().cameraOn) {
      this.cameraSuspended = true;
      this.camera.close();
    } else if (!document.hidden && this.cameraSuspended) {
      this.cameraSuspended = false;
      void this.openCamera(useCallStore.getState().facing).then(() => this.syncDetector(), () => {});
      void this.acquireWakeLock();
    }
  };

  private onPageHide = (): void => {
    this.transport.hangUp();
    this.dispose();
  };

  private async acquireWakeLock(): Promise<void> {
    try {
      this.wakeLock = (await navigator.wakeLock?.request("screen")) ?? null;
    } catch {
      // not supported or not allowed; the call still works
    }
  }

  // ── endings ─────────────────────────────────────────────

  private finish(reason: EndReason | "connection_lost"): void {
    if (useCallStore.getState().phase !== "live") return;
    this.dispose();
    setCall({ phase: "ended", endReason: reason, endedAt: Date.now(), zen: "listening" });
  }

  private fail(kind: CallErrorKind, detail: string): void {
    this.dispose();
    setCall({ phase: "error", error: { kind, detail } });
  }

  private teardown(): void {
    document.removeEventListener("visibilitychange", this.onVisibility);
    window.removeEventListener("pagehide", this.onPageHide);
    this.timers.forEach(clearTimeout);
    this.timers.clear();
    this.player?.stop();
    void this.vad?.destroy().catch(() => {});
    this.vad = null;
    this.detector?.close();
    this.detector = null;
    this.camera.close();
    this.mic?.getTracks().forEach((t) => t.stop());
    this.mic = null;
    if (this.video) this.video.srcObject = null;
    void this.wakeLock?.release().catch(() => {});
    this.wakeLock = null;
    if (this.ctx && this.ctx.state !== "closed") void this.ctx.close();
    this.ctx = null;
  }

  /** Server voice is gone: say whether the phone's own voice takes over. */
  private voiceFallbackNotice(captionsMessage: string): void {
    const onDevice = this.deviceVoice.ready;
    setCall({ deviceVoice: onDevice });
    this.notify(onDevice ? "Switched to your phone's voice for now." : captionsMessage);
  }

  private notify(message: string): void {
    setCall({ notice: message });
    this.later("notice", NOTICE_MS, () => setCall({ notice: null }));
  }

  private later(key: string, ms: number, fn: () => void): void {
    this.clearTimer(key);
    this.timers.set(key, setTimeout(() => {
      this.timers.delete(key);
      fn();
    }, ms));
  }

  private clearTimer(key: string): void {
    const timer = this.timers.get(key);
    if (timer) clearTimeout(timer);
    this.timers.delete(key);
  }
}
