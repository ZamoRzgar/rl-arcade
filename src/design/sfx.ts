/** Synthesized 8-bit sound effects (WebAudio, no audio files). Muted by default; starts on a user gesture. */

let ctx: AudioContext | null = null;
let lastBlip = 0;

function ac(): AudioContext | null {
  if (typeof window === 'undefined') return null;
  if (!ctx) {
    const C = window.AudioContext ?? (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!C) return null;
    ctx = new C();
  }
  if (ctx.state === 'suspended') void ctx.resume();
  return ctx;
}

function tone(freq: number, dur: number, type: OscillatorType, gain = 0.05, slide = 0, delay = 0) {
  const a = ac();
  if (!a) return;
  const t0 = a.currentTime + delay;
  const o = a.createOscillator();
  const g = a.createGain();
  o.type = type;
  o.frequency.setValueAtTime(freq, t0);
  if (slide) o.frequency.linearRampToValueAtTime(freq + slide, t0 + dur);
  g.gain.setValueAtTime(gain, t0);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(a.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.02);
}

export const sfx = {
  unlock() {
    ac();
  },
  blip() {
    const now = performance.now();
    if (now - lastBlip < 70) return;
    lastBlip = now;
    tone(660, 0.04, 'square', 0.025);
  },
  goal() {
    [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.09, 'square', 0.04, 0, i * 0.07));
  },
  lava() {
    tone(180, 0.25, 'sawtooth', 0.05, -120);
  },
  insert() {
    tone(220, 0.05, 'square', 0.04);
    tone(330, 0.07, 'square', 0.04, 0, 0.05);
  },
  clear() {
    [392, 523, 659, 784, 1047, 1319].forEach((f, i) => tone(f, 0.12, 'triangle', 0.05, 0, i * 0.08));
  },
};
