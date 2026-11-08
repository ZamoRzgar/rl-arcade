/** Reads the design tokens from CSS so canvas drawing uses exactly the same palette as the page. */

export type RGB = [number, number, number];

export interface Theme {
  bg: RGB;
  screen: RGB;
  panel: RGB;
  line: RGB;
  text: RGB;
  dim: RGB;
  pos: RGB;
  neg: RGB;
  yellow: RGB;
  orange: RGB;
  ghost: RGB;
  good: RGB;
  bad: RGB;
  wall: RGB;
  ice: RGB;
  fontData: string;
  fontDisplay: string;
}

const parse = (v: string): RGB => {
  const s = v.trim();
  if (s.startsWith('#')) {
    const h = s.length === 4 ? s.slice(1).split('').map((c) => c + c).join('') : s.slice(1, 7);
    return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
  }
  const m = /rgba?\(([^)]+)\)/.exec(s);
  if (m) {
    const [r, g, b] = m[1].split(',').map((x) => parseFloat(x));
    return [r, g, b];
  }
  return [255, 0, 255];
};

let cached: Theme | null = null;
let cachedKey = '';

export function readTheme(): Theme {
  const root = document.documentElement;
  const key = root.dataset.palette ?? 'neon';
  if (cached && cachedKey === key) return cached;
  const cs = getComputedStyle(root);
  const g = (n: string) => parse(cs.getPropertyValue(n) || '#ff00ff');
  cached = {
    bg: g('--bg'),
    screen: g('--screen'),
    panel: g('--panel'),
    line: g('--line'),
    text: g('--text'),
    dim: g('--text-dim'),
    pos: g('--val-pos'),
    neg: g('--val-neg'),
    yellow: g('--neon-yellow'),
    orange: g('--neon-orange'),
    ghost: g('--ghost'),
    good: g('--good'),
    bad: g('--bad'),
    wall: g('--wall'),
    ice: g('--ice'),
    fontData: cs.getPropertyValue('--font-data').trim() || 'monospace',
    fontDisplay: cs.getPropertyValue('--font-display').trim() || 'monospace',
  };
  cachedKey = key;
  return cached;
}

export const invalidateTheme = () => {
  cached = null;
};

export const rgba = (c: RGB, a = 1) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
export const mix = (a: RGB, b: RGB, t: number): RGB => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t];

/** Diverging color for value v on a symmetric scale ±m: neg ← screen (0) → pos. */
export function diverging(th: Theme, v: number, m: number): RGB {
  if (!Number.isFinite(v) || m <= 0) return th.screen;
  const t = Math.max(-1, Math.min(1, v / m));
  const k = Math.pow(Math.abs(t), 0.75) * 0.9;
  return mix(th.screen, t >= 0 ? th.pos : th.neg, k);
}
