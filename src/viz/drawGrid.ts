import type { GridFrame, GridDiff } from './frame';
import { diverging, mix, rgba, type RGB, type Theme } from './theme';
import { DR, DC } from '../core/envs/gridworld';

export interface GridDrawOptions {
  frame: GridFrame;
  ghost: GridFrame | null;
  diff: GridDiff | null;
  overlays: { values: boolean; q: boolean; arrows: boolean; visits: boolean; trail: boolean; ghost: boolean; diff: boolean; numbers: boolean };
  theme: Theme;
  cell: number;
  /** ms clock for slow ambient animation */
  time: number;
  /** 0..1 progress between the previous and current agent cell */
  frac: number;
  /** policy probabilities drawn as glyphs instead of greedy arrows */
  policyGlyphs: boolean;
  traces: Float32Array | null;
  nWindow: number;
  horizon: number;
  hoverCell: number;
  reducedMotion: boolean;
  /** fixed value scale (so baseline and current share colors) */
  valueScale: number;
  playerColor: RGB;
}

const AGENT = ['..####..', '.#....#.', '#.#..#.#', '#......#', '.######.', '..#..#..', '.##..##.'];
const FLAG = ['.#####.', '.##..#.', '.#####.', '.#.....', '.#.....', '###....'];

function pixels(ctx: CanvasRenderingContext2D, art: string[], x: number, y: number, size: number, color: string) {
  const h = art.length;
  const w = art[0].length;
  const p = Math.max(1, Math.floor(size / Math.max(w, h)));
  const ox = x + Math.floor((size - p * w) / 2);
  const oy = y + Math.floor((size - p * h) / 2);
  ctx.fillStyle = color;
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) if (art[r][c] === '#') ctx.fillRect(ox + c * p, oy + r * p, p, p);
}

function arrow(ctx: CanvasRenderingContext2D, cx: number, cy: number, dir: number, len: number, width: number, color: string) {
  const dx = DC[dir];
  const dy = DR[dir];
  const px = -dy;
  const py = dx;
  const tipX = cx + dx * len;
  const tipY = cy + dy * len;
  const head = Math.max(3, width * 2.2);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = width;
  ctx.lineCap = 'square';
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(tipX - dx * head * 0.6, tipY - dy * head * 0.6);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(tipX, tipY);
  ctx.lineTo(tipX - dx * head + px * head * 0.75, tipY - dy * head + py * head * 0.75);
  ctx.lineTo(tipX - dx * head - px * head * 0.75, tipY - dy * head - py * head * 0.75);
  ctx.closePath();
  ctx.fill();
}

export function drawGrid(ctx: CanvasRenderingContext2D, o: GridDrawOptions) {
  const { frame: f, theme: th, cell: C, overlays: ov } = o;
  const L = f.layout;
  const W = L.w * C;
  const H = L.h * C;
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = rgba(th.screen);
  ctx.fillRect(0, 0, W, H);
  const xy = (s: number) => [(s % L.w) * C, Math.floor(s / L.w) * C] as const;
  const center = (s: number) => [(s % L.w) * C + C / 2, Math.floor(s / L.w) * C + C / 2] as const;
  const m = o.valueScale;
  const pulse = o.reducedMotion ? 1 : 0.6 + 0.4 * Math.sin(o.time / 320);

  const diffOn = ov.diff && o.diff;
  const diffScale = o.diff ? Math.max(o.diff.maxAbs, 1e-6) : 1;

  let maxVisits = 1;
  if (ov.visits) for (let i = 0; i < f.visits.length; i++) maxVisits = Math.max(maxVisits, f.visits[i]);

  // ---------------------------------------------------------------- cell fills
  for (let s = 0; s < L.cells.length; s++) {
    const c = L.cells[s];
    const [x, y] = xy(s);
    if (c.t === 'wall') continue;
    const terminal = c.t === 'goal' || c.t === 'lava';
    if (!terminal && c.t !== 'cliff') {
      if (diffOn) {
        ctx.fillStyle = rgba(diverging(th, o.diff!.dV[s], diffScale));
        ctx.fillRect(x, y, C, C);
      } else if (ov.q && f.Q) {
        const cx = x + C / 2;
        const cy = y + C / 2;
        const corners = [
          [x, y, x + C, y],
          [x + C, y, x + C, y + C],
          [x + C, y + C, x, y + C],
          [x, y + C, x, y],
        ];
        for (let a = 0; a < 4; a++) {
          const k = corners[a];
          ctx.fillStyle = rgba(diverging(th, f.Q[s * 4 + a], m));
          ctx.beginPath();
          ctx.moveTo(cx, cy);
          ctx.lineTo(k[0], k[1]);
          ctx.lineTo(k[2], k[3]);
          ctx.closePath();
          ctx.fill();
        }
        ctx.strokeStyle = rgba(th.bg, 0.55);
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + C, y + C);
        ctx.moveTo(x + C, y);
        ctx.lineTo(x, y + C);
        ctx.stroke();
      } else if (ov.values) {
        ctx.fillStyle = rgba(diverging(th, f.V[s], m));
        ctx.fillRect(x, y, C, C);
      }
    }
    if (ov.visits && !terminal) {
      const k = Math.sqrt(f.visits[s] / maxVisits);
      if (k > 0) {
        const sz = Math.max(2, Math.round(C * 0.5 * k));
        ctx.fillStyle = rgba(th.text, 0.18 + 0.3 * k);
        ctx.fillRect(x + C - sz - 2, y + C - sz - 2, sz, sz);
      }
    }
  }

  // eligibility traces (SARSA(λ))
  if (o.traces) {
    for (let s = 0; s < o.traces.length; s++) {
      const e = Math.min(1, o.traces[s]);
      if (e < 0.02) continue;
      const [x, y] = xy(s);
      ctx.fillStyle = rgba(th.yellow, 0.45 * e);
      ctx.fillRect(x + 2, y + 2, C - 4, C - 4);
    }
  }

  // ---------------------------------------------------------------- tiles
  for (let s = 0; s < L.cells.length; s++) {
    const c = L.cells[s];
    const [x, y] = xy(s);
    switch (c.t) {
      case 'wall': {
        ctx.fillStyle = rgba(th.wall);
        ctx.fillRect(x, y, C, C);
        const b = Math.max(3, Math.floor(C / 4));
        ctx.fillStyle = rgba(th.line);
        for (let r = 0; r * b < C; r++) {
          ctx.fillRect(x, y + r * b, C, 1);
          const off = r % 2 ? b : 0;
          for (let k = off; k < C; k += b * 2) ctx.fillRect(x + k, y + r * b, 1, b);
        }
        break;
      }
      case 'goal': {
        ctx.fillStyle = rgba(mix(th.screen, th.good, 0.28));
        ctx.fillRect(x, y, C, C);
        pixels(ctx, FLAG, x + C * 0.15, y + C * 0.15, C * 0.7, rgba(th.good));
        break;
      }
      case 'lava': {
        ctx.fillStyle = rgba(mix(th.screen, th.orange, 0.55));
        ctx.fillRect(x, y, C, C);
        const p = Math.max(2, Math.floor(C / 8));
        ctx.fillStyle = rgba(th.yellow, 0.7);
        const shift = o.reducedMotion ? 0 : Math.floor(o.time / 400) % 2;
        for (let r = 1; r < 8; r += 3) for (let k = (r + shift) % 2; k < 8; k += 2) ctx.fillRect(x + k * p, y + r * p, p, p);
        break;
      }
      case 'cliff': {
        ctx.fillStyle = rgba(th.bg);
        ctx.fillRect(x, y, C, C);
        const p = Math.max(2, Math.floor(C / 8));
        ctx.fillStyle = rgba(th.neg, 0.8);
        for (let k = 0; k < 8; k++) ctx.fillRect(x + k * p, y + (k % 2) * p, p, p);
        ctx.fillStyle = rgba(th.neg, 0.25);
        ctx.fillRect(x, y + C - p, C, p);
        break;
      }
      case 'ice': {
        ctx.strokeStyle = rgba(th.ice, 0.5);
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let k = -C; k < C; k += Math.max(5, C / 4)) {
          ctx.moveTo(x + Math.max(0, k), y + Math.max(0, -k));
          ctx.lineTo(x + Math.min(C, k + C), y + Math.min(C, C - k));
        }
        ctx.stroke();
        break;
      }
      case 'wind': {
        const d = c.dir ?? 0;
        const k = c.strength ?? 1;
        const [cx, cy] = center(s);
        ctx.strokeStyle = rgba(th.text, 0.35);
        ctx.lineWidth = Math.max(1, C / 22);
        for (let i = 0; i < k; i++) {
          const off = (i - (k - 1) / 2) * C * 0.22;
          const bx = cx + DC[d] * off;
          const by = cy + DR[d] * off;
          const px = -DR[d];
          const py = DC[d];
          ctx.beginPath();
          ctx.moveTo(bx - DC[d] * C * 0.1 + px * C * 0.18, by - DR[d] * C * 0.1 + py * C * 0.18);
          ctx.lineTo(bx + DC[d] * C * 0.1, by + DR[d] * C * 0.1);
          ctx.lineTo(bx - DC[d] * C * 0.1 - px * C * 0.18, by - DR[d] * C * 0.1 - py * C * 0.18);
          ctx.stroke();
        }
        break;
      }
      case 'teleport': {
        const [cx, cy] = center(s);
        ctx.strokeStyle = rgba(th.pos, 0.9);
        ctx.lineWidth = Math.max(2, C / 14);
        ctx.beginPath();
        ctx.arc(cx, cy, C * 0.32, 0, Math.PI * 2);
        ctx.stroke();
        if (c.target !== undefined && C >= 22) {
          ctx.fillStyle = rgba(th.text);
          ctx.font = `${Math.round(C * 0.36)}px ${th.fontData}`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${c.label ?? 'T'}`, cx, cy - C * 0.02);
        }
        break;
      }
      case 'start': {
        ctx.strokeStyle = rgba(th.yellow, 0.7);
        ctx.lineWidth = Math.max(1, C / 18);
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x + 3, y + 3, C - 6, C - 6);
        ctx.setLineDash([]);
        break;
      }
      default:
        break;
    }
    if (c.label && c.t === 'empty' && C >= 22) {
      ctx.fillStyle = rgba(th.pos, 0.9);
      ctx.font = `${Math.round(C * 0.3)}px ${th.fontData}`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.fillText(c.label, x + 3, y + 2);
    }
  }

  // teleport links (thin dashed line)
  for (let s = 0; s < L.cells.length; s++) {
    const c = L.cells[s];
    if (c.t !== 'teleport' || c.target === undefined) continue;
    const [ax, ay] = center(s);
    const [bx, by] = center(c.target);
    ctx.strokeStyle = rgba(th.pos, 0.35);
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // grid lines
  ctx.strokeStyle = rgba(th.line, 0.85);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= L.w; c++) {
    ctx.moveTo(c * C + 0.5, 0);
    ctx.lineTo(c * C + 0.5, H);
  }
  for (let r = 0; r <= L.h; r++) {
    ctx.moveTo(0, r * C + 0.5);
    ctx.lineTo(W, r * C + 0.5);
  }
  ctx.stroke();

  // ---------------------------------------------------------------- trail
  if (ov.trail && f.trail.length > 0) {
    const pts = [...f.trail];
    if (f.agent >= 0) pts.push(f.agent);
    const n = pts.length;
    ctx.lineWidth = Math.max(2, C / 12);
    for (let i = 1; i < n; i++) {
      const age = (n - i) / n;
      ctx.strokeStyle = rgba(o.playerColor, 0.15 + 0.6 * (1 - age));
      const [x0, y0] = center(pts[i - 1]);
      const [x1, y1] = center(pts[i]);
      if (Math.abs(x0 - x1) + Math.abs(y0 - y1) > C * 1.6) continue; // teleports / cliff resets
      ctx.beginPath();
      ctx.moveTo(x0, y0);
      ctx.lineTo(x1, y1);
      ctx.stroke();
    }
    // exploratory steps get a spark
    for (let i = 0; i < f.trail.length; i++) {
      if (!f.trailExplore[i]) continue;
      const [x, y] = center(f.trail[i]);
      const r = Math.max(2, C / 10);
      ctx.fillStyle = rgba(th.neg, 0.9);
      ctx.fillRect(x - r / 2, y - r * 1.5, r, r * 3);
      ctx.fillRect(x - r * 1.5, y - r / 2, r * 3, r);
    }
  }
  // n-step window
  if (o.nWindow > 1 && f.trail.length > 0) {
    const win = f.trail.slice(-o.nWindow);
    ctx.strokeStyle = rgba(th.yellow, 0.9);
    ctx.lineWidth = 2;
    ctx.setLineDash([2, 3]);
    for (const s of win) {
      const [x, y] = xy(s);
      ctx.strokeRect(x + 4, y + 4, C - 8, C - 8);
    }
    ctx.setLineDash([]);
  }

  // ---------------------------------------------------------------- arrows
  const drawArrows = (fr: GridFrame, color: RGB, alpha: number, widthK: number, glyphs: boolean) => {
    for (let s = 0; s < fr.layout.cells.length; s++) {
      const c = fr.layout.cells[s];
      if (c.t === 'wall' || c.t === 'goal' || c.t === 'lava' || c.t === 'cliff') continue;
      const [cx, cy] = center(s);
      if (glyphs && fr.P) {
        for (let a = 0; a < 4; a++) {
          const p = fr.P[s * 4 + a];
          if (p < 0.04) continue;
          arrow(ctx, cx, cy, a, C * 0.42 * p, Math.max(1.5, C / 18) * widthK, rgba(color, alpha * (0.35 + 0.65 * p)));
        }
      } else {
        const mask = fr.best[s];
        if (!mask || mask === 15) continue;
        for (let a = 0; a < 4; a++) if (mask & (1 << a)) arrow(ctx, cx, cy, a, C * 0.34, Math.max(1.5, C / 16) * widthK, rgba(color, alpha));
      }
    }
  };
  if (ov.ghost && o.ghost && ov.arrows) drawArrows(o.ghost, th.ghost, 0.55, 0.8, o.policyGlyphs);
  if (ov.arrows) drawArrows(f, th.yellow, 0.95, 1, o.policyGlyphs);

  // policy flips (pulsing outline)
  if (o.diff && (ov.diff || ov.ghost)) {
    ctx.lineWidth = 2;
    ctx.strokeStyle = rgba(th.text, 0.4 + 0.5 * pulse);
    for (let s = 0; s < o.diff.flips.length; s++) {
      if (!o.diff.flips[s]) continue;
      const [x, y] = xy(s);
      ctx.strokeRect(x + 1.5, y + 1.5, C - 3, C - 3);
    }
  }

  // values as numbers
  if (ov.numbers && C >= 40) {
    ctx.font = `${Math.round(C * 0.26)}px ${th.fontData}`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    for (let s = 0; s < L.cells.length; s++) {
      const c = L.cells[s];
      if (c.t === 'wall' || c.t === 'goal' || c.t === 'lava' || c.t === 'cliff') continue;
      const v = diffOn ? o.diff!.dV[s] : f.V[s];
      const [x, y] = xy(s);
      const txt = Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2);
      ctx.fillStyle = rgba(th.bg, 0.7);
      ctx.fillText((diffOn && v > 0 ? '+' : '') + txt, x + C / 2 + 1, y + C - 2 + 1);
      ctx.fillStyle = rgba(th.text, 0.92);
      ctx.fillText((diffOn && v > 0 ? '+' : '') + txt, x + C / 2, y + C - 2);
    }
  }

  // planner scan cursor
  if (f.planner && f.scan >= 0 && f.scan < L.cells.length) {
    const [x, y] = xy(f.scan);
    ctx.fillStyle = rgba(th.yellow, 0.25);
    ctx.fillRect(0, y, W, C);
    ctx.strokeStyle = rgba(th.yellow, 1);
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 1.5, y + 1.5, C - 3, C - 3);
  }

  // planning flashes (Dyna)
  for (const s of f.plan) {
    const [x, y] = xy(s);
    ctx.strokeStyle = rgba(th.pos, 0.85);
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 2]);
    ctx.strokeRect(x + 6, y + 6, C - 12, C - 12);
    ctx.setLineDash([]);
  }

  // last update glow
  if (f.updS >= 0) {
    const [x, y] = xy(f.updS);
    ctx.strokeStyle = rgba(th.yellow, 0.95);
    ctx.lineWidth = 3;
    ctx.strokeRect(x + 1.5, y + 1.5, C - 3, C - 3);
  }

  // effective horizon ring (hovering γ)
  if (o.horizon > 0 && f.agent >= 0) {
    const [cx, cy] = center(f.agent);
    ctx.strokeStyle = rgba(th.yellow, 0.8);
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.beginPath();
    ctx.arc(cx, cy, Math.min(o.horizon, 60) * C, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // editor hover
  if (o.hoverCell >= 0 && o.hoverCell < L.cells.length) {
    const [x, y] = xy(o.hoverCell);
    ctx.strokeStyle = rgba(th.text, 0.9);
    ctx.lineWidth = 2;
    ctx.strokeRect(x + 1, y + 1, C - 2, C - 2);
  }

  // ---------------------------------------------------------------- agents
  const agentAt = (fr: GridFrame, frac: number) => {
    const [x1, y1] = xy(fr.agent);
    if (fr.prevAgent < 0 || fr.prevAgent === fr.agent) return [x1, y1];
    const [x0, y0] = xy(fr.prevAgent);
    if (Math.abs(x0 - x1) + Math.abs(y0 - y1) > C * 1.6) return [x1, y1];
    const k = Math.min(1, frac);
    return [x0 + (x1 - x0) * k, y0 + (y1 - y0) * k];
  };
  if (ov.ghost && o.ghost && o.ghost.agent >= 0) {
    const [gx, gy] = agentAt(o.ghost, o.frac);
    pixels(ctx, AGENT, gx + C * 0.2, gy + C * 0.18, C * 0.6, rgba(th.ghost, 0.6));
  }
  if (f.agent >= 0) {
    const [ax, ay] = agentAt(f, o.frac);
    ctx.fillStyle = rgba(th.bg, 0.5);
    ctx.fillRect(ax + C * 0.22, ay + C * 0.78, C * 0.56, Math.max(2, C * 0.06));
    pixels(ctx, AGENT, ax + C * 0.15, ay + C * 0.1, C * 0.7, rgba(o.playerColor));
  }
}

/** Disagreement map between two policies: cells where they differ, colored by who values the cell more. */
export function drawDisagreement(ctx: CanvasRenderingContext2D, a: GridFrame, b: GridFrame, diff: GridDiff, th: Theme, C: number, colA: RGB, colB: RGB) {
  const L = a.layout;
  ctx.fillStyle = rgba(th.screen);
  ctx.fillRect(0, 0, L.w * C, L.h * C);
  for (let s = 0; s < L.cells.length; s++) {
    const c = L.cells[s];
    const x = (s % L.w) * C;
    const y = Math.floor(s / L.w) * C;
    if (c.t === 'wall') {
      ctx.fillStyle = rgba(th.wall);
      ctx.fillRect(x, y, C, C);
      continue;
    }
    if (c.t === 'goal' || c.t === 'lava' || c.t === 'cliff') {
      ctx.fillStyle = rgba(c.t === 'goal' ? th.good : c.t === 'lava' ? th.orange : th.bg, 0.5);
      ctx.fillRect(x, y, C, C);
      continue;
    }
    if (diff.flips[s]) {
      const who = diff.dV[s] >= 0 ? colA : colB;
      ctx.fillStyle = rgba(who, 0.75);
      ctx.fillRect(x + 1, y + 1, C - 2, C - 2);
      const cx = x + C / 2;
      const cy = y + C / 2;
      for (let d = 0; d < 4; d++) {
        if (a.best[s] & (1 << d)) arrow(ctx, cx, cy, d, C * 0.32, Math.max(1, C / 16), rgba(th.bg, 0.95));
      }
    }
  }
  ctx.strokeStyle = rgba(th.line, 0.8);
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let c = 0; c <= L.w; c++) {
    ctx.moveTo(c * C + 0.5, 0);
    ctx.lineTo(c * C + 0.5, L.h * C);
  }
  for (let r = 0; r <= L.h; r++) {
    ctx.moveTo(0, r * C + 0.5);
    ctx.lineTo(L.w * C, r * C + 0.5);
  }
  ctx.stroke();
  void b;
}
