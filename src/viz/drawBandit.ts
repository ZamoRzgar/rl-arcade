import type { BanditRunResult } from '../core/engine';
import type { BanditFrame } from './frame';
import { mix, rgba, type RGB, type Theme } from './theme';

export interface BanditDrawOptions {
  res: BanditRunResult;
  frame: BanditFrame;
  ghost: BanditFrame | null;
  theme: Theme;
  width: number;
  height: number;
  showTruth: boolean;
  time: number;
  playerColor: RGB;
  hoverParam: string | null;
}

function lgamma(x: number): number {
  // Lanczos approximation
  const g = 7;
  const c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61582229025619, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7];
  if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  x -= 1;
  let a = c[0];
  const t = x + g + 0.5;
  for (let i = 1; i < g + 2; i++) a += c[i] / (x + i);
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}
const betaPdf = (x: number, a: number, b: number) =>
  x <= 0 || x >= 1 ? 0 : Math.exp((a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x) + lgamma(a + b) - lgamma(a) - lgamma(b));
const normalPdf = (x: number, m: number, s: number) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));

/** Arcade "cabinet" of k arms: estimate bars, true means, UCB whiskers, posterior violins, softmax π. */
export function drawBandit(ctx: CanvasRenderingContext2D, o: BanditDrawOptions) {
  const { res: r, frame: f, theme: th, width: W, height: H } = o;
  const k = r.k;
  const bern = r.dist === 'bernoulli';
  const lo = bern ? 0 : -3;
  const hi = bern ? 1 : 3;
  const top = 60;
  const piStrip = r.xKind === 'softmax' ? 54 : 0;
  const bottom = H - 40 - piStrip;
  const y = (v: number) => bottom - ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * (bottom - top);
  const colW = W / k;

  ctx.fillStyle = rgba(th.screen);
  ctx.fillRect(0, 0, W, H);

  // axis grid
  ctx.font = `14px ${th.fontData}`;
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  const ticks = bern ? [0, 0.25, 0.5, 0.75, 1] : [-3, -2, -1, 0, 1, 2, 3];
  for (const v of ticks) {
    ctx.strokeStyle = rgba(th.line, v === 0 ? 1 : 0.6);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(y(v)) + 0.5);
    ctx.lineTo(W, Math.round(y(v)) + 0.5);
    ctx.stroke();
    ctx.fillStyle = rgba(th.dim);
    ctx.fillText(String(v), 4, y(v) - 7);
  }

  for (let a = 0; a < k; a++) {
    const x0 = a * colW;
    const cx = x0 + colW / 2;
    const bw = Math.max(6, colW * 0.34);
    const isLast = f.lastArm === a;
    if (isLast) {
      ctx.fillStyle = rgba(o.playerColor, 0.12);
      ctx.fillRect(x0 + 2, top - 36, colW - 4, bottom - top + 40);
    }

    // posterior violin (Thompson)
    if (r.xKind === 'beta' || r.xKind === 'normal') {
      const p1 = f.X[a * 2];
      const p2 = f.X[a * 2 + 1];
      const pts: [number, number][] = [];
      let maxD = 0;
      for (let i = 0; i <= 60; i++) {
        const v = lo + ((hi - lo) * i) / 60;
        const d = r.xKind === 'beta' ? betaPdf(v, p1, p2) : normalPdf(v, p1, p2);
        pts.push([v, d]);
        maxD = Math.max(maxD, d);
      }
      const half = colW * 0.42;
      ctx.fillStyle = rgba(th.pos, 0.18);
      ctx.strokeStyle = rgba(th.pos, 0.8);
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      pts.forEach(([v, d], i) => {
        const xx = cx + (maxD > 0 ? (d / maxD) * half : 0);
        if (i === 0) ctx.moveTo(xx, y(v));
        else ctx.lineTo(xx, y(v));
      });
      for (let i = pts.length - 1; i >= 0; i--) {
        const [v, d] = pts[i];
        ctx.lineTo(cx - (maxD > 0 ? (d / maxD) * half : 0), y(v));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    } else {
      // estimate bar from 0 (or bottom for bernoulli)
      const base = y(bern ? 0 : 0);
      const yv = y(f.Q[a]);
      ctx.fillStyle = rgba(mix(th.screen, f.Q[a] >= 0 ? th.pos : th.neg, 0.8));
      ctx.fillRect(cx - bw / 2, Math.min(base, yv), bw, Math.max(2, Math.abs(base - yv)));
    }

    // estimate marker
    ctx.fillStyle = rgba(th.text);
    ctx.fillRect(cx - bw / 2 - 3, y(f.Q[a]) - 1.5, bw + 6, 3);

    // ghost (baseline) estimate
    if (o.ghost) {
      ctx.strokeStyle = rgba(th.ghost, 0.9);
      ctx.lineWidth = 2;
      ctx.setLineDash([3, 2]);
      ctx.strokeRect(cx - bw / 2 - 5, y(o.ghost.Q[a]) - 3, bw + 10, 6);
      ctx.setLineDash([]);
    }

    // UCB whisker
    if (r.xKind === 'ucb') {
      const bonus = f.X[a * 2];
      const yTop = Number.isFinite(bonus) ? y(f.Q[a] + bonus) : top - 6;
      ctx.strokeStyle = rgba(th.yellow, o.hoverParam === 'c' ? 1 : 0.85);
      ctx.lineWidth = o.hoverParam === 'c' ? 3 : 2;
      ctx.beginPath();
      ctx.moveTo(cx, y(f.Q[a]));
      ctx.lineTo(cx, yTop);
      ctx.moveTo(cx - 6, yTop);
      ctx.lineTo(cx + 6, yTop);
      ctx.stroke();
      if (!Number.isFinite(bonus)) {
        ctx.fillStyle = rgba(th.yellow);
        ctx.textAlign = 'center';
        ctx.fillText('∞', cx, yTop - 8);
      }
    }

    // true mean
    if (o.showTruth) {
      ctx.strokeStyle = rgba(r.optimal === a ? th.good : th.orange, 1);
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 3]);
      ctx.beginPath();
      ctx.moveTo(x0 + 6, y(r.means[a]));
      ctx.lineTo(x0 + colW - 6, y(r.means[a]));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // lever on top
    ctx.fillStyle = rgba(isLast ? o.playerColor : th.line);
    const pull = isLast ? 6 : 0;
    ctx.fillRect(cx - 2, top - 26 + pull, 4, 16);
    ctx.fillRect(cx - 6, top - 32 + pull, 12, 8);

    // softmax strip
    if (r.xKind === 'softmax') {
      const pi = f.X[a * 2];
      const sy = H - 40 - piStrip + 10;
      ctx.fillStyle = rgba(th.line);
      ctx.fillRect(cx - bw / 2, sy, bw, piStrip - 16);
      const hh = (piStrip - 16) * pi;
      ctx.fillStyle = rgba(th.yellow);
      ctx.fillRect(cx - bw / 2, sy + (piStrip - 16) - hh, bw, hh);
    }

    // labels
    ctx.textAlign = 'center';
    ctx.fillStyle = rgba(isLast ? o.playerColor : th.text);
    ctx.font = `16px ${th.fontData}`;
    ctx.fillText(`#${a + 1}`, cx, H - 26);
    ctx.fillStyle = rgba(th.dim);
    ctx.font = `14px ${th.fontData}`;
    ctx.fillText(`n=${f.N[a]}`, cx, H - 10);
  }
  if (r.xKind === 'softmax') {
    ctx.fillStyle = rgba(th.dim);
    ctx.textAlign = 'left';
    ctx.font = `14px ${th.fontData}`;
    ctx.fillText('π(a)', 4, H - 40 - piStrip + 4);
  }

  // last reward popup
  if (f.lastArm >= 0) {
    const cx = f.lastArm * colW + colW / 2;
    ctx.font = `18px ${th.fontData}`;
    ctx.textAlign = 'center';
    ctx.fillStyle = rgba(f.lastReward >= (bern ? 0.5 : 0) ? th.good : th.bad);
    const txt = bern ? (f.lastReward > 0 ? 'WIN' : 'MISS') : (f.lastReward >= 0 ? '+' : '') + f.lastReward.toFixed(2);
    ctx.fillText(txt, cx, 12);
  }
}
