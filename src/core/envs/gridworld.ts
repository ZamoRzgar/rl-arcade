import type { Cell, GridLayout, Outcome, Transition } from '../types';
import type { Rng } from '../rng';

/** Actions: 0 up, 1 right, 2 down, 3 left. */
export const ACTIONS = 4;
export const DR = [-1, 0, 1, 0];
export const DC = [0, 1, 0, -1];
export const ACTION_LABELS = ['↑', '→', '↓', '←'];

export const isTerminalCell = (c: Cell) => c.t === 'goal' || c.t === 'lava';

/** The layout as it looks during episode `ep`, after applying scheduled events. */
export function layoutAt(layout: GridLayout, ep: number): GridLayout {
  if (!layout.events || layout.events.length === 0) return layout;
  const active = layout.events.filter((e) => e.atEpisode <= ep);
  if (active.length === 0) return layout;
  const cells = layout.cells.slice();
  for (const ev of active) for (const [i, c] of ev.set) cells[i] = c;
  return { ...layout, cells };
}

/**
 * Editable grid world (Sutton & Barto ch. 3–6 style).
 *
 * Dynamics, in order:
 * 1. Slip: with probability p (global `slip`, or `iceSlip` when leaving an ice cell) the chosen
 *    action is replaced by a uniformly random one. Exactly one uniform is drawn per step whether or
 *    not slipping is possible, so two runs see the same random numbers at the same (episode, step).
 * 2. Teleport cells move the agent to their target with their reward, whatever the action (Ex. 3.5).
 * 3. Otherwise move one cell; walls and edges block the move and give `bumpReward`.
 * 4. Wind of the cell being left pushes the agent `strength` cells (Windy Gridworld, Ex. 6.5).
 * 5. Entering a goal or lava ends the episode and adds its reward. Entering a cliff gives
 *    `cliffReward` and sends the agent back to the start without ending the episode (Ex. 6.6).
 */
export class GridEnv {
  layout!: GridLayout;
  w = 0;
  h = 0;
  S = 0;
  start = 0;
  valid!: Uint8Array;
  terminal!: Uint8Array;
  /** valid, non-terminal states, in raster order */
  live: number[] = [];
  private base: GridLayout;
  private currentEp = -1;

  constructor(layout: GridLayout) {
    this.base = layout;
    this.setLayout(layout);
  }

  private setLayout(layout: GridLayout) {
    this.layout = layout;
    this.w = layout.w;
    this.h = layout.h;
    this.S = layout.w * layout.h;
    this.valid = new Uint8Array(this.S);
    this.terminal = new Uint8Array(this.S);
    this.live = [];
    let start = -1;
    for (let s = 0; s < this.S; s++) {
      const c = layout.cells[s];
      this.valid[s] = c.t === 'wall' ? 0 : 1;
      this.terminal[s] = isTerminalCell(c) ? 1 : 0;
      if (c.t === 'start' && start < 0) start = s;
      if (this.valid[s] && !this.terminal[s]) this.live.push(s);
    }
    this.start = start >= 0 ? start : this.live[0] ?? 0;
  }

  /** Apply scheduled layout events for episode `ep` (no-op when nothing changes). */
  prepareEpisode(ep: number) {
    if (!this.base.events || this.base.events.length === 0) return;
    const before = this.base.events.filter((e) => e.atEpisode <= this.currentEp).length;
    const after = this.base.events.filter((e) => e.atEpisode <= ep).length;
    this.currentEp = ep;
    if (before !== after) this.setLayout(layoutAt(this.base, ep));
  }

  reset(rng: Rng, randomStart: boolean): number {
    // Always draw once so the stream position does not depend on the start mode.
    const u = rng.next();
    if (randomStart || this.layout.randomStart) {
      return this.live[Math.min(this.live.length - 1, Math.floor(u * this.live.length))];
    }
    return this.start;
  }

  slipProb(s: number): number {
    const c = this.layout.cells[s];
    return c.t === 'ice' ? Math.max(this.layout.slip, this.layout.iceSlip) : this.layout.slip;
  }

  /** Deterministic result of actually performing `a` in `s`. */
  move(s: number, a: number): Transition {
    const L = this.layout;
    const cell = L.cells[s];
    if (cell.t === 'teleport' && cell.target !== undefined) {
      return { next: cell.target, reward: cell.reward ?? 0, done: false };
    }
    const r = Math.floor(s / this.w);
    const c = s % this.w;
    let nr = r + DR[a];
    let nc = c + DC[a];
    let reward = L.stepReward;
    if (!this.open(nr, nc)) {
      nr = r;
      nc = c;
      reward = L.bumpReward;
    }
    if (cell.t === 'wind') {
      const d = cell.dir ?? 0;
      const k = cell.strength ?? 1;
      for (let i = 0; i < k; i++) {
        const wr = nr + DR[d];
        const wc = nc + DC[d];
        if (!this.open(wr, wc)) break;
        nr = wr;
        nc = wc;
      }
    }
    const next = nr * this.w + nc;
    const nt = L.cells[next];
    if (nt.t === 'goal') return { next, reward: reward + (nt.reward ?? L.goalReward), done: true };
    if (nt.t === 'lava') return { next, reward: reward + L.lavaReward, done: true };
    if (nt.t === 'cliff') return { next: this.start, reward: L.cliffReward, done: false };
    return { next, reward, done: false };
  }

  private open(r: number, c: number): boolean {
    if (r < 0 || c < 0 || r >= this.h || c >= this.w) return false;
    return this.layout.cells[r * this.w + c].t !== 'wall';
  }

  step(s: number, a: number, rng: Rng): Transition {
    const u = rng.next();
    const p = this.slipProb(s);
    const eff = u < p ? Math.min(ACTIONS - 1, Math.floor((u / p) * ACTIONS)) : a;
    return this.move(s, eff);
  }

  /** Full model p(s', r | s, a) for dynamic programming. Terminal states have no outcomes. */
  model(s: number, a: number): Outcome[] {
    if (this.terminal[s] || !this.valid[s]) return [];
    const p = this.slipProb(s);
    const out: Outcome[] = [];
    for (let e = 0; e < ACTIONS; e++) {
      const prob = (e === a ? 1 - p : 0) + p / ACTIONS;
      if (prob <= 0) continue;
      const tr = this.move(s, e);
      const same = out.find((o) => o.next === tr.next && o.reward === tr.reward && o.done === tr.done);
      if (same) same.p += prob;
      else out.push({ p: prob, ...tr });
    }
    return out;
  }
}

export function emptyLayout(w: number, h: number): GridLayout {
  return {
    w,
    h,
    cells: Array.from({ length: w * h }, () => ({ t: 'empty' as const })),
    stepReward: -1,
    bumpReward: -1,
    goalReward: 0,
    lavaReward: -50,
    cliffReward: -100,
    slip: 0,
    iceSlip: 0.5,
    randomStart: false,
  };
}

/** Resize keeping the top-left content; teleport targets that fall outside are dropped. */
export function resizeLayout(L: GridLayout, w: number, h: number): GridLayout {
  const cells: Cell[] = [];
  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      if (r < L.h && c < L.w) {
        const old = L.cells[r * L.w + c];
        if (old.t === 'teleport' && old.target !== undefined) {
          const tr = Math.floor(old.target / L.w);
          const tc = old.target % L.w;
          if (tr < h && tc < w) cells.push({ ...old, target: tr * w + tc });
          else cells.push({ t: 'empty' });
        } else cells.push({ ...old });
      } else cells.push({ t: 'empty' });
    }
  }
  return { ...L, w, h, cells, events: undefined };
}
