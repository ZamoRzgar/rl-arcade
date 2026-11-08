import type { BanditRunResult, GridRunResult } from '../core/engine';
import { greedyMask } from '../core/engine';
import { layoutAt, ACTIONS } from '../core/envs/gridworld';
import type { GridLayout } from '../core/types';

/** Everything a renderer needs to draw a grid run at (episode, step). */
export interface GridFrame {
  layout: GridLayout;
  ep: number;
  t: number;
  len: number;
  V: Float32Array;
  Q: Float32Array | null;
  P: Float32Array | null;
  best: Uint8Array;
  agent: number;
  prevAgent: number;
  trail: number[];
  trailExplore: number[];
  updS: number;
  delta: number;
  target: number;
  line: number;
  plan: number[];
  planner: boolean;
  phase: number;
  /** planner scan position (states < scan are already updated this sweep) */
  scan: number;
  visits: Float32Array;
  done: boolean;
}

export const epLength = (r: GridRunResult, ep: number) => (r.planner ? r.S : r.lengths[Math.max(0, Math.min(r.E - 1, ep))] ?? 0);

export function buildGridFrame(r: GridRunResult, baseLayout: GridLayout, epIn: number, tIn: number): GridFrame {
  const S = r.S;
  const A = r.A;
  const ep = Math.max(0, Math.min(r.E - 1, epIn));
  const len = epLength(r, ep);
  const t = Math.max(0, Math.min(len, tIn));
  const layout = layoutAt(baseLayout, ep);
  const atEnd = t >= len;
  const snap = atEnd ? ep + 1 : ep;
  const V = r.snapV.slice(snap * S, (snap + 1) * S);
  const Q = r.hasQ ? r.snapQ.slice(snap * S * A, (snap + 1) * S * A) : null;
  const P = r.hasP ? r.snapP.slice(snap * S * A, (snap + 1) * S * A) : null;
  const visits = r.visits.slice(snap * S, (snap + 1) * S);
  const frame: GridFrame = {
    layout,
    ep,
    t,
    len,
    V,
    Q,
    P,
    best: new Uint8Array(S),
    agent: -1,
    prevAgent: -1,
    trail: [],
    trailExplore: [],
    updS: -1,
    delta: NaN,
    target: NaN,
    line: -1,
    plan: [],
    planner: r.planner,
    phase: r.planner ? r.phases[ep] ?? 0 : 0,
    scan: -1,
    visits,
    done: atEnd && ep >= r.E - 1,
  };

  if (r.planner) {
    // states before the scan position already show this sweep's result
    if (!atEnd) {
      const next = r.snapV.subarray((ep + 1) * S, (ep + 2) * S);
      const nextQ = r.hasQ ? r.snapQ.subarray((ep + 1) * S * A, (ep + 2) * S * A) : null;
      const nextP = r.hasP ? r.snapP.subarray((ep + 1) * S * A, (ep + 2) * S * A) : null;
      for (let s = 0; s < t; s++) {
        V[s] = next[s];
        if (Q && nextQ) for (let a = 0; a < A; a++) Q[s * A + a] = nextQ[s * A + a];
        if (P && nextP) for (let a = 0; a < A; a++) P[s * A + a] = nextP[s * A + a];
      }
      frame.scan = t;
    }
    frame.best = greedyMask(Q, P, S);
    return frame;
  }

  const start = r.epStart[ep];
  // apply the rows updated by steps < t of this episode
  if (!atEnd) {
    for (let j = start; j < start + t; j++) {
      const us = r.stUpdS[j];
      if (us < 0) continue;
      V[us] = r.stRowV[j];
      if (Q) for (let a = 0; a < A; a++) Q[us * A + a] = r.stRowQ[j * A + a];
      if (P) for (let a = 0; a < A; a++) P[us * A + a] = r.stRowP[j * A + a];
    }
    for (let j = start; j < start + t; j++) visits[r.stS[j]] += 1;
  }
  frame.best = greedyMask(Q, P, S);
  for (let j = start; j < start + t; j++) {
    frame.trail.push(r.stS[j]);
    frame.trailExplore.push(r.stExplore[j]);
  }
  if (len > 0) {
    frame.agent = t < len ? r.stS[start + t] : r.stNext[start + len - 1];
    frame.prevAgent = t > 0 ? r.stS[start + t - 1] : frame.agent;
    if (t === len && len > 0) frame.trail.push(r.stS[start + len - 1]);
    if (t > 0) {
      const j = start + t - 1;
      frame.updS = r.stUpdS[j];
      frame.delta = r.stDelta[j];
      frame.target = r.stTarget[j];
      frame.line = r.stLine[j];
      for (let i = 0; i < 4; i++) if (r.stPlan[j * 4 + i] >= 0) frame.plan.push(r.stPlan[j * 4 + i]);
    }
  }
  return frame;
}

/** Eligibility trace of the current episode's trail, reconstructed exactly: e ← γλe, then mark. */
export function tracesFromTrail(trail: number[], S: number, gammaLambda: number, replacing: boolean): Float32Array {
  // e(s) = Σ over visits k of (γλ)^(T−1−k)  (accumulating), or (γλ)^(T−1−k_last)  (replacing)
  const e = new Float32Array(S);
  const T = trail.length;
  let w = 1;
  for (let k = T - 1; k >= 0; k--) {
    const s = trail[k];
    if (replacing) {
      if (e[s] === 0) e[s] = w;
    } else e[s] += w;
    w *= gammaLambda;
  }
  return e;
}

export interface BanditFrame {
  t: number;
  k: number;
  Q: Float32Array;
  N: Float32Array;
  X: Float32Array;
  lastArm: number;
  lastReward: number;
  lastExplore: boolean;
  line: number;
}

export function buildBanditFrame(r: BanditRunResult, tIn: number): BanditFrame {
  const t = Math.max(0, Math.min(r.T, tIn));
  const k = r.k;
  return {
    t,
    k,
    Q: r.snapQ.slice(t * k, (t + 1) * k),
    N: r.snapN.slice(t * k, (t + 1) * k),
    X: r.snapX.slice(t * k * 2, (t + 1) * k * 2),
    lastArm: t > 0 ? r.arms[t - 1] : -1,
    lastReward: t > 0 ? r.rewards[t - 1] : NaN,
    lastExplore: t > 0 ? r.explore[t - 1] === 1 : false,
    line: t > 0 ? r.lines[t - 1] : -1,
  };
}

/** Comparison of two grid frames at the same (episode, step). */
export interface GridDiff {
  dV: Float32Array;
  maxAbs: number;
  flips: Uint8Array;
  flipCount: number;
  liveCount: number;
}

export function diffGridFrames(a: GridFrame, b: GridFrame): GridDiff {
  const S = a.V.length;
  const dV = new Float32Array(S);
  const flips = new Uint8Array(S);
  let maxAbs = 0;
  let flipCount = 0;
  let liveCount = 0;
  for (let s = 0; s < S; s++) {
    const c = a.layout.cells[s];
    if (c.t === 'wall' || c.t === 'goal' || c.t === 'lava') continue;
    liveCount++;
    dV[s] = a.V[s] - b.V[s];
    maxAbs = Math.max(maxAbs, Math.abs(dV[s]));
    if (a.best[s] && b.best[s] && (a.best[s] & b.best[s]) === 0) {
      flips[s] = 1;
      flipCount++;
    }
  }
  return { dV, maxAbs, flips, flipCount, liveCount };
}

export { ACTIONS };
