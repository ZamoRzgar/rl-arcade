import type { ParamDef, ParamValues } from '../types';
import type { Rng } from '../rng';

export const num = (p: ParamValues, id: string, fallback = 0): number => {
  const v = p[id];
  return typeof v === 'number' ? v : fallback;
};
export const bool = (p: ParamValues, id: string): boolean => p[id] === true;
export const str = (p: ParamValues, id: string, fallback = ''): string => {
  const v = p[id];
  return typeof v === 'string' ? v : fallback;
};

export function defaults(params: ParamDef[]): ParamValues {
  const out: ParamValues = {};
  for (const d of params) out[d.id] = d.default;
  return out;
}

// ------------------------------------------------------------- shared parameter definitions

export const gammaParam = (def = 0.9): ParamDef => ({
  id: 'gamma',
  label: 'Discount',
  symbol: 'γ',
  scale: 'linear',
  min: 0,
  max: 0.999,
  step: 0.001,
  default: def,
  explainUp: 'Higher γ: the agent cares about rewards far in the future. Values spread further from the goal.',
  explainDown: 'Lower γ: the agent becomes short-sighted and only values rewards a few steps away.',
  affects: ['viz:horizon'],
});

export const alphaParam = (def = 0.5, id = 'alpha', label = 'Step size', symbol = 'α'): ParamDef => ({
  id,
  label,
  symbol,
  scale: 'log',
  min: 0.001,
  max: 1,
  default: def,
  explainUp: 'Higher: each update moves the estimate further. Faster learning, but noisier and can oscillate.',
  explainDown: 'Lower: small, careful updates. Smoother, but learning is slower.',
  affects: [],
});

export const epsParam = (def = 0.1): ParamDef => ({
  id: 'epsilon',
  label: 'Exploration',
  symbol: 'ε',
  scale: 'linear',
  min: 0,
  max: 1,
  step: 0.01,
  default: def,
  explainUp: 'Higher ε: more random actions. Finds new paths, but the agent stumbles more while learning.',
  explainDown: 'Lower ε: the agent mostly follows its current best guess and may never find better routes.',
  affects: ['viz:explore'],
});

export const epsDecayParam: ParamDef = {
  id: 'epsDecay',
  label: 'ε decay / episode',
  scale: 'linear',
  min: 0.9,
  max: 1,
  step: 0.001,
  default: 1,
  explainUp: 'Closer to 1: exploration stays high for longer (1 = never decays).',
  explainDown: 'Lower: exploration fades quickly, and the agent commits to what it knows early.',
  affects: ['viz:explore'],
};

export const q0Param: ParamDef = {
  id: 'q0',
  label: 'Initial value',
  symbol: 'Q₀',
  scale: 'linear',
  min: -10,
  max: 10,
  step: 0.5,
  default: 0,
  explainUp: 'Optimistic start: every untried action looks great, so the agent is pulled to try everything.',
  explainDown: 'Pessimistic start: untried actions look bad, so the agent sticks with the first things it tries.',
  affects: [],
};

// ------------------------------------------------------------- action selection

/** argmax over Q[off .. off+n) with uniformly random tie-breaking. */
export function argmaxRandom(Q: Float32Array, off: number, n: number, rng: Rng): number {
  let best = -Infinity;
  let count = 0;
  let pick = 0;
  for (let a = 0; a < n; a++) {
    const v = Q[off + a];
    if (v > best + 1e-12) {
      best = v;
      count = 1;
      pick = a;
    } else if (Math.abs(v - best) <= 1e-12) {
      count++;
      if (rng.next() * count < 1) pick = a;
    }
  }
  return pick;
}

export function rowMax(Q: Float32Array, off: number, n: number): number {
  let m = -Infinity;
  for (let a = 0; a < n; a++) if (Q[off + a] > m) m = Q[off + a];
  return m;
}

/** Is `a` one of the greedy actions of the row? */
export function isGreedy(Q: Float32Array, off: number, n: number, a: number): boolean {
  return Q[off + a] >= rowMax(Q, off, n) - 1e-9;
}

/**
 * ε-greedy. Always draws exactly one uniform for the coin flip (plus one more for the action), so
 * the agent's random stream stays aligned across ε values as much as possible.
 */
export function epsGreedy(Q: Float32Array, off: number, n: number, eps: number, rng: Rng): { a: number; explore: boolean } {
  const coin = rng.next();
  if (coin < eps) {
    const a = rng.int(n);
    return { a, explore: !isGreedy(Q, off, n, a) };
  }
  return { a: argmaxRandom(Q, off, n, rng), explore: false };
}

/** Write softmax(θ[off..off+n)) into P[off..off+n). */
export function softmaxRow(theta: Float32Array, P: Float32Array, off: number, n: number) {
  let m = -Infinity;
  for (let a = 0; a < n; a++) m = Math.max(m, theta[off + a]);
  let z = 0;
  for (let a = 0; a < n; a++) {
    const e = Math.exp(theta[off + a] - m);
    P[off + a] = e;
    z += e;
  }
  for (let a = 0; a < n; a++) P[off + a] /= z;
}

export function sampleRow(P: Float32Array, off: number, n: number, rng: Rng): number {
  const u = rng.next();
  let acc = 0;
  for (let a = 0; a < n; a++) {
    acc += P[off + a];
    if (u < acc) return a;
  }
  return n - 1;
}

export const scheduledEps = (p: ParamValues, ep: number) =>
  num(p, 'epsilon', 0.1) * Math.pow(num(p, 'epsDecay', 1), ep);
