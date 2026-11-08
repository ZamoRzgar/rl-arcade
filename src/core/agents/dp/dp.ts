import type { AlgoMeta, GridPlanner, ParamDef, ParamValues, PlannerFrame, PseudoLine, Tables } from '../../types';
import { ACTIONS, GridEnv } from '../../envs/gridworld';
import { gammaParam, num, str } from '../common';

const L = (id: string, text: string, indent: number, params?: string[]): PseudoLine => ({ id, text, indent, params });
const TIE = 1e-7;

/** q(s,a) = Σ p(s',r|s,a) [r + γ V(s')]  (terminal next states have V = 0) */
function qValue(env: GridEnv, V: ArrayLike<number>, s: number, a: number, gamma: number): number {
  let q = 0;
  for (const o of env.model(s, a)) q += o.p * (o.reward + (o.done ? 0 : gamma * V[o.next]));
  return q;
}

function fillQ(env: GridEnv, V: ArrayLike<number>, Q: Float32Array | Float64Array, gamma: number) {
  for (const s of env.live) for (let a = 0; a < ACTIONS; a++) Q[s * ACTIONS + a] = qValue(env, V, s, a, gamma);
}

/** Greedy policy as probabilities, ties sharing the mass equally. */
function greedyFromQ(env: GridEnv, Q: Float32Array | Float64Array, P: Float32Array) {
  P.fill(0);
  for (const s of env.live) {
    const off = s * ACTIONS;
    let m = -Infinity;
    for (let a = 0; a < ACTIONS; a++) m = Math.max(m, Q[off + a]);
    let n = 0;
    for (let a = 0; a < ACTIONS; a++) if (Q[off + a] >= m - TIE) n++;
    for (let a = 0; a < ACTIONS; a++) P[off + a] = Q[off + a] >= m - TIE ? 1 / n : 0;
  }
}

/**
 * Iterative policy evaluation of a stochastic policy P (Sutton & Barto §4.1), in place.
 * Exported for tests (equiprobable random policy on the Example 3.5 grid).
 */
export function evaluatePolicy(env: GridEnv, P: Float32Array, gamma: number, theta: number, maxSweeps = 10_000): Float64Array {
  const V = new Float64Array(env.S);
  for (let k = 0; k < maxSweeps; k++) {
    let delta = 0;
    for (const s of env.live) {
      let v = 0;
      for (let a = 0; a < ACTIONS; a++) {
        const pa = P[s * ACTIONS + a];
        if (pa > 0) v += pa * qValue(env, V, s, a, gamma);
      }
      delta = Math.max(delta, Math.abs(v - V[s]));
      V[s] = v;
    }
    if (delta < theta) break;
  }
  return V;
}

const thetaParam: ParamDef = {
  id: 'theta',
  label: 'Stop threshold',
  symbol: 'θ',
  scale: 'log',
  min: 1e-6,
  max: 0.1,
  default: 1e-4,
  explainUp: 'Larger θ: stop sweeping earlier, which is faster but less exact.',
  explainDown: 'Smaller θ: keep sweeping until the values barely move.',
  affects: [],
};

// ==========================================================================================
// Value iteration (Sutton & Barto §4.4)
// ==========================================================================================

export class ValueIteration implements GridPlanner {
  V: Float32Array;
  Q: Float32Array;
  P: Float32Array;
  /** values are computed in float64; V is the float32 copy shown to the UI */
  private v: Float64Array;
  constructor(private readonly env: GridEnv, private readonly p: ParamValues) {
    this.v = new Float64Array(env.S);
    this.q = new Float64Array(env.S * ACTIONS);
    this.V = new Float32Array(env.S);
    this.Q = new Float32Array(env.S * ACTIONS);
    this.P = new Float32Array(env.S * ACTIONS);
    this.syncQ(num(p, 'gamma', 0.9));
  }
  private q: Float64Array;
  private syncQ(gamma: number) {
    fillQ(this.env, this.v, this.q, gamma);
    this.Q.set(this.q);
    greedyFromQ(this.env, this.q, this.P);
  }
  frame(): PlannerFrame {
    const gamma = num(this.p, 'gamma', 0.9);
    const theta = num(this.p, 'theta', 1e-4);
    const inPlace = str(this.p, 'sweep', 'in-place') === 'in-place';
    const env = this.env;
    const src = inPlace ? this.v : this.v.slice();
    let delta = 0;
    for (const s of env.live) {
      let best = -Infinity;
      for (let a = 0; a < ACTIONS; a++) best = Math.max(best, qValue(env, src, s, a, gamma));
      delta = Math.max(delta, Math.abs(best - this.v[s]));
      this.v[s] = best;
    }
    this.V.set(this.v);
    this.syncQ(gamma);
    return { phase: 0, delta, done: delta < theta };
  }
  tables(): Tables {
    return { V: this.V, Q: this.Q, P: this.P };
  }
}

export const valueIterationMeta: AlgoMeta = {
  id: 'value-iteration',
  name: 'Value iteration',
  short: 'VALUE ITER',
  family: 'dp',
  envKind: 'grid',
  needsModel: true,
  params: [
    gammaParam(0.9),
    thetaParam,
    {
      id: 'sweep',
      label: 'Sweep mode',
      scale: 'enum',
      options: ['in-place', 'synchronous'],
      default: 'in-place',
      explainUp: 'synchronous: every state reads the previous sweep’s values (a clean two-array update).',
      explainDown: 'in-place: states read values already updated in this sweep, which usually converges in fewer sweeps.',
      affects: [],
    },
  ],
  equation: [
    { id: 'lhsV', tex: 'V(s)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'max', tex: '\\max_{a}' },
    { id: 'sum-p', tex: "\\textstyle\\sum_{s',r} p(s',r\\mid s,a)" },
    { id: 'lb', tex: '\\Big[' },
    { id: 'R', tex: 'r' },
    { id: 'plus2', tex: '+' },
    { id: 'gamma', tex: '\\gamma', param: 'gamma' },
    { id: 'boot-v', tex: "V(s')" },
    { id: 'rb', tex: '\\Big]' },
  ],
  pseudocode: [
    L('init', 'Initialize V(s) = 0 for all s', 0),
    L('loop', 'Loop (one sweep per frame):', 0),
    L('delta0', 'Δ ← 0', 1),
    L('each', 'For each state s:', 1),
    L('bellman', 'V(s) ← maxₐ Σ p(s′,r|s,a)[r + γV(s′)]', 2, ['gamma']),
    L('delta', 'Δ ← max(Δ, |v − V(s)|)', 2),
    L('until', 'until Δ < θ', 0, ['theta']),
    L('output', 'π(s) = argmaxₐ Σ p(s′,r|s,a)[r + γV(s′)]', 0),
  ],
  backup: {
    nodes: [
      { id: 's0', kind: 'state', x: 50, y: 10 },
      { id: 'aL', kind: 'action', x: 22, y: 42 },
      { id: 'a0', kind: 'action', x: 50, y: 42 },
      { id: 'aR', kind: 'action', x: 78, y: 42 },
      { id: 'sL1', kind: 'state', x: 12, y: 80 },
      { id: 'sL2', kind: 'state', x: 30, y: 80 },
      { id: 's1', kind: 'state', x: 42, y: 80 },
      { id: 's1b', kind: 'state', x: 58, y: 80 },
      { id: 'sR1', kind: 'state', x: 70, y: 80 },
      { id: 'sR2', kind: 'state', x: 88, y: 80 },
    ],
    edges: [['s0', 'aL'], ['s0', 'a0'], ['s0', 'aR'], ['aL', 'sL1'], ['aL', 'sL2'], ['a0', 's1'], ['a0', 's1b'], ['aR', 'sR1'], ['aR', 'sR2']],
    arcs: [{ id: 'arc-top', over: ['aL', 'a0', 'aR'], label: 'max' }],
    caption: 'Full-width backup: every action and every possible next state, using the known model.',
  },
  summary:
    'Uses the known model of the world. Every sweep, each cell takes the best one-step lookahead over all actions and all possible outcomes. Values ripple outward from the rewards until nothing changes.',
  watchFor: ['Watch the scan line sweep the grid. Each pass spreads value one more step from the goal.', 'Change γ and see how far the value ripple reaches.'],
  reference: 'Sutton & Barto §4.4',
  traits: {
    target: 'the best expected one-step lookahead, computed exactly from the known model',
    onPolicy: false,
    model: 'given',
    timing: 'in sweeps over every state, with no episodes or experience',
    learns: 'state values V(s)',
  },
};

// ==========================================================================================
// Policy iteration (Sutton & Barto §4.3). Each frame = one evaluation sweep or one improvement.
// ==========================================================================================

export class PolicyIteration implements GridPlanner {
  V: Float32Array;
  Q: Float32Array;
  P: Float32Array;
  private v: Float64Array;
  private phase: 'eval' | 'improve' = 'eval';
  constructor(private readonly env: GridEnv, private readonly p: ParamValues) {
    this.v = new Float64Array(env.S);
    this.q = new Float64Array(env.S * ACTIONS);
    this.V = new Float32Array(env.S);
    this.Q = new Float32Array(env.S * ACTIONS);
    this.P = new Float32Array(env.S * ACTIONS);
    for (const s of env.live) for (let a = 0; a < ACTIONS; a++) this.P[s * ACTIONS + a] = 1 / ACTIONS;
    fillQ(env, this.v, this.q, num(p, 'gamma', 0.9));
    this.Q.set(this.q);
  }
  private q: Float64Array;
  frame(): PlannerFrame {
    const gamma = num(this.p, 'gamma', 0.9);
    const theta = num(this.p, 'theta', 1e-4);
    const env = this.env;
    if (this.phase === 'eval') {
      let delta = 0;
      for (const s of env.live) {
        let v = 0;
        for (let a = 0; a < ACTIONS; a++) {
          const pa = this.P[s * ACTIONS + a];
          if (pa > 0) v += pa * qValue(env, this.v, s, a, gamma);
        }
        delta = Math.max(delta, Math.abs(v - this.v[s]));
        this.v[s] = v;
      }
      this.V.set(this.v);
      fillQ(env, this.v, this.q, gamma);
      this.Q.set(this.q);
      if (delta < theta) this.phase = 'improve';
      return { phase: 0, delta, done: false };
    }
    // improvement: greedy w.r.t. current V; keep ties so the policy is stable when optimal
    const old = this.P.slice();
    fillQ(env, this.v, this.q, gamma);
    this.Q.set(this.q);
    greedyFromQ(env, this.q, this.P);
    let stable = true;
    for (let i = 0; i < old.length; i++) {
      if (Math.abs(old[i] - this.P[i]) > 1e-9) {
        stable = false;
        break;
      }
    }
    this.phase = 'eval';
    return { phase: 1, delta: 0, done: stable };
  }
  tables(): Tables {
    return { V: this.V, Q: this.Q, P: this.P };
  }
}

export const policyIterationMeta: AlgoMeta = {
  id: 'policy-iteration',
  name: 'Policy iteration',
  short: 'POLICY ITER',
  family: 'dp',
  envKind: 'grid',
  needsModel: true,
  params: [gammaParam(0.9), thetaParam],
  equation: [
    { id: 'lhsV', tex: 'V(s)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'sum-pi', tex: '\\textstyle\\sum_{a}\\pi(a\\mid s)' },
    { id: 'sum-p', tex: "\\textstyle\\sum_{s',r} p(s',r\\mid s,a)" },
    { id: 'lb', tex: '\\Big[' },
    { id: 'R', tex: 'r' },
    { id: 'plus2', tex: '+' },
    { id: 'gamma', tex: '\\gamma', param: 'gamma' },
    { id: 'boot-v', tex: "V(s')" },
    { id: 'rb', tex: '\\Big]' },
    { id: 'improve', tex: ',\\quad \\pi(s)\\leftarrow\\arg\\max_a q_\\pi(s,a)' },
  ],
  pseudocode: [
    L('init', 'Initialize V(s) = 0 and π = uniformly random', 0),
    L('eval-head', '1. Policy evaluation (one sweep per frame):', 0),
    L('bellman-pi', 'V(s) ← Σₐ π(a|s) Σ p(s′,r|s,a)[r + γV(s′)]', 1, ['gamma']),
    L('eval-until', 'repeat until Δ < θ', 1, ['theta']),
    L('improve-head', '2. Policy improvement (one frame):', 0),
    L('improve', 'π(s) ← argmaxₐ Σ p(s′,r|s,a)[r + γV(s′)]', 1),
    L('stable', 'if π did not change: stop, else go to 1', 1),
  ],
  backup: {
    nodes: valueIterationMeta.backup.nodes,
    edges: valueIterationMeta.backup.edges,
    arcs: [{ id: 'arc-top', over: ['aL', 'a0', 'aR'], label: 'Σπ' }],
    caption: 'Evaluate: expectation over the policy’s actions. Improve: take the max.',
  },
  summary:
    'Alternates two phases: evaluate how good the current policy is (sweeps until the values settle), then make the policy greedy with respect to those values. It stops when improving changes nothing.',
  watchFor: ['EVALUATE frames move only the values; IMPROVE frames flip arrows.', 'Usually needs only a handful of improvements, even on big grids.'],
  reference: 'Sutton & Barto §4.3',
  traits: {
    target: 'the expected return of its current policy, then switches to the greedy policy',
    onPolicy: true,
    model: 'given',
    timing: 'in evaluation sweeps followed by improvement steps',
    learns: 'state values V(s) and an explicit policy π',
  },
};
