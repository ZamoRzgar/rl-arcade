import type { AlgoMeta, GridAgent, ParamDef, ParamValues, PseudoLine, StepInfo, Tables, Transition } from '../../types';
import type { Rng } from '../../rng';
import { ACTIONS } from '../../envs/gridworld';
import { alphaParam, argmaxRandom, epsDecayParam, epsGreedy, epsParam, gammaParam, num, q0Param, rowMax, scheduledEps, str } from '../common';
import { BOOT_MAX, POLICY_TOKEN, qBackup, qPseudo, tdEquation } from './td';

const L = (id: string, text: string, indent: number, params?: string[]): PseudoLine => ({ id, text, indent, params });

// ==========================================================================================
// Monte Carlo control, on-policy, ε-soft (Sutton & Barto §5.4). Constant-α or sample-average.
// ==========================================================================================

export class MonteCarloAgent implements GridAgent {
  Q: Float32Array;
  V: Float32Array;
  N: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private eps = 0.1;
  private traj: { s: number; a: number; r: number }[] = [];

  constructor(S: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(S * ACTIONS).fill(num(params, 'q0', 0));
    this.N = new Float32Array(S * ACTIONS);
    this.V = new Float32Array(S);
    for (let s = 0; s < S; s++) this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  beginEpisode(ep: number) {
    this.eps = scheduledEps(this.p, ep);
    this.traj = [];
  }
  act(s: number): number {
    const r = epsGreedy(this.Q, s * ACTIONS, ACTIONS, this.eps, this.rng);
    this.lastExplore = r.explore;
    return r.a;
  }
  update(s: number, a: number, tr: Transition): StepInfo {
    this.traj.push({ s, a, r: tr.reward });
    return { line: 'record', s: -1 };
  }
  endEpisode() {
    const gamma = num(this.p, 'gamma', 1);
    const alphaMode = str(this.p, 'stepMode', 'sample-average');
    const alpha = num(this.p, 'alpha', 0.1);
    const firstVisit = str(this.p, 'visit', 'first-visit') === 'first-visit';
    const T = this.traj.length;
    // first occurrence index of each (s,a) for first-visit MC
    const first = new Map<number, number>();
    if (firstVisit) {
      for (let t = 0; t < T; t++) {
        const k = this.traj[t].s * ACTIONS + this.traj[t].a;
        if (!first.has(k)) first.set(k, t);
      }
    }
    let G = 0;
    for (let t = T - 1; t >= 0; t--) {
      const { s, a, r } = this.traj[t];
      G = gamma * G + r;
      const k = s * ACTIONS + a;
      if (firstVisit && first.get(k) !== t) continue;
      this.N[k] += 1;
      const step = alphaMode === 'constant' ? alpha : 1 / this.N[k];
      this.Q[k] += step * (G - this.Q[k]);
      this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    }
    this.traj = [];
  }
  tables(): Tables {
    return { Q: this.Q, V: this.V };
  }
}

const stepModeParam: ParamDef = {
  id: 'stepMode',
  label: 'Averaging',
  scale: 'enum',
  options: ['sample-average', 'constant'],
  default: 'sample-average',
  explainUp: 'constant α: recent episodes count more, which helps when the world changes.',
  explainDown: 'sample-average: every return counts equally, so the estimate converges to the true mean.',
  affects: [],
};
const visitParam: ParamDef = {
  id: 'visit',
  label: 'Visit rule',
  scale: 'enum',
  options: ['first-visit', 'every-visit'],
  default: 'first-visit',
  explainUp: 'every-visit: each occurrence of (s,a) in an episode counts as a sample.',
  explainDown: 'first-visit: only the first occurrence of (s,a) per episode counts.',
  affects: [],
};
const exploringStartsParam: ParamDef = {
  id: 'exploringStarts',
  label: 'Exploring starts',
  scale: 'bool',
  default: false,
  explainUp: 'On: each episode starts in a random cell with a random first action, so every pair gets tried.',
  explainDown: 'Off: every episode starts at S and exploration comes only from ε.',
  affects: [],
};

export const monteCarloMeta: AlgoMeta = {
  id: 'monte-carlo',
  name: 'Monte Carlo control',
  short: 'MONTE CARLO',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: [gammaParam(0.99), epsParam(0.1), epsDecayParam, stepModeParam, alphaParam(0.1), visitParam, exploringStartsParam, q0Param],
  equation: [
    { id: 'lhs', tex: 'Q(S_t,A_t)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'old', tex: 'Q(S_t,A_t)' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha', param: 'alpha' },
    { id: 'lb', tex: '\\Big[' },
    { id: 'G', tex: 'G_t' },
    { id: 'minus', tex: '-' },
    { id: 'old2', tex: 'Q(S_t,A_t)' },
    { id: 'rb', tex: '\\Big]' },
    { id: 'G-def', tex: ',\\;\\; G_t = \\textstyle\\sum_{k} \\gamma^{k} R_{t+k+1}', param: 'gamma' },
  ],
  pseudocode: [
    L('init', 'Initialize Q(s,a) = Q₀, N(s,a) = 0', 0, ['q0']),
    L('loop-ep', 'Loop for each episode:', 0),
    L('generate', 'Generate an episode with ε-greedy(Q): S₀,A₀,R₁,…,S_T', 1, ['epsilon']),
    L('record', '(record every step, no learning yet)', 2),
    L('G0', 'G ← 0', 1),
    L('backward', 'Loop t = T−1 … 0:', 1),
    L('G', 'G ← γG + R_{t+1}', 2, ['gamma']),
    L('first', 'if first visit of (S_t,A_t)  [or every visit]:', 2),
    L('update-mc', 'Q(S_t,A_t) ← Q(S_t,A_t) + α[G − Q(S_t,A_t)]', 3, ['alpha']),
  ],
  backup: {
    nodes: [
      { id: 's0', kind: 'state', x: 50, y: 8 },
      { id: 'a0', kind: 'action', x: 50, y: 24 },
      { id: 's1', kind: 'state', x: 50, y: 40 },
      { id: 'a1', kind: 'action', x: 50, y: 56 },
      { id: 'dots', kind: 'dots', x: 50, y: 72 },
      { id: 'sT', kind: 'terminal', x: 50, y: 90 },
    ],
    edges: [['s0', 'a0'], ['a0', 's1'], ['s1', 'a1'], ['a1', 'dots'], ['dots', 'sT']],
    caption: 'No bootstrapping: wait for the episode to end, then use the real return.',
  },
  summary:
    'Plays a whole episode, then walks backward through it, nudging every visited Q(s,a) toward the actual return that followed. No bootstrapping and no learning mid-episode.',
  watchFor: [
    'Values change only when an episode ends, all along the trail at once.',
    'Long episodes produce high-variance returns. Compare the learning curve with SARSA on the same level.',
  ],
  reference: 'Sutton & Barto §5.4',
  traits: {
    target: 'the full return actually observed until the episode ends',
    onPolicy: true,
    model: 'none',
    timing: 'only at the end of each episode',
    learns: 'action values Q(s,a)',
  },
};

// ==========================================================================================
// n-step SARSA (Sutton & Barto §7.2)
// ==========================================================================================

export class NStepSarsaAgent implements GridAgent {
  Q: Float32Array;
  V: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private eps = 0.1;
  private S_: number[] = [];
  private A_: number[] = [];
  private R_: number[] = [0];
  private t = 0;
  private pending = -1;

  constructor(S: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(S * ACTIONS).fill(num(params, 'q0', 0));
    this.V = new Float32Array(S);
    for (let s = 0; s < S; s++) this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  beginEpisode(ep: number) {
    this.eps = scheduledEps(this.p, ep);
    this.S_ = [];
    this.A_ = [];
    this.R_ = [0];
    this.t = 0;
    this.pending = -1;
  }
  act(s: number): number {
    if (this.pending >= 0) {
      const a = this.pending;
      this.pending = -1;
      return a;
    }
    const r = epsGreedy(this.Q, s * ACTIONS, ACTIONS, this.eps, this.rng);
    this.lastExplore = r.explore;
    return r.a;
  }

  /** Update Q(S_τ, A_τ) toward the n-step return; bootstrap from (S_{τ+n}, A_{τ+n}) if available. */
  private updateTau(tau: number, T: number, bootS: number, bootA: number): StepInfo {
    const n = Math.max(1, Math.round(num(this.p, 'n', 4)));
    const gamma = num(this.p, 'gamma', 0.9);
    const alpha = num(this.p, 'alpha', 0.5);
    let G = 0;
    const end = Math.min(tau + n, T);
    for (let i = tau + 1; i <= end; i++) G += Math.pow(gamma, i - tau - 1) * this.R_[i];
    if (tau + n < T && bootS >= 0) G += Math.pow(gamma, n) * this.Q[bootS * ACTIONS + bootA];
    const s = this.S_[tau];
    const k = s * ACTIONS + this.A_[tau];
    const delta = G - this.Q[k];
    this.Q[k] += alpha * delta;
    this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    return { line: 'update-n', s, target: G, delta };
  }

  update(s: number, a: number, tr: Transition): StepInfo {
    const n = Math.max(1, Math.round(num(this.p, 'n', 4)));
    const t = this.t;
    this.S_[t] = s;
    this.A_[t] = a;
    this.R_[t + 1] = tr.reward;
    let T = Infinity;
    if (tr.done) {
      T = t + 1;
      this.S_[t + 1] = tr.next;
    } else {
      const r = epsGreedy(this.Q, tr.next * ACTIONS, ACTIONS, this.eps, this.rng);
      this.pending = r.a;
      this.lastExplore = r.explore;
      this.S_[t + 1] = tr.next;
      this.A_[t + 1] = r.a;
    }
    this.t = t + 1;
    const tau = t - n + 1;
    let info: StepInfo = { line: 'store', s: -1 };
    if (tau >= 0) {
      const bs = tau + n <= t + 1 ? this.S_[tau + n] : -1;
      const ba = tau + n <= t + 1 ? this.A_[tau + n] ?? 0 : 0;
      info = this.updateTau(tau, T, tr.done ? -1 : bs, ba);
    }
    if (tr.done) {
      // flush the remaining τ = t−n+2 … T−1
      for (let tt = Math.max(0, tau + 1); tt < T; tt++) info = this.updateTau(tt, T, -1, 0);
    }
    return info;
  }

  endEpisode(truncated: boolean) {
    if (!truncated) return;
    // Time limit, not a real terminal: flush pending updates, bootstrapping from the last pair.
    const n = Math.max(1, Math.round(num(this.p, 'n', 4)));
    const tEnd = this.t; // index of the last state S_t (non-terminal)
    const lastA = this.A_[tEnd] ?? this.pending;
    for (let tau = Math.max(0, tEnd - n + 1); tau < tEnd; tau++) {
      const gamma = num(this.p, 'gamma', 0.9);
      const alpha = num(this.p, 'alpha', 0.5);
      let G = 0;
      for (let i = tau + 1; i <= tEnd; i++) G += Math.pow(gamma, i - tau - 1) * this.R_[i];
      if (lastA !== undefined && lastA >= 0) G += Math.pow(gamma, tEnd - tau) * this.Q[this.S_[tEnd] * ACTIONS + lastA];
      const s = this.S_[tau];
      const k = s * ACTIONS + this.A_[tau];
      this.Q[k] += alpha * (G - this.Q[k]);
      this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    }
    this.pending = -1;
  }
  tables(): Tables {
    return { Q: this.Q, V: this.V };
  }
}

const nParam: ParamDef = {
  id: 'n',
  label: 'Steps n',
  symbol: 'n',
  scale: 'int',
  min: 1,
  max: 32,
  step: 1,
  default: 4,
  explainUp: 'Larger n: each update looks further ahead with real rewards, so credit travels back faster, with more variance.',
  explainDown: 'n = 1 is plain SARSA: credit crawls back one cell per visit.',
  affects: ['viz:nstep'],
};

export const nStepSarsaMeta: AlgoMeta = {
  id: 'nstep-sarsa',
  name: 'n-step SARSA',
  short: 'N-STEP',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: [nParam, alphaParam(0.5), gammaParam(0.9), epsParam(0.1), epsDecayParam, q0Param],
  equation: [
    { id: 'lhs', tex: 'Q(S_\\tau,A_\\tau)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'old', tex: 'Q(S_\\tau,A_\\tau)' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha', param: 'alpha' },
    { id: 'lb', tex: '\\Big[' },
    { id: 'Gn', tex: '\\textstyle\\sum_{i=1}^{n}\\gamma^{i-1}R_{\\tau+i}', param: 'n' },
    { id: 'plus2', tex: '+' },
    { id: 'gamma', tex: '\\gamma^{n}', param: 'gamma' },
    { id: 'boot-sarsa', tex: 'Q(S_{\\tau+n},A_{\\tau+n})' },
    { id: 'minus', tex: '-' },
    { id: 'old2', tex: 'Q(S_\\tau,A_\\tau)' },
    { id: 'rb', tex: '\\Big]' },
    POLICY_TOKEN,
  ],
  pseudocode: [
    L('init', 'Initialize Q(s,a) = Q₀ for all s, a', 0, ['q0']),
    L('loop-ep', 'Loop for each episode:', 0),
    L('first', 'S ← start;  A ← ε-greedy(Q, S)', 1, ['epsilon']),
    L('loop-step', 'Loop for each step t:', 1),
    L('act', 'Take action A, observe R, S′;  A′ ← ε-greedy(Q, S′)', 2, ['epsilon']),
    L('store', 'Store R, S′, A′ in the n-step buffer', 2, ['n']),
    L('tau', 'τ ← t − n + 1   (the step whose estimate is updated now)', 2, ['n']),
    L('update-n', 'Q(S_τ,A_τ) ← Q(S_τ,A_τ) + α[G_{τ:τ+n} − Q(S_τ,A_τ)]', 2, ['alpha', 'gamma', 'n']),
    L('until', 'until τ = T − 1', 1),
  ],
  backup: {
    nodes: [
      { id: 's0', kind: 'state', x: 50, y: 6 },
      { id: 'a0', kind: 'action', x: 50, y: 20 },
      { id: 's1', kind: 'state', x: 50, y: 34 },
      { id: 'a1', kind: 'action', x: 50, y: 48 },
      { id: 'dots', kind: 'dots', x: 50, y: 62 },
      { id: 'sn', kind: 'state', x: 50, y: 76 },
      { id: 'an', kind: 'action', x: 50, y: 90 },
    ],
    edges: [['s0', 'a0'], ['a0', 's1'], ['s1', 'a1'], ['a1', 'dots'], ['dots', 'sn'], ['sn', 'an']],
    caption: 'n real rewards, then bootstrap from Q n steps later.',
  },
  summary: 'Sits between SARSA (n = 1) and Monte Carlo (n = ∞). Each update uses n real rewards before bootstrapping, so credit travels back up to n cells per update.',
  watchFor: ['On Long Hallway, compare n = 1 and n = 8 in 2P VERSUS: value reaches the start far sooner with n = 8.', 'The bracket over the trail shows the n-step window being updated.'],
  reference: 'Sutton & Barto §7.2',
  traits: {
    target: 'n real rewards plus the value of the state–action n steps later',
    onPolicy: true,
    model: 'none',
    timing: 'every step, for the step n steps back',
    learns: 'action values Q(s,a)',
  },
};

// ==========================================================================================
// SARSA(λ) with eligibility traces (Sutton & Barto §12.7, tabular)
// ==========================================================================================

export class SarsaLambdaAgent implements GridAgent {
  Q: Float32Array;
  V: Float32Array;
  E: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private eps = 0.1;
  private pending = -1;
  private touched: number[] = [];
  private isTouched: Uint8Array;

  constructor(S: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(S * ACTIONS).fill(num(params, 'q0', 0));
    this.V = new Float32Array(S);
    this.E = new Float32Array(S * ACTIONS);
    this.isTouched = new Uint8Array(S * ACTIONS);
    for (let s = 0; s < S; s++) this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  beginEpisode(ep: number) {
    this.eps = scheduledEps(this.p, ep);
    this.E.fill(0);
    for (const i of this.touched) this.isTouched[i] = 0;
    this.touched = [];
    this.pending = -1;
  }
  act(s: number): number {
    if (this.pending >= 0) {
      const a = this.pending;
      this.pending = -1;
      return a;
    }
    const r = epsGreedy(this.Q, s * ACTIONS, ACTIONS, this.eps, this.rng);
    this.lastExplore = r.explore;
    return r.a;
  }
  update(s: number, a: number, tr: Transition): StepInfo {
    const alpha = num(this.p, 'alpha', 0.5);
    const gamma = num(this.p, 'gamma', 0.9);
    const lambda = num(this.p, 'lambda', 0.9);
    const replacing = str(this.p, 'trace', 'replacing') === 'replacing';
    const Q = this.Q;
    let boot = 0;
    if (!tr.done) {
      const r = epsGreedy(Q, tr.next * ACTIONS, ACTIONS, this.eps, this.rng);
      this.pending = r.a;
      this.lastExplore = r.explore;
      boot = Q[tr.next * ACTIONS + r.a];
    }
    const k = s * ACTIONS + a;
    const target = tr.reward + gamma * boot;
    const delta = target - Q[k];
    if (!this.isTouched[k]) {
      this.isTouched[k] = 1;
      this.touched.push(k);
    }
    if (replacing) this.E[k] = 1;
    else this.E[k] += 1;
    const decay = gamma * lambda;
    for (const i of this.touched) {
      Q[i] += alpha * delta * this.E[i];
      this.E[i] *= decay;
    }
    // refresh V for touched states
    for (const i of this.touched) {
      const st = (i / ACTIONS) | 0;
      this.V[st] = rowMax(Q, st * ACTIONS, ACTIONS);
    }
    return { line: 'update-l', s, target, delta };
  }
  endEpisode() {
    this.pending = -1;
  }
  tables(): Tables {
    return { Q: this.Q, V: this.V };
  }
}

export const sarsaLambdaMeta: AlgoMeta = {
  id: 'sarsa-lambda',
  name: 'SARSA(λ)',
  short: 'SARSA(λ)',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: [
    {
      id: 'lambda',
      label: 'Trace decay',
      symbol: 'λ',
      scale: 'linear',
      min: 0,
      max: 1,
      step: 0.01,
      default: 0.9,
      explainUp: 'Higher λ: a long fading memory of recent cells, so one surprise updates the whole recent trail (closer to Monte Carlo).',
      explainDown: 'λ = 0 is plain one-step SARSA: only the last cell learns.',
      affects: ['viz:trace'],
    },
    {
      id: 'trace',
      label: 'Trace type',
      scale: 'enum',
      options: ['replacing', 'accumulating'],
      default: 'replacing',
      explainUp: 'accumulating: revisiting a cell stacks its trace above 1.',
      explainDown: 'replacing: revisiting a cell resets its trace to 1, which is usually more stable.',
      affects: ['viz:trace'],
    },
    alphaParam(0.3),
    gammaParam(0.9),
    epsParam(0.1),
    epsDecayParam,
    q0Param,
  ],
  equation: [
    { id: 'lhs', tex: 'Q' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'old', tex: 'Q' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha', param: 'alpha' },
    { id: 'delta', tex: '\\delta\\,E' },
    { id: 'sep', tex: ',\\quad \\delta = R +' },
    { id: 'gamma', tex: '\\gamma', param: 'gamma' },
    { id: 'boot-sarsa', tex: "Q(S',A')" },
    { id: 'minus', tex: '-' },
    { id: 'old2', tex: 'Q(S,A)' },
    { id: 'trace', tex: ',\\quad E \\leftarrow \\gamma\\lambda E', param: 'lambda' },
    POLICY_TOKEN,
  ],
  pseudocode: [
    L('init', 'Initialize Q(s,a) = Q₀', 0, ['q0']),
    L('loop-ep', 'Loop for each episode:', 0),
    L('first', 'E ← 0;  S ← start;  A ← ε-greedy(Q, S)', 1, ['epsilon']),
    L('loop-step', 'Loop for each step of episode:', 1),
    L('act', 'Take action A, observe R, S′;  A′ ← ε-greedy(Q, S′)', 2, ['epsilon']),
    L('delta', 'δ ← R + γ·Q(S′,A′) − Q(S,A)', 2, ['gamma']),
    L('mark', 'E(S,A) ← 1   (replacing)  or  E(S,A) + 1', 2, ['trace']),
    L('update-l', 'for all s,a:  Q(s,a) ← Q(s,a) + α·δ·E(s,a)', 2, ['alpha']),
    L('decay', 'for all s,a:  E(s,a) ← γλ·E(s,a)', 2, ['gamma', 'lambda']),
    L('advance', 'S ← S′;  A ← A′', 2),
  ],
  backup: {
    nodes: [
      { id: 's0', kind: 'state', x: 50, y: 6 },
      { id: 'a0', kind: 'action', x: 50, y: 20 },
      { id: 's1', kind: 'state', x: 50, y: 34 },
      { id: 'a1', kind: 'action', x: 50, y: 48 },
      { id: 'dots', kind: 'dots', x: 50, y: 62 },
      { id: 'sT', kind: 'terminal', x: 50, y: 82 },
    ],
    edges: [['s0', 'a0'], ['a0', 's1'], ['s1', 'a1'], ['a1', 'dots'], ['dots', 'sT']],
    arcs: [{ id: 'arc-lambda', over: ['a1', 'dots', 'sT'], label: 'λ-mix' }],
    caption: 'A λ-weighted mix of every n-step return, computed online with traces.',
  },
  summary:
    'Every visited state–action leaves a fading eligibility trace. Each TD error updates all traced pairs at once, in proportion to their trace, so one surprise teaches the whole recent path.',
  watchFor: ['The glowing tail behind the agent is the eligibility trace. Raise λ and it gets longer.', 'On Long Hallway, credit reaches the start in a few episodes.'],
  reference: 'Sutton & Barto §12.7',
  traits: {
    target: 'a λ-weighted blend of multi-step returns, spread over a fading trace',
    onPolicy: true,
    model: 'none',
    timing: 'after every step, for every recently visited pair',
    learns: 'action values Q(s,a)',
  },
};

// ==========================================================================================
// Dyna-Q (Sutton & Barto §8.2)
// ==========================================================================================

export class DynaQAgent implements GridAgent {
  Q: Float32Array;
  V: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private eps = 0.1;
  /** learned deterministic model: key s*A+a -> [reward, next, done] */
  private model = new Map<number, [number, number, boolean]>();
  private keys: number[] = [];

  constructor(S: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(S * ACTIONS).fill(num(params, 'q0', 0));
    this.V = new Float32Array(S);
    for (let s = 0; s < S; s++) this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  beginEpisode(ep: number) {
    this.eps = scheduledEps(this.p, ep);
  }
  act(s: number): number {
    const r = epsGreedy(this.Q, s * ACTIONS, ACTIONS, this.eps, this.rng);
    this.lastExplore = r.explore;
    return r.a;
  }
  private qUpdate(s: number, a: number, r: number, next: number, done: boolean): { target: number; delta: number } {
    const alpha = num(this.p, 'alpha', 0.1);
    const gamma = num(this.p, 'gamma', 0.95);
    const k = s * ACTIONS + a;
    const target = r + (done ? 0 : gamma * rowMax(this.Q, next * ACTIONS, ACTIONS));
    const delta = target - this.Q[k];
    this.Q[k] += alpha * delta;
    this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
    return { target, delta };
  }
  update(s: number, a: number, tr: Transition): StepInfo {
    const { target, delta } = this.qUpdate(s, a, tr.reward, tr.next, tr.done);
    const k = s * ACTIONS + a;
    if (!this.model.has(k)) this.keys.push(k);
    this.model.set(k, [tr.reward, tr.next, tr.done]);
    const n = Math.max(0, Math.round(num(this.p, 'planning', 10)));
    const plan: number[] = [];
    for (let i = 0; i < n; i++) {
      const pk = this.keys[this.rng.int(this.keys.length)];
      const m = this.model.get(pk)!;
      const ps = (pk / ACTIONS) | 0;
      this.qUpdate(ps, pk % ACTIONS, m[0], m[1], m[2]);
      if (plan.length < 4) plan.push(ps);
    }
    return { line: n > 0 ? 'plan' : 'update', s, target, delta, plan };
  }
  endEpisode() {}
  tables(): Tables {
    return { Q: this.Q, V: this.V };
  }
}

export const dynaQMeta: AlgoMeta = {
  id: 'dyna-q',
  name: 'Dyna-Q',
  short: 'DYNA-Q',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: [
    {
      id: 'planning',
      label: 'Planning steps',
      symbol: 'n',
      scale: 'int',
      min: 0,
      max: 100,
      step: 1,
      default: 10,
      explainUp: 'More planning: after each real step, replay more remembered transitions. Far fewer real episodes needed.',
      explainDown: 'n = 0 is plain Q-learning: it learns only from real experience.',
      affects: ['viz:planning'],
    },
    alphaParam(0.1),
    gammaParam(0.95),
    epsParam(0.1),
    epsDecayParam,
    q0Param,
  ],
  equation: [
    ...tdEquation(BOOT_MAX),
    { id: 'dyna', tex: '\\;\\;\\times(1+n)\\text{ using }\\hat{M}(s,a)', param: 'planning' },
    POLICY_TOKEN,
  ],
  pseudocode: [
    ...qPseudo.slice(0, 7),
    L('model', 'Model(S,A) ← R, S′   (remember what happened)', 2),
    L('plan', 'Repeat n times: pick a remembered (s,a), get r, s′ from the model, do the same Q update', 2, ['planning']),
    ...qPseudo.slice(7),
  ],
  backup: {
    nodes: [...qBackup.nodes, { id: 'model', kind: 'model', x: 86, y: 34, label: 'M̂' }],
    edges: qBackup.edges,
    dashed: [['model', 's1']],
    arcs: qBackup.arcs,
    caption: 'Q-learning backups from real steps, plus extra backups imagined from a learned model.',
  },
  summary:
    'Q-learning plus a memory of what each action did. After every real step it replays n imagined transitions from that memory, so planning and learning share one update rule.',
  watchFor: ['Ghost flashes across the grid are planning updates, not real moves.', 'On Shifting Maze, compare n = 0 and n = 50: same real experience, much faster learning.'],
  reference: 'Sutton & Barto §8.2',
  traits: {
    target: 'the best next action’s value, from real steps and from imagined steps replayed from its model',
    onPolicy: false,
    model: 'learned',
    timing: 'after every real step, plus n planning updates',
    learns: 'action values Q(s,a) and a model of the world',
  },
};

export { argmaxRandom };
