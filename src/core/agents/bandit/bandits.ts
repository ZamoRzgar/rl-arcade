import type { AlgoMeta, BackupSpec, BanditAgent, BanditExtraKind, BanditTables, ParamValues, PseudoLine } from '../../types';
import type { Rng } from '../../rng';
import { argmaxRandom, bool, epsParam, num, q0Param, sampleRow, softmaxRow, str } from '../common';

const L = (id: string, text: string, indent: number, params?: string[]): PseudoLine => ({ id, text, indent, params });

const banditBackup = (label: string, caption: string): BackupSpec => ({
  nodes: [
    { id: 's0', kind: 'state', x: 50, y: 14 },
    { id: 'b1', kind: 'action', x: 14, y: 56 },
    { id: 'aL', kind: 'action', x: 32, y: 56 },
    { id: 'a0', kind: 'action', x: 50, y: 56 },
    { id: 'aR', kind: 'action', x: 68, y: 56 },
    { id: 'b5', kind: 'action', x: 86, y: 56 },
  ],
  edges: [['s0', 'b1'], ['s0', 'aL'], ['s0', 'a0'], ['s0', 'aR'], ['s0', 'b5']],
  arcs: [{ id: 'arc-top', over: ['b1', 'aL', 'a0', 'aR', 'b5'], label }],
  caption,
});

const sampleAvgLine = L('update-b', 'N(A) ← N(A) + 1;   Q(A) ← Q(A) + 1/N(A) · [R − Q(A)]', 1);

// ------------------------------------------------------------------ ε-greedy

export class EpsGreedyBandit implements BanditAgent {
  readonly xKind: BanditExtraKind = 'none';
  Q: Float32Array;
  N: Float32Array;
  X: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  constructor(private readonly k: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(k).fill(num(params, 'q0', 0));
    this.N = new Float32Array(k);
    this.X = new Float32Array(k * 2);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  act(): number {
    const eps = num(this.p, 'epsilon', 0.1);
    const coin = this.rng.next();
    if (coin < eps) {
      const a = this.rng.int(this.k);
      const greedy = argmaxRandom(this.Q, 0, this.k, this.rng);
      this.lastExplore = this.Q[a] < this.Q[greedy] - 1e-12;
      return a;
    }
    this.lastExplore = false;
    return argmaxRandom(this.Q, 0, this.k, this.rng);
  }
  update(a: number, r: number): string {
    this.N[a] += 1;
    const step = str(this.p, 'stepMode', 'sample-average') === 'constant' ? num(this.p, 'alpha', 0.1) : 1 / this.N[a];
    this.Q[a] += step * (r - this.Q[a]);
    return 'update-b';
  }
  tables(): BanditTables {
    return { Q: this.Q, N: this.N, X: this.X };
  }
}

export const epsGreedyBanditMeta: AlgoMeta = {
  id: 'eps-greedy',
  name: 'ε-greedy bandit',
  short: 'ε-GREEDY',
  family: 'bandit',
  envKind: 'bandit',
  needsModel: false,
  params: [
    epsParam(0.1),
    { ...q0Param, explainUp: 'Optimistic Q₀ (try +5): every arm looks great until tried, so even ε = 0 explores early.' },
    {
      id: 'stepMode',
      label: 'Averaging',
      scale: 'enum',
      options: ['sample-average', 'constant'],
      default: 'sample-average',
      explainUp: 'constant α: recent rewards count more, which tracks a changing bandit.',
      explainDown: 'sample-average: the exact mean of all rewards seen so far.',
      affects: [],
    },
    {
      id: 'alpha',
      label: 'Step size',
      symbol: 'α',
      scale: 'log',
      min: 0.001,
      max: 1,
      default: 0.1,
      explainUp: 'Bigger steps (constant mode): faster tracking, noisier estimates.',
      explainDown: 'Smaller steps: smoother, slower estimates.',
      affects: [],
    },
  ],
  equation: [
    { id: 'lhsQa', tex: 'Q(A)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'oldQa', tex: 'Q(A)' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\tfrac{1}{N(A)}', param: 'alpha' },
    { id: 'lb', tex: '\\big[' },
    { id: 'R', tex: 'R' },
    { id: 'minus', tex: '-' },
    { id: 'oldQa2', tex: 'Q(A)' },
    { id: 'rb', tex: '\\big]' },
    { id: 'select', tex: ',\\;\\; A = \\begin{cases}\\arg\\max_a Q(a) & 1-\\varepsilon\\\\ \\text{random} & \\varepsilon\\end{cases}', param: 'epsilon' },
  ],
  pseudocode: [
    L('init', 'Q(a) ← Q₀, N(a) ← 0 for all arms', 0, ['q0']),
    L('loop', 'Loop for each step:', 0),
    L('select-eps', 'A ← argmaxₐ Q(a) with prob 1−ε, else a random arm', 1, ['epsilon']),
    L('pull', 'R ← pull(A)', 1),
    sampleAvgLine,
  ],
  backup: banditBackup('ε', 'One state, k arms. Each pull updates only the pulled arm’s estimate.'),
  summary: 'Pull the arm that looks best, except a fraction ε of the time when it pulls a random arm to keep learning about the others.',
  watchFor: ['With ε = 0 it often locks onto a mediocre arm forever. Watch % optimal flatline.', 'Optimistic Q₀ = 5 with ε = 0 explores on its own.'],
  reference: 'Sutton & Barto §2.2–2.5',
  traits: { target: 'the average reward of each arm', onPolicy: true, model: 'none', timing: 'after every pull', learns: 'arm values Q(a)' },
};

// ------------------------------------------------------------------ UCB1

export class UcbBandit implements BanditAgent {
  readonly xKind: BanditExtraKind = 'ucb';
  Q: Float32Array;
  N: Float32Array;
  X: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private t = 0;
  constructor(private readonly k: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(k);
    this.N = new Float32Array(k);
    this.X = new Float32Array(k * 2);
    this.setParams(params);
    this.refresh();
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  private refresh() {
    const c = num(this.p, 'c', 2);
    for (let a = 0; a < this.k; a++) {
      this.X[a * 2] = this.N[a] === 0 ? Infinity : c * Math.sqrt(Math.log(Math.max(1, this.t)) / this.N[a]);
    }
  }
  act(): number {
    this.t += 1;
    this.refresh();
    const score = new Float32Array(this.k);
    for (let a = 0; a < this.k; a++) score[a] = this.N[a] === 0 ? 1e30 : this.Q[a] + this.X[a * 2];
    const a = argmaxRandom(score, 0, this.k, this.rng);
    let gq = 0;
    for (let b = 1; b < this.k; b++) if (this.Q[b] > this.Q[gq]) gq = b;
    this.lastExplore = this.N[a] > 0 && this.Q[a] < this.Q[gq] - 1e-12;
    return a;
  }
  update(a: number, r: number): string {
    this.N[a] += 1;
    this.Q[a] += (r - this.Q[a]) / this.N[a];
    this.refresh();
    return 'update-b';
  }
  tables(): BanditTables {
    return { Q: this.Q, N: this.N, X: this.X };
  }
}

export const ucbMeta: AlgoMeta = {
  id: 'ucb',
  name: 'UCB1',
  short: 'UCB',
  family: 'bandit',
  envKind: 'bandit',
  needsModel: false,
  params: [
    {
      id: 'c',
      label: 'Confidence',
      symbol: 'c',
      scale: 'linear',
      min: 0,
      max: 5,
      step: 0.1,
      default: 2,
      explainUp: 'Higher c: wider confidence bars, so uncertain arms get tried more often.',
      explainDown: 'c = 0 is pure greedy: no bonus for uncertainty.',
      affects: ['viz:whiskers'],
    },
  ],
  equation: [
    { id: 'A', tex: 'A_t =' },
    { id: 'argmax', tex: '\\arg\\max_{a}' },
    { id: 'lb', tex: '\\Big[' },
    { id: 'Qa', tex: 'Q(a)' },
    { id: 'plus', tex: '+' },
    { id: 'c', tex: 'c', param: 'c' },
    { id: 'bonus', tex: '\\sqrt{\\tfrac{\\ln t}{N(a)}}' },
    { id: 'rb', tex: '\\Big]' },
  ],
  pseudocode: [
    L('init', 'Q(a) ← 0, N(a) ← 0 for all arms', 0),
    L('loop', 'Loop for each step t:', 0),
    L('select-ucb', 'A ← argmaxₐ [ Q(a) + c·√(ln t / N(a)) ]   (untried arms first)', 1, ['c']),
    L('pull', 'R ← pull(A)', 1),
    sampleAvgLine,
  ],
  backup: banditBackup('max+bonus', 'Optimism under uncertainty: act as if each arm were as good as plausibly possible.'),
  summary: 'Adds an uncertainty bonus to each arm’s estimate and pulls the arm with the highest optimistic score. Rarely tried arms get a big bonus, so exploration is targeted, not random.',
  watchFor: ['The whiskers above each bar are the confidence bonus. They shrink as an arm gets pulled.', 'Beats ε-greedy on the 10-armed testbed. Compare them in 2P VERSUS.'],
  reference: 'Sutton & Barto §2.7',
  traits: { target: 'the average reward plus an uncertainty bonus', onPolicy: true, model: 'none', timing: 'after every pull', learns: 'arm values Q(a) and pull counts N(a)' },
};

// ------------------------------------------------------------------ Thompson sampling

export class ThompsonBandit implements BanditAgent {
  readonly xKind: BanditExtraKind;
  Q: Float32Array;
  N: Float32Array;
  X: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private sum: Float32Array;
  constructor(private readonly k: number, private readonly bernoulli: boolean, params: ParamValues, private readonly rng: Rng) {
    this.xKind = bernoulli ? 'beta' : 'normal';
    this.Q = new Float32Array(k);
    this.N = new Float32Array(k);
    this.X = new Float32Array(k * 2);
    this.sum = new Float32Array(k);
    this.setParams(params);
    this.refresh();
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  /** Posterior parameters: Beta(a, b) for Bernoulli; Normal(mean, sd) with known noise σ = 1 for Gaussian. */
  private refresh() {
    const prior = num(this.p, 'prior', 1);
    for (let a = 0; a < this.k; a++) {
      if (this.bernoulli) {
        this.X[a * 2] = prior + this.sum[a];
        this.X[a * 2 + 1] = prior + this.N[a] - this.sum[a];
        this.Q[a] = this.X[a * 2] / (this.X[a * 2] + this.X[a * 2 + 1]);
      } else {
        const prec = prior + this.N[a];
        this.X[a * 2] = this.sum[a] / prec;
        this.X[a * 2 + 1] = 1 / Math.sqrt(prec);
        this.Q[a] = this.X[a * 2];
      }
    }
  }
  act(): number {
    const draws = new Float32Array(this.k);
    for (let a = 0; a < this.k; a++) {
      draws[a] = this.bernoulli ? this.rng.beta(this.X[a * 2], this.X[a * 2 + 1]) : this.X[a * 2] + this.X[a * 2 + 1] * this.rng.normal();
    }
    const a = argmaxRandom(draws, 0, this.k, this.rng);
    let gq = 0;
    for (let b = 1; b < this.k; b++) if (this.Q[b] > this.Q[gq]) gq = b;
    this.lastExplore = this.Q[a] < this.Q[gq] - 1e-12;
    return a;
  }
  update(a: number, r: number): string {
    this.N[a] += 1;
    this.sum[a] += r;
    this.refresh();
    return 'posterior';
  }
  tables(): BanditTables {
    return { Q: this.Q, N: this.N, X: this.X };
  }
}

export const thompsonMeta: AlgoMeta = {
  id: 'thompson',
  name: 'Thompson sampling',
  short: 'THOMPSON',
  family: 'bandit',
  envKind: 'bandit',
  needsModel: false,
  params: [
    {
      id: 'prior',
      label: 'Prior strength',
      scale: 'log',
      min: 0.1,
      max: 20,
      default: 1,
      explainUp: 'Stronger prior: the agent needs more pulls before believing an arm differs from average.',
      explainDown: 'Weaker prior: beliefs swing hard after the first few rewards.',
      affects: ['viz:posterior'],
    },
  ],
  equation: [
    { id: 'sample', tex: '\\tilde\\theta_a \\sim \\text{Posterior}(a\\mid\\text{data})' },
    { id: 'sep', tex: ',\\quad' },
    { id: 'A', tex: 'A_t =' },
    { id: 'argmax', tex: '\\arg\\max_{a}' },
    { id: 'theta-a', tex: '\\tilde\\theta_a' },
  ],
  pseudocode: [
    L('init', 'Prior belief for every arm (Beta(1,1) or Normal)', 0, ['prior']),
    L('loop', 'Loop for each step:', 0),
    L('draw', 'For each arm, draw θ̃ₐ from its posterior', 1),
    L('select-ts', 'A ← argmaxₐ θ̃ₐ', 1),
    L('pull', 'R ← pull(A)', 1),
    L('posterior', 'Update the posterior of arm A with R (Bayes’ rule)', 1),
  ],
  backup: banditBackup('sample', 'Probability matching: pull each arm with the probability that it is the best.'),
  summary: 'Keeps a probability distribution over how good each arm might be. Every step it samples one plausible world from those beliefs and acts optimally for it, so uncertain arms get tried exactly as often as they might be best.',
  watchFor: ['The curves above the arms are the beliefs. Good arms sharpen into tall narrow peaks.', 'Very little regret once one arm clearly dominates.'],
  reference: 'Sutton & Barto §2.10; Russo et al. (2018)',
  traits: { target: 'a posterior belief about each arm’s mean, sampling from it to choose', onPolicy: true, model: 'none', timing: 'after every pull', learns: 'a Bayesian posterior per arm' },
};

// ------------------------------------------------------------------ Gradient bandit

export class GradientBandit implements BanditAgent {
  readonly xKind: BanditExtraKind = 'softmax';
  Q: Float32Array;
  N: Float32Array;
  X: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private H: Float32Array;
  private pi: Float32Array;
  private avg = 0;
  private t = 0;
  constructor(private readonly k: number, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(k);
    this.N = new Float32Array(k);
    this.X = new Float32Array(k * 2);
    this.H = new Float32Array(k);
    this.pi = new Float32Array(k).fill(1 / k);
    this.setParams(params);
    this.refresh();
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  private refresh() {
    softmaxRow(this.H, this.pi, 0, this.k);
    for (let a = 0; a < this.k; a++) {
      this.X[a * 2] = this.pi[a];
      this.X[a * 2 + 1] = this.H[a];
    }
  }
  act(): number {
    const a = sampleRow(this.pi, 0, this.k, this.rng);
    let best = 0;
    for (let b = 1; b < this.k; b++) if (this.pi[b] > this.pi[best]) best = b;
    this.lastExplore = this.pi[a] < this.pi[best] - 1e-9;
    return a;
  }
  update(a: number, r: number): string {
    const alpha = num(this.p, 'alpha', 0.1);
    this.t += 1;
    this.N[a] += 1;
    this.Q[a] += (r - this.Q[a]) / this.N[a];
    const baseline = bool(this.p, 'baseline') ? this.avg : 0;
    for (let b = 0; b < this.k; b++) this.H[b] += alpha * (r - baseline) * ((b === a ? 1 : 0) - this.pi[b]);
    this.avg += (r - this.avg) / this.t;
    this.refresh();
    return 'update-h';
  }
  tables(): BanditTables {
    return { Q: this.Q, N: this.N, X: this.X };
  }
}

export const gradientBanditMeta: AlgoMeta = {
  id: 'gradient-bandit',
  name: 'Gradient bandit',
  short: 'GRADIENT',
  family: 'bandit',
  envKind: 'bandit',
  needsModel: false,
  params: [
    {
      id: 'alpha',
      label: 'Step size',
      symbol: 'α',
      scale: 'log',
      min: 0.01,
      max: 1,
      default: 0.1,
      explainUp: 'Bigger steps: preferences jump quickly, so it commits fast (maybe too fast).',
      explainDown: 'Smaller steps: preferences drift slowly and keep exploring for longer.',
      affects: [],
    },
    {
      id: 'baseline',
      label: 'Baseline R̄',
      scale: 'bool',
      default: true,
      explainUp: 'On: rewards are judged relative to the running average, which matters when all rewards are shifted.',
      explainDown: 'Off: every positive reward raises the pulled arm, so performance collapses if rewards are offset.',
      affects: [],
    },
  ],
  equation: [
    { id: 'lhsH', tex: 'H(A)' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'oldH', tex: 'H(A)' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha', param: 'alpha' },
    { id: 'lb', tex: '(R' },
    { id: 'baseline', tex: '-\\bar R)', param: 'baseline' },
    { id: 'grad', tex: '(1-\\pi(A))' },
    { id: 'softmax', tex: ',\\;\\; \\pi(a)=\\tfrac{e^{H(a)}}{\\sum_b e^{H(b)}}' },
  ],
  pseudocode: [
    L('init', 'H(a) ← 0 for all arms (uniform π)', 0),
    L('loop', 'Loop for each step:', 0),
    L('select-g', 'A ~ π = softmax(H)', 1),
    L('pull', 'R ← pull(A)', 1),
    L('update-h', 'H(a) ← H(a) + α(R − R̄)(1[a=A] − π(a)) for all a', 1, ['alpha', 'baseline']),
    L('avg', 'R̄ ← running average of rewards', 1, ['baseline']),
  ],
  backup: banditBackup('softmax', 'Stochastic gradient ascent on expected reward, over arm preferences.'),
  summary: 'Learns preferences, not values. Arms are pulled with softmax probabilities; a better-than-average reward raises the pulled arm’s preference and lowers the others.',
  watchFor: ['The bars show π(a), the probability of pulling each arm.', 'Turn the baseline off and see how much worse it gets.'],
  reference: 'Sutton & Barto §2.8',
  traits: { target: 'arm preferences, by gradient ascent on expected reward', onPolicy: true, model: 'none', timing: 'after every pull', learns: 'preferences H(a) and a softmax policy' },
};
