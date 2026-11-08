import type { AlgoMeta, GridAgent, ParamValues, PseudoLine, StepInfo, Tables, Transition } from '../../types';
import type { Rng } from '../../rng';
import { ACTIONS } from '../../envs/gridworld';
import { alphaParam, bool, gammaParam, num, sampleRow, softmaxRow } from '../common';

const L = (id: string, text: string, indent: number, params?: string[]): PseudoLine => ({ id, text, indent, params });

/** Tabular softmax policy: π(a|s) = exp θ(s,a) / Σ_b exp θ(s,b). ∇_θ(s,·) ln π(a|s) = 1[·=a] − π(·|s). */
abstract class SoftmaxAgent implements GridAgent {
  theta: Float32Array;
  P: Float32Array;
  V: Float32Array;
  lastExplore = false;
  protected p: ParamValues = {};

  constructor(S: number, params: ParamValues, protected readonly rng: Rng) {
    this.theta = new Float32Array(S * ACTIONS);
    this.P = new Float32Array(S * ACTIONS).fill(1 / ACTIONS);
    this.V = new Float32Array(S);
    this.setParams(params);
  }
  setParams(p: ParamValues) {
    this.p = p;
  }
  act(s: number): number {
    const off = s * ACTIONS;
    const a = sampleRow(this.P, off, ACTIONS, this.rng);
    let best = 0;
    for (let b = 1; b < ACTIONS; b++) if (this.P[off + b] > this.P[off + best]) best = b;
    this.lastExplore = this.P[off + a] < this.P[off + best] - 1e-6;
    return a;
  }
  /** θ(s,·) += step · (1[·=a] − π(·|s)), then refresh π(·|s). */
  protected policyStep(s: number, a: number, step: number) {
    const off = s * ACTIONS;
    for (let b = 0; b < ACTIONS; b++) this.theta[off + b] += step * ((b === a ? 1 : 0) - this.P[off + b]);
    softmaxRow(this.theta, this.P, off, ACTIONS);
  }
  abstract beginEpisode(ep: number, s: number): void;
  abstract update(s: number, a: number, tr: Transition): StepInfo | null;
  abstract endEpisode(truncated: boolean, lastState: number): void;
  tables(): Tables {
    return { P: this.P, V: this.V };
  }
}

// ==========================================================================================
// REINFORCE, optionally with a learned state-value baseline (Sutton & Barto §13.3–13.4)
// ==========================================================================================

export class ReinforceAgent extends SoftmaxAgent {
  private traj: { s: number; a: number; r: number }[] = [];
  beginEpisode() {
    this.traj = [];
  }
  update(s: number, a: number, tr: Transition): StepInfo {
    this.traj.push({ s, a, r: tr.reward });
    return { line: 'generate', s: -1 };
  }
  endEpisode() {
    const gamma = num(this.p, 'gamma', 0.99);
    const aTheta = num(this.p, 'alphaTheta', 0.1);
    const aW = num(this.p, 'alphaW', 0.1);
    const useBaseline = bool(this.p, 'baseline');
    const T = this.traj.length;
    const G = new Float64Array(T);
    let acc = 0;
    for (let t = T - 1; t >= 0; t--) {
      acc = gamma * acc + this.traj[t].r;
      G[t] = acc;
    }
    let disc = 1;
    for (let t = 0; t < T; t++) {
      const { s, a } = this.traj[t];
      let adv = G[t];
      if (useBaseline) {
        const d = G[t] - this.V[s];
        this.V[s] += aW * d;
        adv = d;
      }
      this.policyStep(s, a, aTheta * disc * adv);
      disc *= gamma;
    }
    this.traj = [];
  }
}

export const reinforceMeta: AlgoMeta = {
  id: 'reinforce',
  name: 'REINFORCE',
  short: 'REINFORCE',
  family: 'policy-gradient',
  envKind: 'grid',
  needsModel: false,
  params: [
    alphaParam(0.1, 'alphaTheta', 'Policy step', 'α^θ'),
    gammaParam(0.99),
    {
      id: 'baseline',
      label: 'Baseline b(s)',
      scale: 'bool',
      default: true,
      explainUp: 'On: subtract a learned V(s) from the return. Same direction on average, far less noise.',
      explainDown: 'Off: raw returns scale the gradient, so updates are much noisier.',
      affects: [],
    },
    alphaParam(0.1, 'alphaW', 'Baseline step', 'α^w'),
  ],
  equation: [
    { id: 'lhsT', tex: '\\theta' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'oldT', tex: '\\theta' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha^{\\theta}', param: 'alphaTheta' },
    { id: 'gamma', tex: '\\gamma^{t}', param: 'gamma' },
    { id: 'G', tex: '\\big(G_t' },
    { id: 'baseline', tex: '-\\,b(S_t)\\big)', param: 'baseline' },
    { id: 'grad', tex: '\\nabla\\ln\\pi(A_t\\mid S_t,\\theta)' },
  ],
  pseudocode: [
    L('init', 'Initialize θ = 0 (uniform policy), V = 0', 0),
    L('loop-ep', 'Loop for each episode:', 0),
    L('generate', 'Generate an episode by sampling A ~ π(·|S, θ)', 1),
    L('backward', 'Loop t = 0 … T−1:', 1),
    L('G', 'G ← Σₖ γᵏ R_{t+k+1}', 2, ['gamma']),
    L('baseline', 'δ ← G − V(S_t);   V(S_t) ← V(S_t) + α^w δ', 2, ['baseline', 'alphaW']),
    L('update-pg', 'θ ← θ + α^θ γᵗ δ ∇ln π(A_t|S_t, θ)', 2, ['alphaTheta']),
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
    caption: 'Monte Carlo return used as the weight on ∇ln π, with no bootstrapping.',
  },
  summary:
    'Learns the policy directly. After each episode it makes the actions it took more likely in proportion to how good the return was (minus a baseline). There is no value table to take a max over.',
  watchFor: ['Arrow length is π(a|s), a probability, not a value.', 'Turn the baseline off and watch the learning curve get much noisier.'],
  reference: 'Sutton & Barto §13.3–13.4',
  traits: {
    target: 'making taken actions more likely in proportion to the full return they led to',
    onPolicy: true,
    model: 'none',
    timing: 'only at the end of each episode',
    learns: 'a stochastic policy π(a|s) directly',
  },
};

// ==========================================================================================
// One-step actor-critic (Sutton & Barto §13.5)
// ==========================================================================================

export class ActorCriticAgent extends SoftmaxAgent {
  private I = 1;
  beginEpisode() {
    this.I = 1;
  }
  update(s: number, a: number, tr: Transition): StepInfo {
    const gamma = num(this.p, 'gamma', 0.99);
    const aTheta = num(this.p, 'alphaTheta', 0.1);
    const aW = num(this.p, 'alphaW', 0.2);
    const target = tr.reward + (tr.done ? 0 : gamma * this.V[tr.next]);
    const delta = target - this.V[s];
    this.V[s] += aW * delta;
    this.policyStep(s, a, aTheta * this.I * delta);
    this.I *= gamma;
    return { line: 'actor', s, target, delta };
  }
  endEpisode() {}
}

export const actorCriticMeta: AlgoMeta = {
  id: 'actor-critic',
  name: 'Actor-critic',
  short: 'ACTOR-CRITIC',
  family: 'policy-gradient',
  envKind: 'grid',
  needsModel: false,
  params: [
    alphaParam(0.1, 'alphaTheta', 'Actor step', 'α^θ'),
    alphaParam(0.2, 'alphaW', 'Critic step', 'α^w'),
    gammaParam(0.99),
  ],
  equation: [
    { id: 'delta-def', tex: '\\delta = R +' },
    { id: 'gamma', tex: '\\gamma', param: 'gamma' },
    { id: 'boot-v', tex: "V(S')" },
    { id: 'minus', tex: '-' },
    { id: 'oldV', tex: 'V(S)' },
    { id: 'sep', tex: ',\\quad' },
    { id: 'lhsT', tex: '\\theta' },
    { id: 'arrow', tex: '\\leftarrow' },
    { id: 'oldT', tex: '\\theta' },
    { id: 'plus', tex: '+' },
    { id: 'alpha', tex: '\\alpha^{\\theta}', param: 'alphaTheta' },
    { id: 'I', tex: 'I\\,\\delta' },
    { id: 'grad', tex: '\\nabla\\ln\\pi(A\\mid S,\\theta)' },
  ],
  pseudocode: [
    L('init', 'Initialize θ = 0, V = 0', 0),
    L('loop-ep', 'Loop for each episode:', 0),
    L('first', 'S ← start;  I ← 1', 1),
    L('loop-step', 'Loop for each step of episode:', 1),
    L('act', 'A ~ π(·|S, θ);  take A, observe R, S′', 2),
    L('delta', 'δ ← R + γV(S′) − V(S)', 2, ['gamma']),
    L('critic', 'V(S) ← V(S) + α^w δ        (critic)', 2, ['alphaW']),
    L('actor', 'θ ← θ + α^θ I δ ∇ln π(A|S, θ)   (actor)', 2, ['alphaTheta']),
    L('advance', 'I ← γI;  S ← S′', 2, ['gamma']),
  ],
  backup: {
    nodes: [
      { id: 's0', kind: 'state', x: 50, y: 10 },
      { id: 'a0', kind: 'action', x: 50, y: 40 },
      { id: 's1', kind: 'state', x: 50, y: 72 },
    ],
    edges: [['s0', 'a0'], ['a0', 's1']],
    caption: 'One real step; the critic’s TD error δ scores the actor’s action.',
  },
  summary:
    'Two learners in one. The critic learns V(s) by TD. The actor nudges the policy toward actions that turned out better than the critic expected (δ > 0) and away from those that were worse.',
  watchFor: ['The heatmap is the critic (V), the arrows are the actor (π).', 'It learns during the episode, unlike REINFORCE. Compare their curves in 2P VERSUS.'],
  reference: 'Sutton & Barto §13.5',
  traits: {
    target: 'the critic’s one-step TD error, used to score each action',
    onPolicy: true,
    model: 'none',
    timing: 'after every step',
    learns: 'a policy π(a|s) (actor) and state values V(s) (critic)',
  },
};
