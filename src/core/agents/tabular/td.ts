import type { AlgoMeta, BackupSpec, EqToken, GridAgent, ParamValues, PseudoLine, StepInfo, Tables, Transition } from '../../types';
import type { Rng } from '../../rng';
import { ACTIONS } from '../../envs/gridworld';
import { alphaParam, epsDecayParam, epsGreedy, epsParam, gammaParam, num, q0Param, rowMax, scheduledEps } from '../common';

// ------------------------------------------------------------------------------------------
// One-step TD control: SARSA, Q-learning, Expected SARSA (Sutton & Barto §6.4–6.6)
// ------------------------------------------------------------------------------------------

type Variant = 'sarsa' | 'qlearning' | 'expected';

export const tdEquation = (boot: EqToken): EqToken[] => [
  { id: 'lhs', tex: 'Q(S,A)' },
  { id: 'arrow', tex: '\\leftarrow' },
  { id: 'old', tex: 'Q(S,A)' },
  { id: 'plus', tex: '+' },
  { id: 'alpha', tex: '\\alpha', param: 'alpha' },
  { id: 'lb', tex: '\\Big[' },
  { id: 'R', tex: 'R' },
  { id: 'plus2', tex: '+' },
  { id: 'gamma', tex: '\\gamma', param: 'gamma' },
  boot,
  { id: 'minus', tex: '-' },
  { id: 'old2', tex: 'Q(S,A)' },
  { id: 'rb', tex: '\\Big]' },
];

export const BOOT_SARSA: EqToken = { id: 'boot-sarsa', tex: "Q(S',A')" };
export const BOOT_MAX: EqToken = { id: 'boot-max', tex: "\\max_{a} Q(S',a)" };
export const BOOT_EXP: EqToken = { id: 'boot-exp', tex: "\\textstyle\\sum_{a}\\pi(a|S')\\,Q(S',a)" };
export const POLICY_TOKEN: EqToken = { id: 'policy', tex: '\\quad A\\sim\\varepsilon\\text{-greedy}(Q)', param: 'epsilon' };

const L = (id: string, text: string, indent: number, params?: string[]): PseudoLine => ({ id, text, indent, params });

export const sarsaPseudo: PseudoLine[] = [
  L('init', 'Initialize Q(s,a) = Q₀ for all s, a', 0, ['q0']),
  L('loop-ep', 'Loop for each episode:', 0),
  L('first', 'S ← start;  A ← ε-greedy(Q, S)', 1, ['epsilon']),
  L('loop-step', 'Loop for each step of episode:', 1),
  L('act', 'Take action A, observe R, S′', 2),
  L('choose-next', 'A′ ← ε-greedy(Q, S′)', 2, ['epsilon']),
  L('update', 'Q(S,A) ← Q(S,A) + α[R + γ·Q(S′,A′) − Q(S,A)]', 2, ['alpha', 'gamma']),
  L('advance', 'S ← S′;  A ← A′', 2),
  L('until', 'until S is terminal', 1),
];

export const qPseudo: PseudoLine[] = [
  L('init', 'Initialize Q(s,a) = Q₀ for all s, a', 0, ['q0']),
  L('loop-ep', 'Loop for each episode:', 0),
  L('first-q', 'S ← start', 1),
  L('loop-step', 'Loop for each step of episode:', 1),
  L('choose', 'A ← ε-greedy(Q, S)', 2, ['epsilon']),
  L('act', 'Take action A, observe R, S′', 2),
  L('update', 'Q(S,A) ← Q(S,A) + α[R + γ·maxₐ Q(S′,a) − Q(S,A)]', 2, ['alpha', 'gamma']),
  L('advance-q', 'S ← S′', 2),
  L('until', 'until S is terminal', 1),
];

export const expectedPseudo: PseudoLine[] = qPseudo.map((l) =>
  l.id === 'update' ? { ...l, text: 'Q(S,A) ← Q(S,A) + α[R + γ·Σₐ π(a|S′)·Q(S′,a) − Q(S,A)]' } : l,
);

const chain: BackupSpec['nodes'] = [
  { id: 's0', kind: 'state', x: 50, y: 10 },
  { id: 'a0', kind: 'action', x: 50, y: 34 },
  { id: 's1', kind: 'state', x: 50, y: 58 },
];
export const sarsaBackup: BackupSpec = {
  nodes: [...chain, { id: 'a1', kind: 'action', x: 50, y: 84 }],
  edges: [['s0', 'a0'], ['a0', 's1'], ['s1', 'a1']],
  caption: 'One real step, then bootstrap from the next action actually chosen.',
};
const fan: BackupSpec['nodes'] = [
  ...chain,
  { id: 'a1L', kind: 'action', x: 30, y: 84 },
  { id: 'a1', kind: 'action', x: 50, y: 84 },
  { id: 'a1R', kind: 'action', x: 70, y: 84 },
];
const fanEdges: [string, string][] = [['s0', 'a0'], ['a0', 's1'], ['s1', 'a1L'], ['s1', 'a1'], ['s1', 'a1R']];
export const qBackup: BackupSpec = {
  nodes: fan,
  edges: fanEdges,
  arcs: [{ id: 'arc-next', over: ['a1L', 'a1', 'a1R'], label: 'max' }],
  caption: 'One real step, then bootstrap from the best next action, whatever is actually done.',
};
export const expectedBackup: BackupSpec = {
  nodes: fan,
  edges: fanEdges,
  arcs: [{ id: 'arc-next', over: ['a1L', 'a1', 'a1R'], label: 'Σπ' }],
  caption: 'One real step, then bootstrap from the average over next actions, weighted by the policy.',
};

export class TDAgent implements GridAgent {
  Q: Float32Array;
  V: Float32Array;
  lastExplore = false;
  private p: ParamValues = {};
  private eps = 0.1;
  private pending = -1;

  constructor(S: number, private readonly variant: Variant, params: ParamValues, private readonly rng: Rng) {
    this.Q = new Float32Array(S * ACTIONS);
    this.V = new Float32Array(S);
    this.setParams(params);
    this.Q.fill(num(params, 'q0', 0));
    for (let s = 0; s < S; s++) this.V[s] = rowMax(this.Q, s * ACTIONS, ACTIONS);
  }

  setParams(p: ParamValues) {
    this.p = p;
  }

  beginEpisode(ep: number) {
    this.eps = scheduledEps(this.p, ep);
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
    const Q = this.Q;
    const off2 = tr.next * ACTIONS;
    let boot = 0;
    if (!tr.done) {
      if (this.variant === 'sarsa') {
        // choose A' now (before the update, as in Sutton & Barto's box) and remember it
        const r = epsGreedy(Q, off2, ACTIONS, this.eps, this.rng);
        this.pending = r.a;
        this.lastExplore = r.explore;
        boot = Q[off2 + r.a];
      } else if (this.variant === 'qlearning') {
        boot = rowMax(Q, off2, ACTIONS);
      } else {
        // expected value under the ε-greedy policy (ties share the greedy mass)
        const m = rowMax(Q, off2, ACTIONS);
        let nBest = 0;
        for (let b = 0; b < ACTIONS; b++) if (Q[off2 + b] >= m - 1e-9) nBest++;
        for (let b = 0; b < ACTIONS; b++) {
          const greedy = Q[off2 + b] >= m - 1e-9 ? (1 - this.eps) / nBest : 0;
          boot += (greedy + this.eps / ACTIONS) * Q[off2 + b];
        }
      }
    }
    const i = s * ACTIONS + a;
    const target = tr.reward + gamma * boot;
    const delta = target - Q[i];
    Q[i] += alpha * delta;
    this.V[s] = rowMax(Q, s * ACTIONS, ACTIONS);
    return { line: 'update', s, target, delta };
  }

  endEpisode() {
    this.pending = -1;
  }

  tables(): Tables {
    return { Q: this.Q, V: this.V };
  }
}

const tdParams = (alpha = 0.5) => [alphaParam(alpha), gammaParam(0.9), epsParam(0.1), epsDecayParam, q0Param];

export const sarsaMeta: AlgoMeta = {
  id: 'sarsa',
  name: 'SARSA',
  short: 'SARSA',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: tdParams(),
  equation: [...tdEquation(BOOT_SARSA), POLICY_TOKEN],
  pseudocode: sarsaPseudo,
  backup: sarsaBackup,
  summary:
    'On-policy TD control. After every step SARSA nudges Q(S,A) toward the reward plus the value of the action it will actually take next, exploration included.',
  watchFor: [
    'On Cliff Walk SARSA learns a path a row or two away from the edge. Its own random moves make the edge dangerous, and it knows that.',
    'Each update touches only the cell the agent just left (the glowing cell).',
  ],
  reference: 'Sutton & Barto §6.4',
  traits: {
    target: 'the value of the next action it actually takes (exploration included)',
    onPolicy: true,
    model: 'none',
    timing: 'after every step',
    learns: 'action values Q(s,a)',
  },
};

export const qLearningMeta: AlgoMeta = {
  id: 'qlearning',
  name: 'Q-learning',
  short: 'Q-LEARN',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: tdParams(),
  equation: [...tdEquation(BOOT_MAX), POLICY_TOKEN],
  pseudocode: qPseudo,
  backup: qBackup,
  summary:
    'Off-policy TD control. It behaves ε-greedily but learns the value of acting greedily, by bootstrapping from the best next action.',
  watchFor: [
    'On Cliff Walk the greedy arrows hug the cliff edge (the optimal path), yet the agent still falls in often while exploring.',
    'Compare its learning curve with SARSA: better final policy, worse return while learning.',
  ],
  reference: 'Sutton & Barto §6.5',
  traits: {
    target: 'the value of the best next action, whatever it actually does next',
    onPolicy: false,
    model: 'none',
    timing: 'after every step',
    learns: 'action values Q(s,a)',
  },
};

export const expectedSarsaMeta: AlgoMeta = {
  id: 'expected-sarsa',
  name: 'Expected SARSA',
  short: 'EXP-SARSA',
  family: 'tabular',
  envKind: 'grid',
  needsModel: false,
  params: tdParams(0.5),
  equation: [...tdEquation(BOOT_EXP), POLICY_TOKEN],
  pseudocode: expectedPseudo,
  backup: expectedBackup,
  summary:
    'Like SARSA, but instead of the one next action it samples, it averages over all next actions weighted by the ε-greedy policy. That removes the noise of the action sample.',
  watchFor: [
    'Handles large step sizes better than SARSA. Try α = 1 on Cliff Walk with both.',
    'With ε = 0 it becomes exactly Q-learning.',
  ],
  reference: 'Sutton & Barto §6.6',
  traits: {
    target: 'the expected value of the next action under its ε-greedy policy',
    onPolicy: true,
    model: 'none',
    timing: 'after every step',
    learns: 'action values Q(s,a)',
  },
};
