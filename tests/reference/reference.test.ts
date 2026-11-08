import { describe, expect, it } from 'vitest';
import { GridEnv, ACTIONS } from '../../src/core/envs/gridworld';
import { evaluatePolicy } from '../../src/core/agents/dp/dp';
import { getLevel } from '../../src/core/envs/presets';
import { runBandit, runGrid, greedyMask } from '../../src/core/engine';
import { mean } from '../../src/core/metrics';
import type { Experiment, GridLayout } from '../../src/core/types';
import { deriveSeed } from '../../src/core/rng';

const gridLayout = (id: string): GridLayout => {
  const env = getLevel(id).env;
  if (env.kind !== 'grid') throw new Error('not grid');
  return env.layout;
};

const gridExp = (layout: GridLayout, algoId: string, params: Record<string, number | string | boolean>, episodes: number, maxSteps: number, seed = 1): Experiment => ({
  env: { kind: 'grid', layout },
  algoId,
  params,
  run: { episodes, maxSteps, seed, avgSeeds: 1 },
  tweaks: [],
});

const topRow = (V: ArrayLike<number>) => Array.from({ length: 5 }, (_, c) => Math.round(V[c] * 10) / 10);

describe('Sutton & Barto Example 3.5 (5×5 grid with teleports, γ = 0.9)', () => {
  const layout = gridLayout('classic-5x5');

  it('equiprobable random policy matches Figure 3.2', () => {
    const env = new GridEnv(layout);
    const P = new Float32Array(env.S * ACTIONS).fill(0.25);
    const V = evaluatePolicy(env, P, 0.9, 1e-8);
    const expected = [3.3, 8.8, 4.4, 5.3, 1.5];
    topRow(V).forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThanOrEqual(0.05 + 1e-9));
    // a couple of other cells from the figure: bottom-left −1.9, centre 0.7
    expect(V[20]).toBeCloseTo(-1.9, 1);
    expect(V[12]).toBeCloseTo(0.7, 1);
  });

  it('value iteration reaches the optimal values of Figure 3.5', () => {
    const r = runGrid(gridExp(layout, 'value-iteration', { gamma: 0.9, theta: 1e-8 }, 2000, 1));
    const V = r.snapV.slice(r.E * r.S, (r.E + 1) * r.S);
    const expected = [22.0, 24.4, 22.0, 19.4, 17.5];
    topRow(V).forEach((v, i) => expect(Math.abs(v - expected[i])).toBeLessThanOrEqual(0.05 + 1e-9));
  });
});

describe('Value iteration and policy iteration agree', () => {
  for (const id of ['classic-5x5', 'cliff-walk', 'ice-rink', 'windy-valley']) {
    it(`on ${id}`, () => {
      const layout = gridLayout(id);
      const vi = runGrid(gridExp(layout, 'value-iteration', { gamma: 0.9, theta: 1e-10 }, 5000, 1));
      const pi = runGrid(gridExp(layout, 'policy-iteration', { gamma: 0.9, theta: 1e-10 }, 50000, 1));
      const S = vi.S;
      const a = vi.snapV.slice(vi.E * S, (vi.E + 1) * S);
      const b = pi.snapV.slice(pi.E * S, (pi.E + 1) * S);
      let maxDiff = 0;
      for (let s = 0; s < S; s++) maxDiff = Math.max(maxDiff, Math.abs(a[s] - b[s]));
      expect(maxDiff).toBeLessThan(1e-4);
    });
  }
});

describe('Cliff walking, Example 6.6 (ε = 0.1, α = 0.5)', () => {
  const layout = gridLayout('cliff-walk');
  const runs = 20;
  const run = (algoId: string, seed: number) => runGrid(gridExp(layout, algoId, { alpha: 0.5, epsilon: 0.1, gamma: 0.999 }, 500, 500, seed));

  it('Q-learning’s greedy policy walks the cliff edge (return −13)', () => {
    let optimal = 0;
    for (let i = 0; i < runs; i++) {
      const r = run('qlearning', deriveSeed(100, i));
      const env = new GridEnv(layout);
      const Q = r.snapQ.slice(r.E * r.S * 4, (r.E + 1) * r.S * 4);
      // follow the greedy policy deterministically
      let s = env.start;
      let ret = 0;
      for (let k = 0; k < 100; k++) {
        const off = s * 4;
        let a = 0;
        for (let b = 1; b < 4; b++) if (Q[off + b] > Q[off + a]) a = b;
        const tr = env.move(s, a);
        ret += tr.reward;
        s = tr.next;
        if (tr.done) break;
      }
      if (ret === -13) optimal++;
    }
    expect(optimal).toBeGreaterThanOrEqual(runs * 0.9);
  });

  it('SARSA earns more online return than Q-learning (episodes 400–500)', () => {
    let sarsa = 0;
    let q = 0;
    for (let i = 0; i < runs; i++) {
      sarsa += mean(run('sarsa', deriveSeed(200, i)).returns, 400, 500);
      q += mean(run('qlearning', deriveSeed(200, i)).returns, 400, 500);
    }
    expect(sarsa / runs).toBeGreaterThan(q / runs);
  });
});

describe('10-armed testbed (Figures 2.2 and 2.4)', () => {
  const exp = (algoId: string, params: Record<string, number>): Experiment => ({
    env: { kind: 'bandit', bandit: { k: 10, dist: 'gaussian', steps: 1000 } },
    algoId,
    params,
    run: { episodes: 1, maxSteps: 1000, seed: 42, avgSeeds: 500 },
    tweaks: [],
  });

  it('UCB (c = 2) beats ε-greedy (ε = 0.1) late in the run', () => {
    const ucb = runBandit(exp('ucb', { c: 2 }), false);
    const eg = runBandit(exp('eps-greedy', { epsilon: 0.1 }), false);
    expect(mean(ucb.avgReward, 900, 1000)).toBeGreaterThan(mean(eg.avgReward, 900, 1000));
  });

  it('ε = 0.1 finds the optimal arm more often than greedy (ε = 0)', () => {
    const eg = runBandit(exp('eps-greedy', { epsilon: 0.1 }), false);
    const greedy = runBandit(exp('eps-greedy', { epsilon: 0 }), false);
    expect(mean(eg.pctOptimal, 900, 1000)).toBeGreaterThan(mean(greedy.pctOptimal, 900, 1000));
  });

  it('Thompson sampling (Bernoulli) has sublinear regret', () => {
    const r = runBandit(
      {
        env: { kind: 'bandit', bandit: { k: 5, dist: 'bernoulli', steps: 10000 } },
        algoId: 'thompson',
        params: { prior: 1 },
        run: { episodes: 1, maxSteps: 1, seed: 9, avgSeeds: 20 },
        tweaks: [],
      },
      false,
    );
    expect(r.cumRegret[9999] / r.cumRegret[999]).toBeLessThan(5);
  });
});

describe('Dyna-Q planning speeds up learning (Example 8.1 idea)', () => {
  it('n = 50 reaches a short path in fewer episodes than n = 0', () => {
    const base = gridLayout('shifting-maze');
    const layout = { ...base, events: undefined };
    const firstGood = (planning: number) => {
      let total = 0;
      for (let i = 0; i < 5; i++) {
        const r = runGrid(gridExp(layout, 'dyna-q', { planning, alpha: 0.1, epsilon: 0.1, gamma: 0.95 }, 60, 2000, deriveSeed(7, i)));
        let e = 0;
        while (e < r.E && r.lengths[e] > 20) e++;
        total += e;
      }
      return total / 5;
    };
    expect(firstGood(50)).toBeLessThan(firstGood(0));
  });
});

describe('n-step returns carry credit further (Long Hallway)', () => {
  it('value reaches the start sooner with n = 8 than n = 1', () => {
    const layout = gridLayout('long-hallway');
    const firstPositive = (n: number) => {
      let total = 0;
      for (let i = 0; i < 5; i++) {
        const r = runGrid(gridExp(layout, 'nstep-sarsa', { n, alpha: 0.5, epsilon: 0.1, gamma: 0.95 }, 60, 400, deriveSeed(3, i)));
        let e = 0;
        while (e < r.E && r.snapV[(e + 1) * r.S + 0] <= 1e-6) e++;
        total += e;
      }
      return total / 5;
    };
    expect(firstPositive(8)).toBeLessThan(firstPositive(1));
  });
});

describe('Determinism and common random numbers', () => {
  const layout = gridLayout('ice-rink');
  it('same experiment twice gives identical results', () => {
    const a = runGrid(gridExp(layout, 'qlearning', { alpha: 0.2 }, 50, 100, 5));
    const b = runGrid(gridExp(layout, 'qlearning', { alpha: 0.2 }, 50, 100, 5));
    expect(Array.from(a.snapQ)).toEqual(Array.from(b.snapQ));
    expect(Array.from(a.stS)).toEqual(Array.from(b.stS));
  });
  it('switching algorithm keeps per-episode start states (random starts)', () => {
    const L = { ...gridLayout('classic-5x5') };
    const a = runGrid(gridExp(L, 'qlearning', {}, 40, 30, 8));
    const b = runGrid(gridExp(L, 'sarsa', {}, 40, 30, 8));
    for (let e = 0; e < 40; e++) expect(a.stS[a.epStart[e]]).toBe(b.stS[b.epStart[e]]);
  });
  it('greedy mask reports ties', () => {
    const Q = new Float32Array([1, 1, 0, 0]);
    expect(greedyMask(Q, null, 1)[0]).toBe(0b0011);
  });
});
