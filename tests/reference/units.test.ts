import { describe, expect, it } from 'vitest';
import { Rng, hash32 } from '../../src/core/rng';
import { GridEnv } from '../../src/core/envs/gridworld';
import { fromAscii, LEVELS } from '../../src/core/envs/presets';
import { TDAgent } from '../../src/core/agents/tabular/td';
import { convergenceEpisode, movingAverage } from '../../src/core/metrics';
import { ALGORITHMS } from '../../src/core/registry';
import { runExperiment } from '../../src/core/engine';
import { PLAIN } from '../../src/core/plain';
import type { Experiment } from '../../src/core/types';

describe('rng', () => {
  it('is deterministic and forks independently of consumption', () => {
    const a = new Rng(123);
    const b = new Rng(123);
    for (let i = 0; i < 100; i++) expect(a.next()).toBe(b.next());
    const c = new Rng(123);
    const forkBefore = c.fork('x').next();
    for (let i = 0; i < 50; i++) c.next();
    expect(c.fork('x').next()).toBe(forkBefore);
    expect(new Rng(123).fork('x').next()).not.toBe(new Rng(123).fork('y').next());
  });
  it('has sane moments', () => {
    const r = new Rng(9);
    let s = 0;
    let s2 = 0;
    const n = 20000;
    for (let i = 0; i < n; i++) {
      const x = r.normal();
      s += x;
      s2 += x * x;
    }
    expect(Math.abs(s / n)).toBeLessThan(0.03);
    expect(Math.abs(s2 / n - 1)).toBeLessThan(0.05);
    let bsum = 0;
    for (let i = 0; i < 5000; i++) bsum += r.beta(2, 6);
    expect(Math.abs(bsum / 5000 - 0.25)).toBeLessThan(0.01);
  });
  it('hash32 differs for different keys', () => {
    expect(hash32(1, 'a')).not.toBe(hash32(1, 'b'));
  });
});

describe('grid world', () => {
  const env = new GridEnv(fromAscii(['S.#', '.LG', 'C..'], { stepReward: -1, bumpReward: -2, goalReward: 5, lavaReward: -10 }));
  it('blocks walls and edges with bump reward', () => {
    expect(env.move(0, 0)).toEqual({ next: 0, reward: -2, done: false });
    expect(env.move(1, 1)).toEqual({ next: 1, reward: -2, done: false });
  });
  it('ends on goal and lava with their rewards', () => {
    expect(env.move(1, 2)).toEqual({ next: 4, reward: -11, done: true });
    expect(env.move(8, 0)).toEqual({ next: 5, reward: 4, done: true });
  });
  it('cliff sends back to start', () => {
    expect(env.move(3, 2)).toEqual({ next: 0, reward: -100, done: false });
  });
  it('model probabilities sum to 1, with slip', () => {
    const slippy = new GridEnv({ ...fromAscii(['S..', '...', '..G']), slip: 0.3 });
    for (const s of slippy.live) for (let a = 0; a < 4; a++) {
      const total = slippy.model(s, a).reduce((x, o) => x + o.p, 0);
      expect(total).toBeCloseTo(1, 6);
    }
  });
  it('wind pushes from the cell being left', () => {
    const windy = new GridEnv(fromAscii(['...', '.w.', '...']));
    expect(windy.move(4, 1).next).toBe(2); // right from centre, then pushed up
  });
});

describe('one hand-computed update', () => {
  it('Q-learning: Q ← Q + α(r + γ max Q′ − Q)', () => {
    const ag = new TDAgent(2, 'qlearning', { alpha: 0.5, gamma: 0.9, epsilon: 0, q0: 0 }, new Rng(1));
    ag.Q.set([1, 2, 3, 4, 0, 10, 0, 0]);
    ag.beginEpisode(0);
    const info = ag.update(0, 1, { next: 1, reward: -1, done: false });
    // target = −1 + 0.9·10 = 8; Q = 2 + 0.5·(8 − 2) = 5
    expect(info.target).toBeCloseTo(8, 6);
    expect(ag.Q[1]).toBeCloseTo(5, 6);
  });
});

describe('metrics', () => {
  it('moving average and convergence', () => {
    expect(Array.from(movingAverage([2, 4, 6], 2))).toEqual([2, 3, 5]);
    const curve = Array.from({ length: 200 }, (_, i) => (i < 50 ? -100 + i * 2 : -13));
    const c = convergenceEpisode(curve);
    expect(c).toBeGreaterThan(40);
    expect(c).toBeLessThan(70);
  });
});

describe('plain-language explanations', () => {
  it('exist for every registered algorithm', () => {
    for (const def of ALGORITHMS) {
      const p = PLAIN[def.meta.id];
      expect(p, def.meta.id).toBeDefined();
      expect(p.simple.length).toBeGreaterThan(20);
      expect(p.formula.length).toBeGreaterThan(20);
    }
  });
});

describe('every algorithm runs on every compatible level', () => {
  for (const level of LEVELS) {
    for (const def of ALGORITHMS) {
      if (def.meta.envKind !== level.env.kind) continue;
      it(`${def.meta.id} on ${level.id}`, () => {
        const exp: Experiment = {
          env: level.env.kind === 'bandit' ? { kind: 'bandit', bandit: { ...level.env.bandit, steps: 200 } } : level.env,
          algoId: def.meta.id,
          params: {},
          run: { ...level.run, episodes: 15, avgSeeds: 2 },
          tweaks: [{ at: 5, params: {} }],
        };
        const r = runExperiment(exp);
        expect(Number.isFinite(r.score)).toBe(true);
        if (r.kind === 'grid') {
          for (let i = 0; i < r.snapV.length; i++) expect(Number.isFinite(r.snapV[i])).toBe(true);
        }
      });
    }
  }
});
