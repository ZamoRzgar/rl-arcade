import type { BanditConfig } from '../types';
import { normalAt, Rng, uniformAt } from '../rng';

/**
 * k-armed bandit (Sutton & Barto ch. 2).
 * Gaussian: q*(a) ~ N(0, 1), reward ~ N(q*(a), 1)  (the 10-armed testbed).
 * Bernoulli: p(a) ~ U(0.05, 0.95), reward ∈ {0, 1}.
 * Rewards are a pure function of (seed, t, arm), so two algorithms that pull the same arm at the
 * same step get the same reward (common random numbers).
 */
export class BanditEnv {
  readonly means: Float32Array;
  readonly optimal: number;
  constructor(readonly cfg: BanditConfig, private readonly seed: number) {
    const rng = new Rng(seed).fork('env').fork('means');
    this.means = new Float32Array(cfg.k);
    for (let a = 0; a < cfg.k; a++) {
      this.means[a] = cfg.dist === 'gaussian' ? rng.normal() : 0.05 + 0.9 * rng.next();
    }
    let best = 0;
    for (let a = 1; a < cfg.k; a++) if (this.means[a] > this.means[best]) best = a;
    this.optimal = best;
  }

  reward(t: number, arm: number): number {
    if (this.cfg.dist === 'gaussian') return this.means[arm] + normalAt(this.seed, 'r', t, arm);
    return uniformAt(this.seed, 'r', t, arm) < this.means[arm] ? 1 : 0;
  }
}
