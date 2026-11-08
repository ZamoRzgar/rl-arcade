/** Learning-curve metrics shared by the engine, the diff chips and the tests. */

export function mean(xs: ArrayLike<number>, from = 0, to = xs.length): number {
  let s = 0;
  const n = Math.max(0, to - from);
  for (let i = from; i < to; i++) s += xs[i];
  return n > 0 ? s / n : 0;
}

export function std(xs: ArrayLike<number>, from = 0, to = xs.length): number {
  const m = mean(xs, from, to);
  let s = 0;
  const n = Math.max(0, to - from);
  for (let i = from; i < to; i++) s += (xs[i] - m) ** 2;
  return n > 1 ? Math.sqrt(s / (n - 1)) : 0;
}

/** Trailing moving average with window w. */
export function movingAverage(xs: ArrayLike<number>, w: number): Float32Array {
  const out = new Float32Array(xs.length);
  let acc = 0;
  for (let i = 0; i < xs.length; i++) {
    acc += xs[i];
    if (i >= w) acc -= xs[i - w];
    out[i] = acc / Math.min(i + 1, w);
  }
  return out;
}

/** Score of a run: mean return over the last 10% of episodes (at least one). */
export function finalScore(returns: ArrayLike<number>): number {
  const n = returns.length;
  if (n === 0) return 0;
  const tail = Math.max(1, Math.floor(n * 0.1));
  return mean(returns, n - tail, n);
}

/**
 * Episode at which learning converged: the first episode after which the moving-average return
 * stays within a tolerance band around the final score for the rest of the run.
 * Window w = max(5, n/20). Tolerance = max(1, 10% of |final|, 10% of the total improvement
 * (final − worst moving average), 2·σ_tail/√w). In words: "within 10% of where it ends up, for good".
 * Returns -1 when the run is too short (< 10 episodes) to say.
 */
export function convergenceEpisode(returns: ArrayLike<number>): number {
  const n = returns.length;
  if (n < 10) return -1;
  const w = Math.max(5, Math.floor(n / 20));
  const ma = movingAverage(returns, w);
  const tail = Math.max(1, Math.floor(n * 0.1));
  const final = mean(returns, n - tail, n);
  let worst = Infinity;
  for (let i = 0; i < n; i++) worst = Math.min(worst, ma[i]);
  const tol = Math.max(1, 0.1 * Math.abs(final), 0.1 * Math.abs(final - worst), (2 * std(returns, n - tail, n)) / Math.sqrt(w));
  let e = n - 1;
  while (e > 0 && Math.abs(ma[e - 1] - final) <= tol) e--;
  return e;
}

/** Downsample a curve to at most `points` values by bucket averaging. */
export function downsample(xs: ArrayLike<number>, points: number): Float32Array {
  const n = xs.length;
  if (n <= points) return Float32Array.from(xs as ArrayLike<number>);
  const out = new Float32Array(points);
  for (let i = 0; i < points; i++) {
    const a = Math.floor((i * n) / points);
    const b = Math.max(a + 1, Math.floor(((i + 1) * n) / points));
    out[i] = mean(xs, a, b);
  }
  return out;
}
