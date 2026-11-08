/**
 * Seeded randomness.
 *
 * - `Rng` is xoshiro128** seeded through splitmix32. It is fast, has good statistical quality,
 *   and is fully deterministic across browsers because it only uses 32-bit integer math.
 * - `fork(label)` derives an independent child stream from the stream's *seed*, not from its
 *   current position. Forks are therefore stable no matter how much the parent was consumed,
 *   which is what makes "common random numbers" between runs possible.
 * - `uniformAt` / `normalAt` are stateless hash-based draws used where a value must depend only
 *   on its coordinates (for example: bandit reward for arm a at step t).
 */

function fmix32(h: number): number {
  h ^= h >>> 16;
  h = Math.imul(h, 0x85ebca6b);
  h ^= h >>> 13;
  h = Math.imul(h, 0xc2b2ae35);
  h ^= h >>> 16;
  return h >>> 0;
}

/** FNV-1a over the keys, finished with murmur3's fmix32. Integers are hashed by their 4 bytes. */
export function hash32(...keys: (number | string)[]): number {
  let h = 0x811c9dc5;
  for (const k of keys) {
    if (typeof k === 'string') {
      for (let i = 0; i < k.length; i++) {
        h ^= k.charCodeAt(i);
        h = Math.imul(h, 0x01000193);
      }
      h ^= 0xff;
      h = Math.imul(h, 0x01000193);
    } else {
      let x = k >>> 0;
      for (let b = 0; b < 4; b++) {
        h ^= x & 0xff;
        h = Math.imul(h, 0x01000193);
        x >>>= 8;
      }
    }
  }
  return fmix32(h >>> 0);
}

/** Uniform in [0, 1) that depends only on the keys. */
export function uniformAt(...keys: (number | string)[]): number {
  return hash32(...keys) / 4294967296;
}

/** Standard normal that depends only on the keys (Box–Muller on two hashed uniforms). */
export function normalAt(...keys: (number | string)[]): number {
  const u1 = 1 - uniformAt(...keys, 'u1');
  const u2 = uniformAt(...keys, 'u2');
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

/** Deterministic seed for the i-th repetition of an experiment (used for multi-seed averaging). */
export function deriveSeed(seed: number, rep: number): number {
  return hash32(seed, 'rep', rep);
}

const rotl = (x: number, k: number) => (x << k) | (x >>> (32 - k));

export class Rng {
  readonly id: number;
  private s0: number;
  private s1: number;
  private s2: number;
  private s3: number;

  constructor(seed: number) {
    this.id = seed >>> 0;
    let s = this.id;
    const sm = () => {
      s = (s + 0x9e3779b9) | 0;
      let z = s;
      z = Math.imul(z ^ (z >>> 16), 0x85ebca6b);
      z = Math.imul(z ^ (z >>> 13), 0xc2b2ae35);
      return (z ^ (z >>> 16)) >>> 0;
    };
    this.s0 = sm();
    this.s1 = sm();
    this.s2 = sm();
    this.s3 = sm();
    if ((this.s0 | this.s1 | this.s2 | this.s3) === 0) this.s0 = 1;
  }

  nextU32(): number {
    const result = Math.imul(rotl(Math.imul(this.s1, 5), 7), 9) >>> 0;
    const t = this.s1 << 9;
    this.s2 ^= this.s0;
    this.s3 ^= this.s1;
    this.s1 ^= this.s2;
    this.s0 ^= this.s3;
    this.s2 ^= t;
    this.s3 = rotl(this.s3, 11);
    return result;
  }

  /** Uniform in [0, 1). */
  next(): number {
    return this.nextU32() / 4294967296;
  }

  /** Uniform integer in 0..n-1. */
  int(n: number): number {
    return Math.min(n - 1, Math.floor(this.next() * n));
  }

  /** Standard normal (Box–Muller, no caching so the stream stays simple to reason about). */
  normal(): number {
    const u1 = 1 - this.next();
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /** Gamma(shape, 1) via Marsaglia–Tsang. */
  gamma(shape: number): number {
    if (shape < 1) {
      const u = 1 - this.next();
      return this.gamma(shape + 1) * Math.pow(u, 1 / shape);
    }
    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);
    for (;;) {
      let x: number;
      let v: number;
      do {
        x = this.normal();
        v = 1 + c * x;
      } while (v <= 0);
      v = v * v * v;
      const u = 1 - this.next();
      if (u < 1 - 0.0331 * x * x * x * x) return d * v;
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
    }
  }

  beta(a: number, b: number): number {
    const x = this.gamma(a);
    const y = this.gamma(b);
    return x / (x + y);
  }

  /** Independent, deterministic child stream. Depends only on this stream's seed and the label. */
  fork(label: string | number): Rng {
    return new Rng(hash32(this.id, label));
  }
}
