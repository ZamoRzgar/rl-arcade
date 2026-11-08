import type { BanditExtraKind, Experiment, GridLayout, ParamValues, Tables } from './types';
import { deriveSeed, Rng } from './rng';
import { ACTIONS, GridEnv } from './envs/gridworld';
import { BanditEnv } from './envs/bandit';
import { getAlgo, resolveParams } from './registry';
import { convergenceEpisode, downsample, finalScore, mean } from './metrics';

/**
 * The engine runs a whole experiment headlessly and records everything the UI needs to play it
 * back: per-episode snapshots of the tables, the full trajectory, and the row of the table updated
 * at every step. Tabular runs are cheap (milliseconds), so the UI can rerun on every slider
 * change, scrub anywhere, and diff two runs at exactly the same (episode, step).
 */

export interface GridRunResult {
  kind: 'grid';
  algoId: string;
  planner: boolean;
  S: number;
  A: number;
  /** episodes actually run (planner: frames) */
  E: number;
  hasQ: boolean;
  hasP: boolean;
  /** (E+1)·S: V at the start of each episode, plus the final one */
  snapV: Float32Array;
  /** (E+1)·S·A, or empty */
  snapQ: Float32Array;
  snapP: Float32Array;
  /** (E+1)·S cumulative visit counts */
  visits: Float32Array;
  returns: Float32Array;
  lengths: Int32Array;
  /** E+1 offsets into the per-step arrays */
  epStart: Int32Array;
  // per step
  stS: Int16Array;
  stA: Int8Array;
  stR: Float32Array;
  stNext: Int16Array;
  stLine: Uint8Array;
  stExplore: Uint8Array;
  stUpdS: Int16Array;
  stTarget: Float32Array;
  stDelta: Float32Array;
  /** per step, the updated row: A floats of Q (if hasQ), 1 float of V, A floats of P (if hasP) */
  stRowQ: Float32Array;
  stRowV: Float32Array;
  stRowP: Float32Array;
  /** per step, up to 4 states touched by planning (-1 = none) */
  stPlan: Int16Array;
  /** planner only: phase (0 eval/sweep, 1 improve) and max |ΔV| of each frame */
  phases: Uint8Array;
  deltas: Float32Array;
  /** ε actually used in each episode (agents with ε) */
  score: number;
  converged: number;
  /** mean return per episode across avgSeeds seeds (equals returns when avgSeeds = 1) */
  avgReturns: Float32Array;
  avgLengths: Float32Array;
  seedsUsed: number;
  elapsedMs: number;
  truncatedSteps: boolean;
}

export interface BanditRunResult {
  kind: 'bandit';
  algoId: string;
  dist: 'gaussian' | 'bernoulli';
  k: number;
  T: number;
  xKind: BanditExtraKind;
  means: Float32Array;
  optimal: number;
  arms: Int8Array;
  rewards: Float32Array;
  explore: Uint8Array;
  lines: Uint8Array;
  /** (T+1)·k */
  snapQ: Float32Array;
  snapN: Float32Array;
  /** (T+1)·k·2 */
  snapX: Float32Array;
  /** averaged over seeds, length T */
  avgReward: Float32Array;
  pctOptimal: Float32Array;
  cumRegret: Float32Array;
  score: number;
  converged: number;
  seedsUsed: number;
  elapsedMs: number;
}

export type RunResult = GridRunResult | BanditRunResult;

/** Lightweight result for sweeps: no trajectories. */
export interface LightResult {
  score: number;
  curve: Float32Array;
  finalV: Float32Array | null;
  finalBest: Uint8Array | null;
  converged: number;
}

export const MAX_TOTAL_STEPS = 1_500_000;

// ------------------------------------------------------------------ growable typed arrays

class Grow<T extends Float32Array | Int16Array | Int8Array | Uint8Array> {
  arr: T;
  n = 0;
  constructor(private readonly make: (n: number) => T, private readonly width: number, cap = 1024) {
    this.arr = make(cap * width);
  }
  reserve(rows: number) {
    const need = (this.n + rows) * this.width;
    if (need <= this.arr.length) return;
    let cap = this.arr.length;
    while (cap < need) cap *= 2;
    const next = this.make(cap);
    next.set(this.arr);
    this.arr = next;
  }
  push(v: number) {
    this.reserve(1);
    this.arr[this.n * this.width] = v;
    this.n++;
  }
  /** append one row of `width` values via a writer */
  row(write: (a: T, off: number) => void) {
    this.reserve(1);
    write(this.arr, this.n * this.width);
    this.n++;
  }
  done(): T {
    return this.arr.slice(0, this.n * this.width) as T;
  }
}

const tweakMap = (exp: Experiment) => {
  const m = new Map<number, ParamValues>();
  for (const t of exp.tweaks) m.set(t.at, { ...(m.get(t.at) ?? {}), ...t.params });
  return m;
};

// ------------------------------------------------------------------ grid

function runGridOnce(exp: Experiment, layout: GridLayout, seed: number, record: boolean): GridRunResult {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const def = getAlgo(exp.algoId);
  const env = new GridEnv(layout);
  const S = env.S;
  const A = ACTIONS;
  const root = new Rng(seed);
  const envRoot = root.fork('env');
  const agentRng = root.fork('agent');
  let params = resolveParams(exp.algoId, exp.params);
  const tweaks = tweakMap(exp);
  const E = Math.max(1, Math.floor(exp.run.episodes));
  const M = Math.max(1, Math.floor(exp.run.maxSteps));
  const lineIndex = new Map(def.meta.pseudocode.map((l, i) => [l.id, i]));

  const isPlanner = !!def.planner;
  let tables: () => Tables;
  let agent = null as ReturnType<NonNullable<typeof def.grid>> | null;
  let planner = null as ReturnType<NonNullable<typeof def.planner>> | null;
  if (isPlanner) {
    planner = def.planner!(env, params);
    tables = () => planner!.tables();
  } else {
    agent = def.grid!(env, params, agentRng);
    tables = () => agent!.tables();
  }
  const t = tables();
  const hasQ = !!t.Q;
  const hasP = !!t.P;
  const snapCount = record ? E + 1 : 2;
  const snapV = new Float32Array(snapCount * S);
  const snapQ = new Float32Array(hasQ ? snapCount * S * A : 0);
  const snapP = new Float32Array(hasP ? snapCount * S * A : 0);
  const visits = new Float32Array(snapCount * S);
  const visitNow = new Float32Array(S);
  const writeSnap = (idx: number) => {
    const tb = tables();
    const slot = record ? idx : Math.min(idx, 1);
    if (tb.V) snapV.set(tb.V, slot * S);
    else if (tb.Q) for (let s = 0; s < S; s++) snapV[slot * S + s] = Math.max(tb.Q[s * A], tb.Q[s * A + 1], tb.Q[s * A + 2], tb.Q[s * A + 3]);
    if (hasQ && tb.Q) snapQ.set(tb.Q, slot * S * A);
    if (hasP && tb.P) snapP.set(tb.P, slot * S * A);
    visits.set(visitNow, slot * S);
  };
  writeSnap(0);

  const returns = new Float32Array(E);
  const lengths = new Int32Array(E);
  const epStart = new Int32Array(E + 1);
  const phases = new Uint8Array(isPlanner ? E : 0);
  const deltas = new Float32Array(isPlanner ? E : 0);

  const gS = new Grow((n) => new Int16Array(n), 1);
  const gA = new Grow((n) => new Int8Array(n), 1);
  const gR = new Grow((n) => new Float32Array(n), 1);
  const gN = new Grow((n) => new Int16Array(n), 1);
  const gLine = new Grow((n) => new Uint8Array(n), 1);
  const gExp = new Grow((n) => new Uint8Array(n), 1);
  const gUpd = new Grow((n) => new Int16Array(n), 1);
  const gTgt = new Grow((n) => new Float32Array(n), 1);
  const gDel = new Grow((n) => new Float32Array(n), 1);
  const gRowQ = new Grow((n) => new Float32Array(n), hasQ ? A : 0);
  const gRowV = new Grow((n) => new Float32Array(n), 1);
  const gRowP = new Grow((n) => new Float32Array(n), hasP ? A : 0);
  const gPlan = new Grow((n) => new Int16Array(n), 4);

  let E_run = E;
  let total = 0;
  let truncatedSteps = false;

  if (isPlanner) {
    let f = 0;
    for (; f < E; f++) {
      const fr = planner!.frame();
      phases[f] = fr.phase;
      deltas[f] = fr.delta;
      writeSnap(f + 1);
      returns[f] = snapV[(record ? f + 1 : 1) * S + env.start];
      lengths[f] = env.S;
      if (fr.done) {
        f++;
        break;
      }
    }
    E_run = f;
  } else {
    const exploringStarts = params.exploringStarts === true;
    for (let e = 0; e < E; e++) {
      const tw = tweaks.get(e);
      if (tw) {
        params = { ...params, ...tw };
        agent!.setParams(params);
      }
      env.prepareEpisode(e);
      const envRng = envRoot.fork(e);
      let s = env.reset(envRng, exploringStarts);
      const forced = exploringStarts ? envRng.int(A) : -1;
      agent!.beginEpisode(e, s);
      epStart[e] = total;
      let ret = 0;
      let steps = 0;
      let done = false;
      for (let k = 0; k < M; k++) {
        let a = agent!.act(s);
        let explore = agent!.lastExplore;
        if (k === 0 && forced >= 0) {
          a = forced;
          explore = true;
        }
        const tr = env.step(s, a, envRng);
        const info = agent!.update(s, a, tr);
        if (record) {
          gS.push(s);
          gA.push(a);
          gR.push(tr.reward);
          gN.push(tr.next);
          gLine.push(lineIndex.get(info?.line ?? '') ?? 255);
          gExp.push(explore ? 1 : 0);
          const us = info && info.s >= 0 ? info.s : -1;
          gUpd.push(us);
          gTgt.push(info?.target ?? NaN);
          gDel.push(info?.delta ?? NaN);
          const tb = tables();
          if (hasQ) gRowQ.row((arr, off) => { for (let b = 0; b < A; b++) arr[off + b] = us >= 0 && tb.Q ? tb.Q[us * A + b] : 0; });
          gRowV.push(us >= 0 ? (tb.V ? tb.V[us] : tb.Q ? Math.max(tb.Q[us * A], tb.Q[us * A + 1], tb.Q[us * A + 2], tb.Q[us * A + 3]) : 0) : 0);
          if (hasP) gRowP.row((arr, off) => { for (let b = 0; b < A; b++) arr[off + b] = us >= 0 && tb.P ? tb.P[us * A + b] : 0; });
          gPlan.row((arr, off) => {
            for (let i = 0; i < 4; i++) arr[off + i] = info?.plan && i < info.plan.length ? info.plan[i] : -1;
          });
        }
        visitNow[s] += 1;
        ret += tr.reward;
        steps++;
        total++;
        s = tr.next;
        if (tr.done) {
          done = true;
          break;
        }
        if (record && total >= MAX_TOTAL_STEPS) break;
      }
      agent!.endEpisode(!done, s);
      returns[e] = ret;
      lengths[e] = steps;
      writeSnap(e + 1);
      if (record && total >= MAX_TOTAL_STEPS) {
        E_run = e + 1;
        truncatedSteps = true;
        break;
      }
    }
    epStart[E_run] = total;
  }

  const ret = returns.slice(0, E_run);
  const len = lengths.slice(0, E_run);
  const elapsed = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  return {
    kind: 'grid',
    algoId: exp.algoId,
    planner: isPlanner,
    S,
    A,
    E: E_run,
    hasQ,
    hasP,
    snapV: record ? snapV.slice(0, (E_run + 1) * S) : snapV,
    snapQ: record ? snapQ.slice(0, hasQ ? (E_run + 1) * S * A : 0) : snapQ,
    snapP: record ? snapP.slice(0, hasP ? (E_run + 1) * S * A : 0) : snapP,
    visits: record ? visits.slice(0, (E_run + 1) * S) : visits,
    returns: ret,
    lengths: len,
    epStart: isPlanner ? new Int32Array(E_run + 1) : epStart.slice(0, E_run + 1),
    stS: gS.done(),
    stA: gA.done(),
    stR: gR.done(),
    stNext: gN.done(),
    stLine: gLine.done(),
    stExplore: gExp.done(),
    stUpdS: gUpd.done(),
    stTarget: gTgt.done(),
    stDelta: gDel.done(),
    stRowQ: gRowQ.done(),
    stRowV: gRowV.done(),
    stRowP: gRowP.done(),
    stPlan: gPlan.done(),
    phases: phases.slice(0, isPlanner ? E_run : 0),
    deltas: deltas.slice(0, isPlanner ? E_run : 0),
    score: isPlanner ? ret[E_run - 1] ?? 0 : finalScore(ret),
    converged: isPlanner ? E_run : convergenceEpisode(ret),
    avgReturns: ret,
    avgLengths: Float32Array.from(len),
    seedsUsed: 1,
    elapsedMs: elapsed,
    truncatedSteps,
  };
}

export function runGrid(exp: Experiment): GridRunResult {
  if (exp.env.kind !== 'grid') throw new Error('runGrid needs a grid experiment');
  const main = runGridOnce(exp, exp.env.layout, exp.run.seed, true);
  const seeds = main.planner ? 1 : Math.max(1, Math.floor(exp.run.avgSeeds));
  if (seeds > 1) {
    const sumR = Float32Array.from(main.returns);
    const sumL = Float32Array.from(main.lengths);
    for (let i = 1; i < seeds; i++) {
      const r = runGridOnce(exp, exp.env.layout, deriveSeed(exp.run.seed, i), false);
      for (let e = 0; e < main.E; e++) {
        sumR[e] += r.returns[e] ?? 0;
        sumL[e] += r.lengths[e] ?? 0;
      }
    }
    for (let e = 0; e < main.E; e++) {
      sumR[e] /= seeds;
      sumL[e] /= seeds;
    }
    main.avgReturns = sumR;
    main.avgLengths = sumL;
    main.seedsUsed = seeds;
    main.score = finalScore(sumR);
    main.converged = convergenceEpisode(sumR);
  }
  return main;
}

// ------------------------------------------------------------------ bandit

export function runBandit(exp: Experiment, record = true): BanditRunResult {
  if (exp.env.kind !== 'bandit') throw new Error('runBandit needs a bandit experiment');
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const cfg = exp.env.bandit;
  const def = getAlgo(exp.algoId);
  const k = cfg.k;
  const T = Math.max(1, Math.floor(cfg.steps));
  const seeds = Math.max(1, Math.floor(exp.run.avgSeeds));
  const lineIndex = new Map(def.meta.pseudocode.map((l, i) => [l.id, i]));
  const tweaks = tweakMap(exp);

  const sumReward = new Float32Array(T);
  const sumOpt = new Float32Array(T);
  const sumRegret = new Float32Array(T);
  let main: Omit<BanditRunResult, 'avgReward' | 'pctOptimal' | 'cumRegret' | 'score' | 'converged' | 'seedsUsed' | 'elapsedMs'> | null = null;

  for (let rep = 0; rep < seeds; rep++) {
    const seed = rep === 0 ? exp.run.seed : deriveSeed(exp.run.seed, rep);
    const env = new BanditEnv(cfg, seed);
    const rng = new Rng(seed).fork('agent');
    let params = resolveParams(exp.algoId, exp.params);
    const agent = def.bandit!(cfg, params, rng);
    const rec = record && rep === 0;
    const arms = new Int8Array(rec ? T : 0);
    const rewards = new Float32Array(rec ? T : 0);
    const explore = new Uint8Array(rec ? T : 0);
    const lines = new Uint8Array(rec ? T : 0);
    const snapQ = new Float32Array(rec ? (T + 1) * k : 0);
    const snapN = new Float32Array(rec ? (T + 1) * k : 0);
    const snapX = new Float32Array(rec ? (T + 1) * k * 2 : 0);
    if (rec) {
      const tb = agent.tables();
      snapQ.set(tb.Q, 0);
      snapN.set(tb.N, 0);
      snapX.set(tb.X, 0);
    }
    let regret = 0;
    const best = env.means[env.optimal];
    for (let t = 0; t < T; t++) {
      const tw = tweaks.get(t);
      if (tw) {
        params = { ...params, ...tw };
        agent.setParams(params);
      }
      const a = agent.act(t);
      const r = env.reward(t, a);
      const line = agent.update(a, r);
      regret += best - env.means[a];
      sumReward[t] += r;
      sumOpt[t] += a === env.optimal ? 1 : 0;
      sumRegret[t] += regret;
      if (rec) {
        arms[t] = a;
        rewards[t] = r;
        explore[t] = agent.lastExplore ? 1 : 0;
        lines[t] = lineIndex.get(line) ?? 255;
        const tb = agent.tables();
        snapQ.set(tb.Q, (t + 1) * k);
        snapN.set(tb.N, (t + 1) * k);
        snapX.set(tb.X, (t + 1) * k * 2);
      }
    }
    if (rep === 0) {
      main = {
        kind: 'bandit',
        algoId: exp.algoId,
        dist: cfg.dist,
        k,
        T,
        xKind: agent.xKind,
        means: env.means,
        optimal: env.optimal,
        arms,
        rewards,
        explore,
        lines,
        snapQ,
        snapN,
        snapX,
      };
    }
  }
  for (let t = 0; t < T; t++) {
    sumReward[t] /= seeds;
    sumOpt[t] /= seeds;
    sumRegret[t] /= seeds;
  }
  const elapsed = (typeof performance !== 'undefined' ? performance.now() : Date.now()) - t0;
  return {
    ...main!,
    avgReward: sumReward,
    pctOptimal: sumOpt,
    cumRegret: sumRegret,
    score: mean(sumReward),
    converged: -1,
    seedsUsed: seeds,
    elapsedMs: elapsed,
  };
}

export function runExperiment(exp: Experiment): RunResult {
  return exp.env.kind === 'grid' ? runGrid(exp) : runBandit(exp);
}

/** Headless summary used by the sensitivity sweep. */
export function runLight(exp: Experiment, seeds: number): LightResult {
  if (exp.env.kind === 'bandit') {
    const r = runBandit({ ...exp, run: { ...exp.run, avgSeeds: Math.max(seeds, Math.min(exp.run.avgSeeds, 50)) } }, false);
    return { score: r.score, curve: downsample(r.avgReward, 48), finalV: null, finalBest: null, converged: -1 };
  }
  const layout = exp.env.layout;
  const n = Math.max(1, seeds);
  let curve: Float32Array | null = null;
  let score = 0;
  let conv = 0;
  let finalV: Float32Array | null = null;
  let finalBest: Uint8Array | null = null;
  for (let i = 0; i < n; i++) {
    const r = runGridOnce(exp, layout, i === 0 ? exp.run.seed : deriveSeed(exp.run.seed, i), false);
    const c = downsample(r.returns, 48);
    if (!curve) curve = c;
    else for (let j = 0; j < c.length; j++) curve[j] += c[j];
    score += r.score;
    conv += r.converged;
    if (i === 0) {
      finalV = r.snapV.slice(r.S, 2 * r.S);
      finalBest = greedyMask(r.hasQ ? r.snapQ.slice(r.S * r.A, 2 * r.S * r.A) : null, r.hasP ? r.snapP.slice(r.S * r.A, 2 * r.S * r.A) : null, r.S);
    }
    if (r.planner) break;
  }
  const used = getAlgo(exp.algoId).planner ? 1 : n;
  if (curve) for (let j = 0; j < curve.length; j++) curve[j] /= used;
  return { score: score / used, curve: curve ?? new Float32Array(0), finalV, finalBest, converged: conv / used };
}

/** Bit mask (bit a set) of greedy actions per state; P takes precedence over Q. */
export function greedyMask(Q: Float32Array | null, P: Float32Array | null, S: number, tol = 1e-6): Uint8Array {
  const out = new Uint8Array(S);
  const src = P ?? Q;
  if (!src) return out;
  for (let s = 0; s < S; s++) {
    const off = s * ACTIONS;
    let m = -Infinity;
    for (let a = 0; a < ACTIONS; a++) m = Math.max(m, src[off + a]);
    let mask = 0;
    for (let a = 0; a < ACTIONS; a++) if (src[off + a] >= m - tol) mask |= 1 << a;
    out[s] = mask;
  }
  return out;
}
