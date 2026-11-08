import type { Experiment } from '../core/types';
import type { RunResult } from '../core/engine';
import type { SweepPoint, WorkerRequest, WorkerResponse } from './protocol';

/**
 * Promise wrapper around the simulation worker. Two workers are used so the A and B slots (or the
 * current run and a sweep) compute in parallel. Results are cached by experiment hash.
 */

type Pending = {
  resolve: (v: unknown) => void;
  reject: (e: Error) => void;
  onProgress?: (done: number, total: number) => void;
};

type DistOmit<T, K extends PropertyKey> = T extends unknown ? Omit<T, K> : never;

class WorkerHandle {
  private w: Worker;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  constructor() {
    this.w = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    this.w.onmessage = (ev: MessageEvent<WorkerResponse>) => {
      const msg = ev.data;
      const p = this.pending.get(msg.id);
      if (!p) return;
      if (msg.type === 'sweep-progress') {
        p.onProgress?.(msg.done, msg.total);
        return;
      }
      this.pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else if (msg.type === 'run-result') p.resolve(msg.result);
      else p.resolve(msg.points);
    };
  }
  request<T>(req: DistOmit<WorkerRequest, 'id'>, onProgress?: Pending['onProgress']): Promise<T> {
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve: resolve as (v: unknown) => void, reject, onProgress });
      this.w.postMessage({ ...req, id } as WorkerRequest);
    });
  }
}

let workers: WorkerHandle[] | null = null;
const pool = () => (workers ??= [new WorkerHandle(), new WorkerHandle(), new WorkerHandle()]);

/** Stable JSON (sorted keys) used as the cache key of an experiment. */
export function expHash(exp: Experiment): string {
  const norm = (v: unknown): unknown => {
    if (Array.isArray(v)) return v.map(norm);
    if (v && typeof v === 'object') {
      const o: Record<string, unknown> = {};
      for (const k of Object.keys(v as object).sort()) {
        const x = (v as Record<string, unknown>)[k];
        if (x !== undefined) o[k] = norm(x);
      }
      return o;
    }
    return v;
  };
  return JSON.stringify(norm(exp));
}

const cache = new Map<string, Promise<RunResult>>();
const CACHE_MAX = 24;

export function runExp(exp: Experiment, slot: 0 | 1): Promise<RunResult> {
  const key = expHash(exp);
  const hit = cache.get(key);
  if (hit) {
    cache.delete(key);
    cache.set(key, hit);
    return hit;
  }
  const p = pool()[slot].request<RunResult>({ type: 'run', exp });
  cache.set(key, p);
  p.catch(() => cache.delete(key));
  while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value as string);
  return p;
}

export function runSweep(
  exp: Experiment,
  paramId: string,
  values: (number | string | boolean)[],
  seeds: number,
  onProgress: (done: number, total: number) => void,
): Promise<SweepPoint[]> {
  return pool()[2].request<SweepPoint[]>({ type: 'sweep', exp, paramId, values, seeds }, onProgress);
}
