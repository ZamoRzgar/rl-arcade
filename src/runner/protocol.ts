import type { Experiment } from '../core/types';
import type { LightResult, RunResult } from '../core/engine';

export type WorkerRequest =
  | { id: number; type: 'run'; exp: Experiment }
  | { id: number; type: 'sweep'; exp: Experiment; paramId: string; values: (number | string | boolean)[]; seeds: number };

export interface SweepPoint {
  value: number | string | boolean;
  result: LightResult;
}

export type WorkerResponse =
  | { id: number; type: 'run-result'; result: RunResult }
  | { id: number; type: 'sweep-progress'; done: number; total: number }
  | { id: number; type: 'sweep-result'; points: SweepPoint[] }
  | { id: number; type: 'error'; message: string };

/** Collect the ArrayBuffers of all typed arrays in an object (one level deep) for zero-copy transfer. */
export function transferables(obj: object): ArrayBuffer[] {
  const out = new Set<ArrayBuffer>();
  for (const v of Object.values(obj)) {
    if (ArrayBuffer.isView(v) && v.buffer instanceof ArrayBuffer) out.add(v.buffer);
  }
  return [...out];
}
