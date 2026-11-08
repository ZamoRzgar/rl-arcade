/// <reference lib="webworker" />
import { runExperiment, runLight } from '../core/engine';
import type { SweepPoint, WorkerRequest, WorkerResponse } from './protocol';
import { transferables } from './protocol';

const post = (msg: WorkerResponse, transfer: ArrayBuffer[] = []) =>
  (self as unknown as { postMessage(m: unknown, o: { transfer: ArrayBuffer[] }): void }).postMessage(msg, { transfer });

self.onmessage = (ev: MessageEvent<WorkerRequest>) => {
  const req = ev.data;
  try {
    if (req.type === 'run') {
      const result = runExperiment(req.exp);
      post({ id: req.id, type: 'run-result', result }, transferables(result));
    } else {
      const points: SweepPoint[] = [];
      req.values.forEach((value, i) => {
        const exp = { ...req.exp, params: { ...req.exp.params, [req.paramId]: value }, tweaks: [] };
        points.push({ value, result: runLight(exp, req.seeds) });
        post({ id: req.id, type: 'sweep-progress', done: i + 1, total: req.values.length });
      });
      post({ id: req.id, type: 'sweep-result', points });
    }
  } catch (e) {
    post({ id: req.id, type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
