import { create } from 'zustand';
import type { Cell, Experiment, GridLayout, ParamValue, ParamValues, RunSettings } from '../core/types';
import type { RunResult } from '../core/engine';
import { getLevel, LEVELS, type Level } from '../core/envs/presets';
import { algoMeta, resolveParams } from '../core/registry';
import { resizeLayout } from '../core/envs/gridworld';
import { expHash, runExp, runSweep } from '../runner/client';
import type { SweepPoint } from '../runner/protocol';
import { epLength } from '../viz/frame';

export type Mode = 'solo' | 'versus';
export type SlotIndex = 0 | 1;
export type OverlayId = 'values' | 'q' | 'arrows' | 'visits' | 'trail' | 'ghost' | 'diff' | 'numbers';
export type Tool = Cell['t'] | 'erase';

export interface Slot {
  exp: Experiment;
  res: RunResult | null;
  status: 'idle' | 'running' | 'error';
  error?: string;
}

export interface HistoryEntry {
  id: number;
  hash: string;
  levelId: string;
  algoId: string;
  exp: Experiment;
  score: number;
  converged: number;
  changes: string[];
  isBandit: boolean;
}

export interface SweepState {
  slot: SlotIndex;
  paramId: string;
  status: 'running' | 'done' | 'error';
  done: number;
  total: number;
  points: SweepPoint[];
  message?: string;
}

export const SPEEDS = [
  { label: '½×', sps: 2 },
  { label: '1×', sps: 5 },
  { label: '2×', sps: 12 },
  { label: '4×', sps: 30 },
  { label: '8×', sps: 90 },
  { label: 'TURBO', eps: 1 },
  { label: 'MAX', eps: 6 },
] as const;

interface State {
  levelId: string;
  mode: Mode;
  slots: [Slot, Slot];
  editSlot: SlotIndex;
  baseline: { exp: Experiment; res: RunResult } | null;
  pinned: boolean;
  /** algorithm before the latest switch in solo mode (drives the equation/pseudocode diff) */
  prevAlgoId: string | null;
  ep: number;
  t: number;
  playing: boolean;
  speed: number;
  overlays: Record<OverlayId, boolean>;
  changeMode: 'rerun' | 'live';
  hoverParam: string | null;
  tool: Tool | null;
  windDir: number;
  teleportFrom: number | null;
  undo: GridLayout[];
  redo: GridLayout[];
  history: HistoryEntry[];
  sweep: SweepState | null;
  sound: boolean;
  crt: boolean;
  palette: 'neon' | 'safe';
  showTruth: boolean;
  help: boolean;
  banner: string | null;
  toast: string | null;
}

interface Actions {
  loadLevel(id: string): void;
  setMode(m: Mode): void;
  setEditSlot(s: SlotIndex): void;
  setAlgo(slot: SlotIndex, algoId: string): void;
  setParam(slot: SlotIndex, id: string, v: ParamValue): void;
  resetParam(slot: SlotIndex, id: string): void;
  clearTweaks(slot: SlotIndex): void;
  setRun(patch: Partial<RunSettings>): void;
  setLayout(next: GridLayout, pushUndo?: boolean): void;
  paintCell(i: number): void;
  resize(w: number, h: number): void;
  undoEdit(): void;
  redoEdit(): void;
  setBanditSteps(n: number): void;
  setBanditK(n: number): void;
  setBanditDist(d: 'gaussian' | 'bernoulli'): void;
  togglePin(): void;
  setPlayhead(ep: number, t: number): void;
  setPlaying(p: boolean): void;
  setSpeed(i: number): void;
  stepBy(n: number): void;
  stepEpisode(dir: 1 | -1): void;
  restart(): void;
  toggleOverlay(id: OverlayId): void;
  set(p: Partial<State>): void;
  restoreHistory(id: number, asB?: boolean): void;
  startSweep(slot: SlotIndex, paramId: string): void;
  loadShared(level: string, mode: Mode, a: Experiment, b: Experiment | null): void;
  flash(msg: string): void;
}

export type Store = State & Actions;

// --------------------------------------------------------------------------- helpers

const expFromLevel = (level: Level, which: 'A' | 'B'): Experiment => {
  const preset = level[which];
  return {
    env: structuredClone(level.env),
    algoId: preset.algoId,
    params: resolveParams(preset.algoId, preset.params),
    run: { ...level.run },
    tweaks: [],
  };
};

const slot = (exp: Experiment): Slot => ({ exp, res: null, status: 'idle' });

/** Values the user sees for an experiment: params with all live tweaks applied. */
export function effectiveParams(exp: Experiment): ParamValues {
  let p = { ...exp.params };
  for (const t of exp.tweaks) p = { ...p, ...t.params };
  return p;
}

/** Keep shared params (γ, α, ε, …) when switching algorithm so the comparison stays fair. */
function carryParams(fromAlgo: string, from: ParamValues, toAlgo: string): ParamValues {
  const toMeta = algoMeta(toAlgo);
  const fromMeta = algoMeta(fromAlgo);
  const carried: ParamValues = {};
  for (const d of toMeta.params) {
    if (fromMeta.params.some((x) => x.id === d.id) && from[d.id] !== undefined) carried[d.id] = from[d.id];
  }
  return resolveParams(toAlgo, carried);
}

const fmt = (v: ParamValue) => (typeof v === 'number' ? (Math.abs(v) >= 100 || Number.isInteger(v) ? String(v) : v.toPrecision(3).replace(/\.?0+$/, '')) : String(v));

function describeChanges(prev: Experiment | null, next: Experiment): string[] {
  if (!prev) return ['first run'];
  const out: string[] = [];
  if (prev.algoId !== next.algoId) out.push(`${algoMeta(prev.algoId).short} → ${algoMeta(next.algoId).short}`);
  const a = effectiveParams(prev);
  const b = effectiveParams(next);
  for (const d of algoMeta(next.algoId).params) {
    if (prev.algoId === next.algoId && a[d.id] !== b[d.id]) out.push(`${d.symbol ?? d.label} ${fmt(a[d.id])}→${fmt(b[d.id])}`);
  }
  if (prev.run.seed !== next.run.seed) out.push(`seed ${next.run.seed}`);
  if (prev.run.episodes !== next.run.episodes) out.push(`episodes ${next.run.episodes}`);
  if (JSON.stringify(prev.env) !== JSON.stringify(next.env)) out.push('level edited');
  if (next.tweaks.length > prev.tweaks.length) out.push('live tweak');
  return out.length ? out : ['rerun'];
}

const HISTORY_KEY = 'rl-arcade-history-v1';
const loadHistory = (): HistoryEntry[] => {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? (JSON.parse(raw) as HistoryEntry[]) : [];
  } catch {
    return [];
  }
};
const saveHistory = (h: HistoryEntry[]) => {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(h));
  } catch {
    /* storage unavailable: history stays in memory */
  }
};

let historyId = 1;
const timers: [number, number] = [0, 0];

// --------------------------------------------------------------------------- store

const initialLevel = getLevel('cliff-walk');

export const useStore = create<Store>((set, get) => {
  /** Ask the worker for a slot's result (debounced), then store it if still current. */
  const schedule = (i: SlotIndex) => {
    clearTimeout(timers[i]);
    timers[i] = window.setTimeout(() => {
      const exp = get().slots[i].exp;
      const hash = expHash(exp);
      set((s) => ({ slots: patchSlot(s.slots, i, { status: 'running' }) }));
      runExp(exp, i).then(
        (res) => {
          const cur = get().slots[i];
          if (expHash(cur.exp) !== hash) return;
          set((s) => ({ slots: patchSlot(s.slots, i, { res, status: 'idle', error: undefined }) }));
          clampPlayhead();
          recordHistory(i, exp, hash, res);
        },
        (err: Error) => set((s) => ({ slots: patchSlot(s.slots, i, { status: 'error', error: err.message }) })),
      );
    }, 90);
  };

  const patchSlot = (slots: [Slot, Slot], i: SlotIndex, patch: Partial<Slot>): [Slot, Slot] => {
    const next: [Slot, Slot] = [slots[0], slots[1]];
    next[i] = { ...slots[i], ...patch };
    return next;
  };

  const recordHistory = (i: SlotIndex, exp: Experiment, hash: string, res: RunResult) => {
    const s = get();
    if (i === 1 && s.mode !== 'versus') return;
    if (s.history.some((h) => h.hash === hash)) return;
    const prev = s.history[0]?.exp ?? null;
    const entry: HistoryEntry = {
      id: historyId++,
      hash,
      levelId: s.levelId,
      algoId: exp.algoId,
      exp,
      score: res.score,
      converged: res.converged,
      changes: describeChanges(prev && prev.env.kind === exp.env.kind ? prev : null, exp),
      isBandit: res.kind === 'bandit',
    };
    const history = [entry, ...s.history].slice(0, 20);
    set({ history });
    saveHistory(history);
  };

  const clampPlayhead = () => {
    const s = get();
    const r = s.slots[0].res;
    if (!r) return;
    if (r.kind === 'bandit') {
      set({ ep: 0, t: Math.min(s.t, r.T) });
      return;
    }
    const E = Math.max(r.E, s.mode === 'versus' && s.slots[1].res?.kind === 'grid' ? s.slots[1].res.E : 0);
    const ep = Math.min(s.ep, E - 1);
    set({ ep, t: Math.min(s.t, maxLen(ep)) });
  };

  /** Longest episode length at `ep` among the visible runs. */
  const maxLen = (ep: number) => {
    const s = get();
    let m = 0;
    for (const i of s.mode === 'versus' ? [0, 1] : [0]) {
      const r = s.slots[i].res;
      if (r?.kind === 'grid' && ep < r.E) m = Math.max(m, epLength(r, ep));
      if (r?.kind === 'bandit') m = Math.max(m, r.T);
    }
    return m;
  };

  /** Replace slot i's experiment; in solo mode the old run becomes the baseline. */
  const commit = (i: SlotIndex, exp: Experiment, opts: { keepPrevAlgo?: boolean } = {}) => {
    const s = get();
    const old = s.slots[i];
    const patch: Partial<State> = {};
    if (i === 0 && s.mode === 'solo' && !s.pinned && old.res) {
      patch.baseline = { exp: old.exp, res: old.res };
    }
    if (i === 0 && s.mode === 'solo' && !opts.keepPrevAlgo) patch.prevAlgoId = old.exp.algoId !== exp.algoId ? old.exp.algoId : s.prevAlgoId;
    set({ ...patch, slots: patchSlot(s.slots, i, { exp }), sweep: s.sweep?.slot === i ? null : s.sweep });
    schedule(i);
  };

  /** Apply a change to the environment/run of both slots (they always share them). */
  const commitShared = (f: (e: Experiment) => Experiment) => {
    commit(0, f(get().slots[0].exp), { keepPrevAlgo: true });
    if (get().mode === 'versus') commit(1, f(get().slots[1].exp));
    else set((s) => ({ slots: patchSlot(s.slots, 1, { exp: f(s.slots[1].exp), res: null }) }));
  };

  const currentLayout = (): GridLayout | null => {
    const e = get().slots[0].exp.env;
    return e.kind === 'grid' ? e.layout : null;
  };

  return {
    levelId: initialLevel.id,
    mode: 'solo',
    slots: [slot(expFromLevel(initialLevel, 'A')), slot(expFromLevel(initialLevel, 'B'))],
    editSlot: 0,
    baseline: null,
    pinned: false,
    prevAlgoId: null,
    ep: 0,
    t: 0,
    playing: false,
    speed: 2,
    overlays: { values: true, q: false, arrows: true, visits: false, trail: true, ghost: true, diff: false, numbers: true },
    changeMode: 'rerun',
    hoverParam: null,
    tool: null,
    windDir: 0,
    teleportFrom: null,
    undo: [],
    redo: [],
    history: loadHistory(),
    sweep: null,
    sound: false,
    crt: true,
    palette: 'neon',
    showTruth: false,
    help: false,
    banner: null,
    toast: null,

    loadLevel(id) {
      const level = getLevel(id);
      set({
        levelId: level.id,
        slots: [slot(expFromLevel(level, 'A')), slot(expFromLevel(level, 'B'))],
        baseline: null,
        pinned: false,
        prevAlgoId: null,
        ep: 0,
        t: 0,
        playing: false,
        undo: [],
        redo: [],
        sweep: null,
        tool: null,
        banner: null,
        editSlot: 0,
      });
      schedule(0);
      if (get().mode === 'versus') schedule(1);
    },

    setMode(m) {
      if (m === get().mode) return;
      const s = get();
      if (m === 'versus') {
        const a = s.slots[0].exp;
        const b = s.slots[1].exp;
        const sameKind = b.env.kind === a.env.kind;
        const level = getLevel(s.levelId);
        const fits = (id: string | undefined) => !!id && algoMeta(id).envKind === a.env.kind;
        // P2 starts as something different from P1: the baseline's algorithm, else the level's suggestion
        const candidates = [s.baseline?.exp.algoId, sameKind ? b.algoId : undefined, level.B.algoId, level.A.algoId];
        const algoB = candidates.find((id) => fits(id) && id !== a.algoId) ?? (fits(b.algoId) ? b.algoId : level.B.algoId);
        const paramsB =
          algoB === b.algoId && sameKind ? b.params : algoB === level.B.algoId ? resolveParams(algoB, level.B.params) : algoB === level.A.algoId ? resolveParams(algoB, level.A.params) : carryParams(a.algoId, effectiveParams(a), algoB);
        const expB: Experiment = { ...a, algoId: algoB, params: paramsB, tweaks: [] };
        set({ mode: 'versus', slots: patchSlot(s.slots, 1, { exp: expB, res: null }), editSlot: 0 });
        schedule(1);
      } else {
        set({ mode: 'solo', editSlot: 0, baseline: s.pinned ? s.baseline : null });
      }
      clampPlayhead();
    },

    setEditSlot(i) {
      set({ editSlot: i });
    },

    setAlgo(i, algoId) {
      const cur = get().slots[i].exp;
      if (cur.algoId === algoId) return;
      const params = carryParams(cur.algoId, effectiveParams(cur), algoId);
      commit(i, { ...cur, algoId, params, tweaks: [] });
    },

    setParam(i, id, v) {
      const s = get();
      const cur = s.slots[i].exp;
      if (s.changeMode === 'live' && s.slots[i].res) {
        const at = cur.env.kind === 'bandit' ? s.t : s.ep;
        const tweaks = [...cur.tweaks.filter((t) => !(t.at === at && id in t.params)), { at, params: { [id]: v } }];
        commit(i, { ...cur, tweaks });
      } else {
        const tweaks = cur.tweaks.map((t) => {
          const p = { ...t.params };
          delete p[id];
          return { ...t, params: p };
        }).filter((t) => Object.keys(t.params).length > 0);
        commit(i, { ...cur, params: { ...cur.params, [id]: v }, tweaks });
      }
    },

    resetParam(i, id) {
      const d = algoMeta(get().slots[i].exp.algoId).params.find((p) => p.id === id);
      if (d) get().setParam(i, id, d.default);
    },

    clearTweaks(i) {
      const cur = get().slots[i].exp;
      if (cur.tweaks.length) commit(i, { ...cur, tweaks: [] });
    },

    setRun(patch) {
      commitShared((e) => ({ ...e, run: { ...e.run, ...patch } }));
    },

    setLayout(next, pushUndo = true) {
      const cur = currentLayout();
      if (!cur) return;
      if (pushUndo) set((s) => ({ undo: [...s.undo.slice(-49), cur], redo: [] }));
      commitShared((e) => ({ ...e, env: { kind: 'grid', layout: next } }));
    },

    paintCell(i) {
      const s = get();
      const L = currentLayout();
      if (!L || !s.tool) return;
      const cells = L.cells.slice();
      const old = cells[i];
      let cell: Cell;
      switch (s.tool) {
        case 'erase':
          cell = { t: 'empty' };
          break;
        case 'wind':
          cell = { t: 'wind', dir: s.windDir, strength: old.t === 'wind' && old.dir === s.windDir ? ((old.strength ?? 1) % 2) + 1 : 1 };
          break;
        case 'teleport':
          if (s.teleportFrom === null) {
            set({ teleportFrom: i, toast: 'Now click where the teleport should send the agent.' });
            return;
          }
          if (s.teleportFrom === i) {
            set({ teleportFrom: null });
            return;
          }
          cells[s.teleportFrom] = { t: 'teleport', target: i, reward: 5, label: 'T' };
          set({ teleportFrom: null });
          get().setLayout({ ...L, cells, events: undefined });
          return;
        case 'start':
          for (let j = 0; j < cells.length; j++) if (cells[j].t === 'start') cells[j] = { t: 'empty' };
          cell = { t: 'start' };
          break;
        default:
          cell = { t: s.tool };
      }
      if (JSON.stringify(old) === JSON.stringify(cell)) return;
      cells[i] = cell;
      // keep at least one non-terminal open cell
      if (!cells.some((c) => c.t !== 'wall' && c.t !== 'goal' && c.t !== 'lava')) return;
      get().setLayout({ ...L, cells, events: undefined });
    },

    resize(w, h) {
      const L = currentLayout();
      if (!L) return;
      const W = Math.max(2, Math.min(16, w));
      const H = Math.max(1, Math.min(16, h));
      if (W === L.w && H === L.h) return;
      get().setLayout(resizeLayout(L, W, H));
    },

    undoEdit() {
      const s = get();
      const prev = s.undo[s.undo.length - 1];
      const cur = currentLayout();
      if (!prev || !cur) return;
      set({ undo: s.undo.slice(0, -1), redo: [...s.redo, cur] });
      get().setLayout(prev, false);
    },

    redoEdit() {
      const s = get();
      const next = s.redo[s.redo.length - 1];
      const cur = currentLayout();
      if (!next || !cur) return;
      set({ redo: s.redo.slice(0, -1), undo: [...s.undo, cur] });
      get().setLayout(next, false);
    },

    setBanditSteps(n) {
      commitShared((e) => (e.env.kind === 'bandit' ? { ...e, env: { kind: 'bandit', bandit: { ...e.env.bandit, steps: n } } } : e));
    },
    setBanditK(n) {
      commitShared((e) => (e.env.kind === 'bandit' ? { ...e, env: { kind: 'bandit', bandit: { ...e.env.bandit, k: n } } } : e));
    },
    setBanditDist(d) {
      commitShared((e) => (e.env.kind === 'bandit' ? { ...e, env: { kind: 'bandit', bandit: { ...e.env.bandit, dist: d } } } : e));
    },

    togglePin() {
      const s = get();
      if (s.pinned) set({ pinned: false, toast: 'Baseline unpinned. The previous run will be the baseline again.' });
      else if (s.slots[0].res) set({ pinned: true, baseline: { exp: s.slots[0].exp, res: s.slots[0].res }, toast: 'This run is now the pinned baseline.' });
    },

    setPlayhead(ep, t) {
      set({ ep: Math.max(0, ep), t: Math.max(0, t) });
      clampPlayhead();
    },
    setPlaying(p) {
      const s = get();
      const r = s.slots[0].res;
      if (p && r) {
        // pressing play at the very end restarts
        const atEnd = r.kind === 'bandit' ? s.t >= r.T : s.ep >= r.E - 1 && s.t >= maxLen(s.ep);
        if (atEnd) set({ ep: 0, t: 0 });
      }
      set({ playing: p, banner: null });
    },
    setSpeed(i) {
      set({ speed: Math.max(0, Math.min(SPEEDS.length - 1, i)) });
    },

    /** Advance (or rewind) the shared playhead by n steps, rolling over episodes. Returns nothing. */
    stepBy(n) {
      const s = get();
      const r = s.slots[0].res;
      if (!r) return;
      if (r.kind === 'bandit') {
        const t = Math.max(0, Math.min(r.T, s.t + n));
        set({ t, ...(t >= r.T ? { playing: false } : {}) });
        return;
      }
      const E = Math.max(r.E, s.mode === 'versus' && s.slots[1].res?.kind === 'grid' ? s.slots[1].res.E : 0);
      let ep = s.ep;
      let t = s.t;
      let left = n;
      while (left > 0) {
        const len = maxLen(ep);
        if (t < len) {
          const d = Math.min(left, len - t);
          t += d;
          left -= d;
        } else if (ep < E - 1) {
          ep++;
          t = 0;
          left--;
        } else {
          set({ ep, t, playing: false });
          return;
        }
      }
      while (left < 0) {
        if (t > 0) {
          t--;
          left++;
        } else if (ep > 0) {
          ep--;
          t = maxLen(ep);
          left++;
        } else break;
      }
      set({ ep, t });
    },

    stepEpisode(dir) {
      const s = get();
      const r = s.slots[0].res;
      if (!r || r.kind === 'bandit') return;
      const E = Math.max(r.E, s.mode === 'versus' && s.slots[1].res?.kind === 'grid' ? s.slots[1].res.E : 0);
      const ep = Math.max(0, Math.min(E - 1, s.ep + dir));
      set({ ep, t: maxLen(ep) });
    },

    restart() {
      set({ ep: 0, t: 0, banner: null });
    },

    toggleOverlay(id) {
      set((s) => {
        const overlays = { ...s.overlays, [id]: !s.overlays[id] };
        if (id === 'q' && overlays.q) overlays.values = false;
        if (id === 'values' && overlays.values) overlays.q = false;
        return { overlays };
      });
    },

    set(p) {
      set(p);
    },

    restoreHistory(id, asB) {
      const h = get().history.find((x) => x.id === id);
      if (!h) return;
      if (h.levelId !== get().levelId) set({ levelId: h.levelId });
      if (asB) {
        if (get().mode !== 'versus') get().setMode('versus');
        commit(1, structuredClone(h.exp));
      } else {
        commit(0, structuredClone(h.exp));
      }
    },

    startSweep(i, paramId) {
      const s = get();
      const exp = s.slots[i].exp;
      const d = algoMeta(exp.algoId).params.find((p) => p.id === paramId);
      if (!d) return;
      let values: (number | string | boolean)[];
      if (d.scale === 'bool') values = [false, true];
      else if (d.scale === 'enum') values = d.options ?? [];
      else {
        const lo = d.min ?? 0;
        const hi = d.max ?? 1;
        const K = 8;
        values = Array.from({ length: K }, (_, j) => {
          const f = j / (K - 1);
          let v = d.scale === 'log' ? lo * Math.pow(hi / lo, f) : lo + (hi - lo) * f;
          if (d.scale === 'int') v = Math.round(v);
          return Number(v.toPrecision(3));
        });
        values = [...new Set(values)];
      }
      const seeds = exp.env.kind === 'bandit' ? 1 : 3;
      set({ sweep: { slot: i, paramId, status: 'running', done: 0, total: values.length, points: [] } });
      runSweep({ ...exp, params: effectiveParams(exp), tweaks: [] }, paramId, values, seeds, (done, total) => {
        const cur = get().sweep;
        if (cur && cur.paramId === paramId && cur.status === 'running') set({ sweep: { ...cur, done, total } });
      }).then(
        (points) => {
          const cur = get().sweep;
          if (cur && cur.paramId === paramId) set({ sweep: { ...cur, status: 'done', points } });
        },
        (e: Error) => set({ sweep: { slot: i, paramId, status: 'error', done: 0, total: 0, points: [], message: e.message } }),
      );
    },

    loadShared(level, mode, a, b) {
      set({
        levelId: LEVELS.some((l) => l.id === level) ? level : get().levelId,
        mode,
        slots: [slot(a), slot(b ?? { ...a })],
        baseline: null,
        pinned: false,
        ep: 0,
        t: 0,
      });
      schedule(0);
      if (mode === 'versus') schedule(1);
    },

    flash(msg) {
      set({ toast: msg });
    },
  };
});

/** Kick off the first computation. */
export function bootStore() {
  const s = useStore.getState();
  s.loadLevel(s.levelId);
}
