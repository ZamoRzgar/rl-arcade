import { useEffect, useRef, useState } from 'react';
import type { BanditRunResult, GridRunResult, RunResult } from '../core/engine';
import type { Experiment, GridLayout } from '../core/types';
import { useStore } from '../state/store';
import { buildBanditFrame, buildGridFrame, diffGridFrames, tracesFromTrail, type GridFrame } from '../viz/frame';
import { drawGrid, drawDisagreement } from '../viz/drawGrid';
import { drawBandit } from '../viz/drawBandit';
import { readTheme, type RGB } from '../viz/theme';
import { clock } from './playback';
import { effectiveParams } from '../state/store';
import { num, str } from '../core/agents/common';

export type ArenaRole = 'solo' | 'p1' | 'p2';

interface Props {
  role: ArenaRole;
  /** the run drawn as a ghost and diffed against (baseline in solo, other player in versus) */
  compare: { exp: Experiment; res: RunResult } | null;
  exp: Experiment;
  res: RunResult | null;
  editable: boolean;
}

const scaleCache = new WeakMap<object, number>();
/** Max |value| over every snapshot of a run, so colors stay comparable across time. */
export function runScale(r: GridRunResult): number {
  const hit = scaleCache.get(r);
  if (hit !== undefined) return hit;
  let m = 0;
  for (let i = 0; i < r.snapV.length; i++) if (Number.isFinite(r.snapV[i])) m = Math.max(m, Math.abs(r.snapV[i]));
  for (let i = 0; i < r.snapQ.length; i++) if (Number.isFinite(r.snapQ[i])) m = Math.max(m, Math.abs(r.snapQ[i]));
  m = Math.max(m, 1e-6);
  scaleCache.set(r, m);
  return m;
}

const sameShape = (a: GridLayout, b: GridLayout) => a.w === b.w && a.h === b.h;

function useWidth(ref: React.RefObject<HTMLElement>) {
  // start at 0 so a too-wide first render never stretches the page on phones
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => setW(Math.floor(entries[0].contentRect.width)));
    ro.observe(el);
    setW(Math.floor(el.getBoundingClientRect().width));
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

const reduced = () => typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;

export function ArenaCanvas({ role, compare, exp, res, editable }: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const width = useWidth(wrap);
  const hover = useRef(-1);
  const painting = useRef<number | null>(null);
  const tool = useStore((s) => s.tool);
  const layout = exp.env.kind === 'grid' ? exp.env.layout : null;

  const cell = layout ? Math.max(14, Math.min(68, Math.floor(width / layout.w), Math.floor(560 / layout.h))) : 0;
  const cssW = layout ? cell * layout.w : width;
  const cssH = layout ? cell * layout.h : Math.max(300, Math.min(380, Math.round(width * 0.55)));

  useEffect(() => {
    const cv = canvas.current;
    if (!cv) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
    cv.style.width = `${cssW}px`;
    cv.style.height = `${cssH}px`;
    const ctx = cv.getContext('2d')!;
    let raf = 0;
    const playerColor = (th: ReturnType<typeof readTheme>): RGB => (role === 'p2' ? th.neg : role === 'p1' ? th.pos : th.text);

    const draw = (time: number) => {
      const s = useStore.getState();
      const th = readTheme();
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (!res) {
        ctx.fillStyle = `rgb(${th.screen.join(',')})`;
        ctx.fillRect(0, 0, cssW, cssH);
      } else if (res.kind === 'grid' && layout) {
        const frame = buildGridFrame(res, layout, s.ep, s.t);
        let ghost: GridFrame | null = null;
        if (compare && compare.res.kind === 'grid' && compare.exp.env.kind === 'grid' && sameShape(compare.exp.env.layout, layout)) {
          ghost = buildGridFrame(compare.res, compare.exp.env.layout, s.ep, s.t);
        }
        const diff = ghost ? diffGridFrames(frame, ghost) : null;
        const p = effectiveParams(exp);
        const traces = exp.algoId === 'sarsa-lambda' && s.overlays.trail ? tracesFromTrail(frame.trail, res.S, num(p, 'gamma', 0.9) * num(p, 'lambda', 0.9), str(p, 'trace', 'replacing') === 'replacing') : null;
        const scale = Math.max(runScale(res), ghost && compare?.res.kind === 'grid' ? runScale(compare.res) : 0);
        drawGrid(ctx, {
          frame,
          ghost: role === 'solo' ? ghost : null,
          diff,
          overlays: { ...s.overlays, ghost: s.overlays.ghost && role === 'solo' },
          theme: th,
          cell,
          time,
          frac: clock.frac,
          policyGlyphs: res.hasP && !res.planner,
          traces,
          nWindow: exp.algoId === 'nstep-sarsa' ? Math.round(num(p, 'n', 4)) : 0,
          horizon: s.hoverParam === 'gamma' ? 1 / Math.max(1e-3, 1 - num(p, 'gamma', 0.9)) : 0,
          hoverCell: editable && s.tool ? hover.current : -1,
          reducedMotion: reduced(),
          valueScale: scale,
          playerColor: playerColor(th),
        });
      } else if (res.kind === 'bandit') {
        const frame = buildBanditFrame(res, s.t);
        const ghost = role === 'solo' && compare?.res.kind === 'bandit' && compare.res.k === res.k && s.overlays.ghost ? buildBanditFrame(compare.res as BanditRunResult, s.t) : null;
        drawBandit(ctx, { res, frame, ghost, theme: th, width: cssW, height: cssH, showTruth: s.showTruth, time, playerColor: playerColor(th), hoverParam: s.hoverParam });
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(raf);
  }, [res, compare, exp, layout, cell, cssW, cssH, role, editable]);

  const cellAt = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!layout) return -1;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * layout.w;
    const y = ((e.clientY - rect.top) / rect.height) * layout.h;
    if (x < 0 || y < 0 || x >= layout.w || y >= layout.h) return -1;
    return Math.floor(y) * layout.w + Math.floor(x);
  };
  const canEdit = editable && !!tool && !!layout;

  return (
    <div ref={wrap} style={{ width: '100%' }}>
      <div className={`screen${canEdit ? ' editing' : ''}`} style={{ width: cssW, maxWidth: '100%', margin: '0 auto' }}>
        <canvas
          ref={canvas}
          role="img"
          aria-label={layout ? `Grid world ${layout.w} by ${layout.h}` : 'Slot machine cabinet'}
          onPointerMove={(e) => {
            hover.current = cellAt(e);
            if (canEdit && painting.current !== null) {
              const c = cellAt(e);
              if (c >= 0 && c !== painting.current && useStore.getState().tool !== 'teleport') {
                painting.current = c;
                useStore.getState().paintCell(c);
              }
            }
          }}
          onPointerLeave={() => {
            hover.current = -1;
            painting.current = null;
          }}
          onPointerDown={(e) => {
            if (!canEdit) return;
            const c = cellAt(e);
            if (c < 0) return;
            e.currentTarget.setPointerCapture(e.pointerId);
            painting.current = c;
            useStore.getState().paintCell(c);
          }}
          onPointerUp={() => {
            painting.current = null;
          }}
        />
      </div>
    </div>
  );
}

/** Small map of where two players' greedy policies disagree. */
export function DisagreementCanvas({ a, b }: { a: { exp: Experiment; res: GridRunResult }; b: { exp: Experiment; res: GridRunResult } }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const width = useWidth(wrap);
  const L = a.exp.env.kind === 'grid' ? a.exp.env.layout : null;
  const cell = L ? Math.max(10, Math.min(30, Math.floor(width / L.w))) : 10;
  useEffect(() => {
    const cv = canvas.current;
    if (!cv || !L || b.exp.env.kind !== 'grid' || !sameShape(L, b.exp.env.layout)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = cell * L.w * dpr;
    cv.height = cell * L.h * dpr;
    cv.style.width = `${cell * L.w}px`;
    cv.style.height = `${cell * L.h}px`;
    const ctx = cv.getContext('2d')!;
    let raf = 0;
    let key = '';
    const loop = () => {
      const s = useStore.getState();
      const k = `${s.ep}|${s.t}`;
      if (k !== key) {
        key = k;
        const th = readTheme();
        const fa = buildGridFrame(a.res, L, s.ep, s.t);
        const fb = buildGridFrame(b.res, (b.exp.env as { layout: GridLayout }).layout, s.ep, s.t);
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        drawDisagreement(ctx, fa, fb, diffGridFrames(fa, fb), th, cell, th.pos, th.neg);
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [a, b, L, cell]);
  return (
    <div ref={wrap}>
      <div className="screen" style={{ width: L ? cell * L.w : 0, maxWidth: '100%' }}>
        <canvas ref={canvas} role="img" aria-label="Cells where the two players' policies disagree" />
      </div>
    </div>
  );
}
