import { useEffect } from 'react';
import { SPEEDS, useStore } from '../state/store';
import { sfx } from '../design/sfx';
import { algoMeta } from '../core/registry';

/** Fractional progress toward the next step, read by the canvas for smooth sprite motion. */
export const clock = { frac: 1 };

/** Drives the shared playhead with requestAnimationFrame while playing. */
export function usePlayback() {
  useEffect(() => {
    let raf = 0;
    let last = performance.now();
    let acc = 0;
    const loop = (now: number) => {
      const dt = Math.min(100, now - last);
      last = now;
      const s = useStore.getState();
      const r = s.slots[0].res;
      if (s.playing && r) {
        const sp = SPEEDS[s.speed];
        const before = { ep: s.ep, t: s.t };
        if ('eps' in sp) {
          // turbo: jump whole episodes (shows end-of-episode values)
          if (r.kind === 'bandit') s.stepBy(sp.eps * 25);
          else for (let i = 0; i < sp.eps; i++) s.stepEpisode(1);
          clock.frac = 1;
          const after = useStore.getState();
          if (!after.playing || (after.ep === before.ep && after.t === before.t)) finish();
        } else {
          acc += (dt / 1000) * sp.sps;
          const n = Math.floor(acc);
          acc -= n;
          clock.frac = sp.sps > 40 ? 1 : acc;
          if (n > 0) {
            s.stepBy(n);
            const after = useStore.getState();
            if (after.sound && sp.sps <= 12) sound(after.ep, after.t);
            if (!after.playing) finish();
          }
        }
      } else {
        clock.frac = 1;
        acc = 0;
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, []);
}

function finish() {
  const s = useStore.getState();
  const r = s.slots[0].res;
  s.set({ playing: false });
  if (!r) return;
  const name = algoMeta(r.algoId).short;
  let msg: string;
  if (r.kind === 'bandit') msg = `GAME OVER\n${name}: average reward ${r.score.toFixed(3)}`;
  else if (r.planner) msg = `LEVEL CLEAR!\n${name} converged in ${r.E} frames`;
  else if (r.converged >= 0) msg = `LEVEL CLEAR!\n${name} settled by episode ${r.converged + 1}`;
  else msg = `RUN COMPLETE\n${name}`;
  s.set({ banner: msg });
  if (s.sound) sfx.clear();
}

function sound(ep: number, t: number) {
  const s = useStore.getState();
  const r = s.slots[0].res;
  if (!r) return;
  if (r.kind === 'bandit') {
    sfx.blip();
    return;
  }
  if (r.planner) return;
  const len = r.lengths[ep] ?? 0;
  if (t === len && len > 0) {
    const exp = s.slots[0].exp;
    if (exp.env.kind !== 'grid') return;
    const last = r.stNext[r.epStart[ep] + len - 1];
    const cell = exp.env.layout.cells[last];
    if (cell?.t === 'goal') sfx.goal();
    else if (cell?.t === 'lava') sfx.lava();
  } else if (t > 0) {
    const j = r.epStart[ep] + t - 1;
    if (r.stR[j] <= -50) sfx.lava();
    else sfx.blip();
  }
}
