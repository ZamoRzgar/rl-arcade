import { algoMeta } from '../core/registry';
import type { RunResult } from '../core/engine';
import type { Experiment } from '../core/types';
import { SPEEDS, effectiveParams, useStore, type OverlayId } from '../state/store';
import { ArenaCanvas, DisagreementCanvas, type ArenaRole } from './ArenaCanvas';
import { epLength } from '../viz/frame';
import { num } from '../core/agents/common';

const OVERLAYS: { id: OverlayId; label: string; title: string; grid?: boolean }[] = [
  { id: 'values', label: 'V', title: 'Value heatmap: how good each cell is', grid: true },
  { id: 'q', label: 'Q', title: 'Q-triangles: the value of each of the 4 actions per cell', grid: true },
  { id: 'arrows', label: 'π', title: 'Policy arrows (greedy action, or probabilities for policy-gradient methods)', grid: true },
  { id: 'trail', label: 'TRAIL', title: 'This episode’s path. Pink sparks mark exploratory moves', grid: true },
  { id: 'visits', label: 'VISITS', title: 'How often each cell has been visited', grid: true },
  { id: 'numbers', label: '123', title: 'Show values as numbers (large cells only)', grid: true },
  { id: 'ghost', label: 'GHOST', title: 'Show the baseline run faded underneath' },
  { id: 'diff', label: 'DIFF', title: 'Color cells by how much the value changed versus the baseline (or the other player)', grid: true },
];

export function Stage() {
  const mode = useStore((s) => s.mode);
  const slots = useStore((s) => s.slots);
  const baseline = useStore((s) => s.baseline);
  const overlays = useStore((s) => s.overlays);
  const banner = useStore((s) => s.banner);
  const crt = useStore((s) => s.crt);
  const tool = useStore((s) => s.tool);
  const st = useStore.getState();
  const kind = slots[0].exp.env.kind;
  const versus = mode === 'versus';
  const compareFor = (i: 0 | 1) => {
    if (versus) {
      const o = slots[i === 0 ? 1 : 0];
      return o.res ? { exp: o.exp, res: o.res } : null;
    }
    return baseline;
  };
  const a = slots[0];
  const b = slots[1];
  return (
    <main className="stage col">
      <section className="panel">
        <div className="overlay-bar" role="group" aria-label="Layers">
          <span className="section-title" style={{ margin: '0 6px 0 0' }}>
            Layers
          </span>
          {OVERLAYS.filter((o) => kind === 'grid' || !o.grid).map((o) => (
            <button
              key={o.id}
              type="button"
              className={`chip-toggle${o.id === 'diff' ? ' diff' : ''}`}
              aria-pressed={overlays[o.id]}
              title={o.title}
              onClick={() => st.toggleOverlay(o.id)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </section>

      <section className={`bezel${crt ? ' crt' : ''}`} aria-label="Arena">
        {banner && (
          <div className="banner" role="status">
            {banner.split('\n')[0]}
            <small>{banner.split('\n')[1]}</small>
          </div>
        )}
        <div className={`arenas${versus ? ' versus' : ''}`}>
          <Arena role={versus ? 'p1' : 'solo'} exp={a.exp} res={a.res} status={a.status} error={a.error} compare={compareFor(0)} editable={!!tool} />
          {versus && <Arena role="p2" exp={b.exp} res={b.res} status={b.status} error={b.error} compare={compareFor(1)} editable={!!tool} />}
        </div>
        <Hud />
        <Transport />
        <Legend />
        {versus && kind === 'grid' && a.res?.kind === 'grid' && b.res?.kind === 'grid' && (
          <div className="disagree">
            <div className="arena-head">
              <span className="who">Disagreement map</span>
              <span className="status">
                cells where P1 and P2 would act differently · <span style={{ color: 'var(--p1)' }}>P1 values it more</span> /{' '}
                <span style={{ color: 'var(--p2)' }}>P2 values it more</span>
              </span>
            </div>
            <DisagreementCanvas a={{ exp: a.exp, res: a.res }} b={{ exp: b.exp, res: b.res }} />
          </div>
        )}
      </section>
    </main>
  );
}

function Arena({ role, exp, res, status, error, compare, editable }: { role: ArenaRole; exp: Experiment; res: RunResult | null; status: string; error?: string; compare: { exp: Experiment; res: RunResult } | null; editable: boolean }) {
  const meta = algoMeta(exp.algoId);
  const pinned = useStore((s) => s.pinned);
  const label = role === 'p1' ? 'P1' : role === 'p2' ? 'P2' : '1P';
  return (
    <div className="arena">
      <div className="arena-head">
        <span className={`who ${role === 'solo' ? '' : role}`}>
          {label} · {meta.name}
        </span>
        <span className="status">
          {res ? `${res.elapsedMs.toFixed(0)} ms · ${res.seedsUsed > 1 ? `${res.seedsUsed} seeds` : `seed ${exp.run.seed}`}` : ''}
          {role === 'solo' && compare ? ` · vs ${pinned ? 'pinned ' : ''}baseline (${algoMeta(compare.exp.algoId).short})` : ''}
        </span>
      </div>
      <div style={{ position: 'relative' }}>
        <ArenaCanvas role={role} exp={exp} res={res} compare={compare} editable={editable} />
        {status === 'running' && <span className="computing">COMPUTING…</span>}
      </div>
      {error && <p className="error">This run failed: {error}. Try fewer episodes or a smaller grid.</p>}
      {res?.kind === 'grid' && res.truncatedSteps && <p className="error">The run hit the 1.5M-step recording cap and stopped early. Lower the episodes or max steps.</p>}
    </div>
  );
}

function Hud() {
  const ep = useStore((s) => s.ep);
  const t = useStore((s) => s.t);
  const slots = useStore((s) => s.slots);
  const mode = useStore((s) => s.mode);
  const items = (mode === 'versus' ? [0, 1] : [0]) as (0 | 1)[];
  return (
    <div className="hud" aria-live="polite" aria-atomic="false">
      {items.map((i) => {
        const r = slots[i].res;
        const p = effectiveParams(slots[i].exp);
        const cls = i === 1 ? 'v p2' : 'v';
        const tag = mode === 'versus' ? `P${i + 1} ` : '';
        if (!r) return <span key={i} className="muted">Loading…</span>;
        if (r.kind === 'bandit') {
          let sum = 0;
          let opt = 0;
          for (let j = 0; j < t; j++) {
            sum += r.rewards[j];
            if (r.arms[j] === r.optimal) opt++;
          }
          return (
            <span key={i} style={{ display: 'contents' }}>
              <span>
                <span className="k">{tag}PULL</span>
                <span className={cls}>{String(t).padStart(4, '0')}</span>
              </span>
              <span>
                <span className="k">AVG</span>
                <span className={cls}>{t ? (sum / t).toFixed(3) : '—'}</span>
              </span>
              <span>
                <span className="k">BEST ARM</span>
                <span className={cls}>{t ? `${Math.round((100 * opt) / t)}%` : '—'}</span>
              </span>
            </span>
          );
        }
        if (r.planner) {
          return (
            <span key={i} style={{ display: 'contents' }}>
              <span>
                <span className="k">{tag}FRAME</span>
                <span className={cls}>
                  {String(Math.min(ep, r.E - 1) + 1).padStart(3, '0')}/{r.E}
                </span>
              </span>
              <span>
                <span className="k">PHASE</span>
                <span className={cls}>{r.phases[Math.min(ep, r.E - 1)] === 1 ? 'IMPROVE' : r.algoId === 'policy-iteration' ? 'EVALUATE' : 'SWEEP'}</span>
              </span>
              <span>
                <span className="k">MAX ΔV</span>
                <span className={cls}>{(r.deltas[Math.min(ep, r.E - 1)] ?? 0).toExponential(1)}</span>
              </span>
            </span>
          );
        }
        const e = Math.min(ep, r.E - 1);
        const len = epLength(r, e);
        const tt = Math.min(t, len);
        let ret = 0;
        for (let j = r.epStart[e]; j < r.epStart[e] + tt; j++) ret += r.stR[j];
        const eps = p.epsilon !== undefined ? num(p, 'epsilon', 0) * Math.pow(num(p, 'epsDecay', 1), e) : null;
        return (
          <span key={i} style={{ display: 'contents' }}>
            <span>
              <span className="k">{tag}EP</span>
              <span className={cls}>
                {String(e + 1).padStart(3, '0')}/{r.E}
              </span>
            </span>
            <span>
              <span className="k">STEP</span>
              <span className={cls}>{String(tt).padStart(3, '0')}</span>
            </span>
            <span>
              <span className="k">RETURN</span>
              <span className={cls}>{ret.toFixed(ret % 1 ? 1 : 0)}</span>
            </span>
            {eps !== null && (
              <span>
                <span className="k">ε</span>
                <span className={cls}>{eps.toFixed(3)}</span>
              </span>
            )}
          </span>
        );
      })}
    </div>
  );
}

function Transport() {
  const playing = useStore((s) => s.playing);
  const speed = useStore((s) => s.speed);
  const ep = useStore((s) => s.ep);
  const t = useStore((s) => s.t);
  const r = useStore((s) => s.slots[0].res);
  const rb = useStore((s) => (s.mode === 'versus' ? s.slots[1].res : null));
  const pinned = useStore((s) => s.pinned);
  const mode = useStore((s) => s.mode);
  const st = useStore.getState();
  const bandit = r?.kind === 'bandit';
  const E = r?.kind === 'grid' ? Math.max(r.E, rb?.kind === 'grid' ? rb.E : 0) : 1;
  const len = r?.kind === 'grid' ? Math.max(epLength(r, ep), rb?.kind === 'grid' && ep < rb.E ? epLength(rb, ep) : 0) : r?.kind === 'bandit' ? r.T : 0;
  return (
    <>
      <div className="transport">
        <button type="button" className="btn icon" title="Back to the start (Home)" aria-label="Restart" onClick={() => st.restart()}>
          ⏮
        </button>
        {!bandit && (
          <button type="button" className="btn icon" title="Previous episode" aria-label="Previous episode" onClick={() => st.stepEpisode(-1)}>
            ⏪
          </button>
        )}
        <button type="button" className="btn icon" title="Step back (←)" aria-label="Step back" onClick={() => st.stepBy(-1)}>
          ◀
        </button>
        <button type="button" className="btn primary" style={{ minWidth: 92 }} onClick={() => st.setPlaying(!playing)} aria-label={playing ? 'Pause' : 'Play'}>
          {playing ? '❚❚ PAUSE' : '▶ START'}
        </button>
        <button type="button" className="btn icon" title="Step forward (→)" aria-label="Step forward" onClick={() => st.stepBy(1)}>
          ▶
        </button>
        {!bandit && (
          <button type="button" className="btn icon" title="Next episode (Shift + →)" aria-label="Next episode" onClick={() => st.stepEpisode(1)}>
            ⏩
          </button>
        )}
        <label className="speed">
          SPEED
          <input type="range" min={0} max={SPEEDS.length - 1} value={speed} onChange={(e) => st.setSpeed(Number(e.target.value))} aria-valuetext={SPEEDS[speed].label} />
          <span style={{ color: 'var(--hi)', minWidth: 40 }}>{SPEEDS[speed].label}</span>
        </label>
        {mode === 'solo' && (
          <button type="button" className={`btn small${pinned ? ' on' : ''}`} onClick={() => st.togglePin()} title="Keep this run as the baseline while you try changes (B)">
            {pinned ? 'Unpin baseline' : 'Pin baseline'}
          </button>
        )}
      </div>
      {r && !bandit && (
        <div className="timeline">
          <label htmlFor="tl-ep">
            <span>{r.kind === 'grid' && r.planner ? 'Frame' : 'Episode'} {ep + 1}</span>
            <span>
              step {Math.min(t, len)} / {len}
            </span>
          </label>
          <input id="tl-ep" type="range" min={0} max={Math.max(0, E - 1)} value={ep} onChange={(e) => st.setPlayhead(Number(e.target.value), 0)} />
          <div className="step-bar" aria-hidden="true">
            <span style={{ width: `${len ? (100 * Math.min(t, len)) / len : 0}%` }} />
          </div>
        </div>
      )}
      {r && bandit && (
        <div className="timeline">
          <label htmlFor="tl-t">
            <span>Pull {t}</span>
            <span>of {r.T}</span>
          </label>
          <input id="tl-t" type="range" min={0} max={r.T} value={t} onChange={(e) => st.setPlayhead(0, Number(e.target.value))} />
        </div>
      )}
    </>
  );
}

function Legend() {
  const overlays = useStore((s) => s.overlays);
  const kind = useStore((s) => s.slots[0].exp.env.kind);
  if (kind === 'bandit') {
    return (
      <div className="legend">
        <span>Bars and white ticks: estimated value</span>
        <span>· yellow whiskers: UCB bonus</span>
        <span>· curves: Thompson beliefs</span>
        <span>· dashed: true mean (when revealed)</span>
        <span>· ghost boxes: baseline estimate</span>
      </div>
    );
  }
  return (
    <div className="legend">
      <span>{overlays.diff ? 'Δ value' : 'value'}</span>
      <span className="muted">low</span>
      <span className="bar" style={{ background: 'linear-gradient(90deg, var(--val-neg), var(--screen), var(--val-pos))' }} />
      <span className="muted">high</span>
      <span>· amber arrows: policy · faded: baseline · outlined cells: policy changed</span>
    </div>
  );
}
