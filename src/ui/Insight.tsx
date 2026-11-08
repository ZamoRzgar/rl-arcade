import { useEffect, useRef, useState } from 'react';
import type { RunResult, GridRunResult, BanditRunResult } from '../core/engine';
import type { AlgoMeta, Experiment } from '../core/types';
import { algoMeta } from '../core/registry';
import { movingAverage } from '../core/metrics';
import { effectiveParams, useStore } from '../state/store';
import { buildGridFrame, diffGridFrames } from '../viz/frame';
import { Chart, type Series } from './Chart';
import { EquationDiff } from './EquationDiff';
import { Pseudocode } from './Pseudocode';
import { BackupDiagram } from './BackupDiagram';
import { readTheme, diverging, rgba } from '../viz/theme';

interface Side {
  exp: Experiment;
  res: RunResult;
  label: string;
  color: string;
}

/** The pair being compared: (current, baseline) in solo, (P2, P1) in versus. */
function useComparison(): { main: Side | null; other: Side | null; versus: boolean } {
  const mode = useStore((s) => s.mode);
  const slots = useStore((s) => s.slots);
  const baseline = useStore((s) => s.baseline);
  const editSlot = useStore((s) => s.editSlot);
  if (mode === 'versus') {
    const a = slots[0];
    const b = slots[1];
    const sa = a.res ? { exp: a.exp, res: a.res, label: `P1 ${algoMeta(a.exp.algoId).short}`, color: 'var(--p1)' } : null;
    const sb = b.res ? { exp: b.exp, res: b.res, label: `P2 ${algoMeta(b.exp.algoId).short}`, color: 'var(--p2)' } : null;
    return editSlot === 1 ? { main: sb, other: sa, versus: true } : { main: sa, other: sb, versus: true };
  }
  const a = slots[0];
  return {
    main: a.res ? { exp: a.exp, res: a.res, label: `Now · ${algoMeta(a.exp.algoId).short}`, color: 'var(--neon-yellow)' } : null,
    other: baseline && baseline.res.kind === a.res?.kind ? { exp: baseline.exp, res: baseline.res, label: `Baseline · ${algoMeta(baseline.exp.algoId).short}`, color: 'var(--ghost)' } : null,
    versus: false,
  };
}

export function Insight() {
  const { main, other, versus } = useComparison();
  const hotParam = useStore((s) => s.hoverParam);
  const mode = useStore((s) => s.mode);
  const editSlot = useStore((s) => s.editSlot);
  const slots = useStore((s) => s.slots);
  const prevAlgoId = useStore((s) => s.prevAlgoId);
  const baseline = useStore((s) => s.baseline);
  const [showDiff, setShowDiff] = useState(true);
  const exp = slots[mode === 'versus' ? editSlot : 0].exp;
  const meta = algoMeta(exp.algoId);
  // the algorithm to diff the equation against
  const fromId = mode === 'versus' ? slots[editSlot === 0 ? 1 : 0].exp.algoId : baseline?.exp.algoId ?? prevAlgoId;
  const fromMeta = fromId && fromId !== meta.id ? algoMeta(fromId) : null;

  return (
    <aside className="insight col" aria-label="Insight">
      <DiffPanel main={main} other={other} versus={versus} />
      {fromMeta && <WhatChanged from={fromMeta} to={meta} versus={versus} />}
      <ChartsPanel main={main} other={other} />
      <section className="panel">
        <h2 className="section-title">
          Update rule <span className="tag">{meta.short}</span>
        </h2>
        <EquationDiff from={fromMeta} to={meta} hotParam={hotParam} />
        {fromMeta && fromMeta.equation.filter((t) => meta.equation.some((u) => u.id === t.id)).length >= 4 && (
          <p className="small muted" style={{ margin: '4px 0 0' }}>Glowing parts are new compared with {fromMeta.short}; yellow parts changed.</p>
        )}
      </section>
      <section className="panel">
        <h2 className="section-title">
          Pseudocode
          {fromMeta && (
            <label className="small" style={{ fontFamily: 'var(--font-body)', letterSpacing: 0, textTransform: 'none', display: 'flex', gap: 6, alignItems: 'center' }}>
              <input type="checkbox" checked={showDiff} onChange={(e) => setShowDiff(e.target.checked)} /> diff vs {fromMeta.short}
            </label>
          )}
        </h2>
        <PseudoPanel meta={meta} from={fromMeta} showDiff={showDiff} hotParam={hotParam} slot={mode === 'versus' ? editSlot : 0} />
      </section>
      <SweepPanel />
      <section className="panel">
        <h2 className="section-title">Backup diagram</h2>
        <BackupDiagram spec={meta.backup} />
      </section>
      <section className="panel explainer">
        <h2 className="section-title">{meta.name}</h2>
        <p>{meta.summary}</p>
        <ul>
          {meta.watchFor.map((w) => (
            <li key={w}>{w}</li>
          ))}
        </ul>
        <div className="ref">{meta.reference}</div>
      </section>
    </aside>
  );
}

// ---------------------------------------------------------------------------------- diff chips

function DiffPanel({ main, other, versus }: { main: Side | null; other: Side | null; versus: boolean }) {
  const ep = useStore((s) => s.ep);
  const t = useStore((s) => s.t);
  if (!main) return null;
  if (!other) {
    return (
      <section className="panel">
        <h2 className="section-title">What changed</h2>
        <p className="small muted" style={{ margin: 0 }}>
          {versus ? 'Waiting for the other player’s run…' : 'Change any knob, switch the cartridge, or edit the level. The current run becomes the baseline, and every difference shows up here and on the board.'}
        </p>
        <div className="chips" style={{ marginTop: 8 }}>
          <span className="chip neutral">Score {fmt(main.res.score)}</span>
          {main.res.converged >= 0 && main.res.kind === 'grid' && !main.res.planner && <span className="chip neutral">Settles at ep {main.res.converged + 1}</span>}
        </div>
      </section>
    );
  }
  const chips: { text: string; tone: 'good' | 'bad' | 'neutral' | 'change' }[] = [];
  const a = main.res;
  const b = other.res;
  const ds = a.score - b.score;
  if (a.kind === 'bandit' && b.kind === 'bandit') {
    chips.push({ text: `Avg reward ${sign(ds)}${fmt(ds)}`, tone: tone(ds) });
    const ra = a.cumRegret[a.T - 1] ?? 0;
    const rb = b.cumRegret[b.T - 1] ?? 0;
    if (rb > 1e-9) chips.push({ text: `Regret ${sign(ra - rb)}${Math.round((100 * (ra - rb)) / rb)}%`, tone: tone(rb - ra) });
    const oa = a.pctOptimal[a.T - 1] ?? 0;
    const ob = b.pctOptimal[b.T - 1] ?? 0;
    chips.push({ text: `Best arm at end ${Math.round(oa * 100)}% vs ${Math.round(ob * 100)}%`, tone: tone(oa - ob) });
  } else if (a.kind === 'grid' && b.kind === 'grid') {
    const unit = a.planner ? 'V(start)' : 'Final return';
    chips.push({ text: `${unit} ${sign(ds)}${fmt(ds)}`, tone: Math.abs(ds) < 1e-6 ? 'neutral' : tone(ds) });
    if (a.planner && b.planner) {
      chips.push({ text: `${a.E} vs ${b.E} frames to converge`, tone: tone(b.E - a.E) });
    } else if (a.converged >= 0 && b.converged >= 0) {
      const d = b.converged - a.converged;
      chips.push({ text: d === 0 ? 'Settles at the same episode' : `Settles ${Math.abs(d)} episodes ${d > 0 ? 'sooner' : 'later'}`, tone: tone(d) });
    }
    if (main.exp.env.kind === 'grid' && other.exp.env.kind === 'grid' && main.exp.env.layout.w === other.exp.env.layout.w && main.exp.env.layout.h === other.exp.env.layout.h) {
      const fa = buildGridFrame(a, main.exp.env.layout, ep, t);
      const fb = buildGridFrame(b, other.exp.env.layout, ep, t);
      const d = diffGridFrames(fa, fb);
      chips.push({ text: `Policy differs in ${d.flipCount}/${d.liveCount} cells now`, tone: d.flipCount ? 'change' : 'neutral' });
      chips.push({ text: `Max |ΔV| now ${fmt(d.maxAbs)}`, tone: 'neutral' });
      const fe = buildGridFrame(a, main.exp.env.layout, a.E - 1, 1e9);
      const fo = buildGridFrame(b, other.exp.env.layout, b.E - 1, 1e9);
      const de = diffGridFrames(fe, fo);
      chips.push({ text: `Final policies differ in ${de.flipCount} cells`, tone: de.flipCount ? 'change' : 'neutral' });
    }
  }
  return (
    <section className="panel">
      <h2 className="section-title">
        {versus ? 'Versus' : 'What changed'}
        <span className="tag">{versus ? `${main.label.split(' ')[0]} vs ${other.label.split(' ')[0]}` : 'vs baseline'}</span>
      </h2>
      {!versus && (
        <div className="chips" style={{ marginBottom: 8 }}>
          {changeList(other.exp, main.exp).map((c) => (
            <span key={c} className="chip change">
              {c}
            </span>
          ))}
        </div>
      )}
      <div className="chips">
        {chips.map((c) => (
          <span key={c.text} className={`chip ${c.tone}`}>
            {c.tone === 'good' ? '▲ ' : c.tone === 'bad' ? '▼ ' : ''}
            {c.text}
          </span>
        ))}
      </div>
    </section>
  );
}

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(1) : v.toFixed(3));
const sign = (v: number) => (v > 0 ? '+' : '');
const tone = (v: number): 'good' | 'bad' | 'neutral' => (v > 1e-9 ? 'good' : v < -1e-9 ? 'bad' : 'neutral');

function changeList(prev: Experiment, next: Experiment): string[] {
  const out: string[] = [];
  if (prev.algoId !== next.algoId) out.push(`${algoMeta(prev.algoId).short} → ${algoMeta(next.algoId).short}`);
  const a = effectiveParams(prev);
  const b = effectiveParams(next);
  if (prev.algoId === next.algoId) {
    for (const d of algoMeta(next.algoId).params) {
      if (a[d.id] !== b[d.id]) out.push(`${d.symbol ?? d.label}: ${show(a[d.id])} → ${show(b[d.id])}`);
    }
  }
  if (prev.run.seed !== next.run.seed) out.push(`seed ${prev.run.seed} → ${next.run.seed}`);
  if (prev.run.episodes !== next.run.episodes) out.push(`episodes ${prev.run.episodes} → ${next.run.episodes}`);
  if (prev.run.avgSeeds !== next.run.avgSeeds) out.push(`averaged over ${next.run.avgSeeds} seeds`);
  if (JSON.stringify(prev.env) !== JSON.stringify(next.env)) out.push('level edited');
  if (next.tweaks.length !== prev.tweaks.length) out.push('live tweak');
  return out.length ? out : ['same settings'];
}
const show = (v: unknown) => (typeof v === 'number' ? Number(v.toPrecision(3)).toString() : String(v));

function WhatChanged({ from, to, versus }: { from: AlgoMeta; to: AlgoMeta; versus: boolean }) {
  const lines: string[] = [];
  if (from.traits.target !== to.traits.target) lines.push(`${to.name} updates toward ${to.traits.target}. ${from.name} updates toward ${from.traits.target}.`);
  if (from.traits.timing !== to.traits.timing) lines.push(`It learns ${to.traits.timing}, where ${from.name} learns ${from.traits.timing}.`);
  if (from.traits.onPolicy !== to.traits.onPolicy) lines.push(to.traits.onPolicy ? `${to.name} is on-policy: it learns the value of how it actually behaves, exploration included.` : `${to.name} is off-policy: it explores, but learns the value of the greedy policy.`);
  if (from.traits.model !== to.traits.model)
    lines.push(to.traits.model === 'given' ? `${to.name} plans with the true model of the world. It needs no experience at all.` : to.traits.model === 'learned' ? `${to.name} also learns a model and replays imagined experience from it.` : `${to.name} learns from experience only, with no model of the world.`);
  if (from.traits.learns !== to.traits.learns) lines.push(`It learns ${to.traits.learns} instead of ${from.traits.learns}.`);
  const pairKey = [from.id, to.id].sort().join('|');
  const expect = EXPECT[pairKey];
  return (
    <section className="panel what-changed">
      <h2 className="section-title">
        Cartridge swap <span className="tag">{from.short} → {to.short}</span>
      </h2>
      {lines.slice(0, 3).map((l) => (
        <p key={l} className="small">
          {l}
        </p>
      ))}
      {expect && (
        <p className="small" style={{ color: 'var(--neon-yellow)', marginBottom: 0 }}>
          Expect: {expect}
        </p>
      )}
      {!versus && <p className="small muted" style={{ margin: 0 }}>Same level, same seed: the differences on the board come from the algorithm alone.</p>}
    </section>
  );
}

const EXPECT: Record<string, string> = {
  'qlearning|sarsa': 'on Cliff Walk, Q-learning’s arrows hug the cliff while SARSA takes a safer path a row up. SARSA earns more while learning.',
  'expected-sarsa|sarsa': 'smoother learning curves, and Expected SARSA tolerates bigger step sizes.',
  'expected-sarsa|qlearning': 'nearly the same final policy. Expected SARSA is a bit more cautious near danger while ε > 0.',
  'policy-iteration|value-iteration': 'the same optimal values. Policy iteration needs fewer improvement steps but many evaluation sweeps.',
  'dyna-q|qlearning': 'Dyna-Q solves mazes in far fewer real episodes, because every real step is reused n times.',
  'monte-carlo|sarsa': 'Monte Carlo only changes values at episode ends, with noisier curves; SARSA learns mid-episode.',
  'actor-critic|reinforce': 'actor-critic learns during the episode and has much less variance.',
  'eps-greedy|ucb': 'UCB explores in a targeted way and pulls ahead after the first few hundred pulls.',
  'thompson|ucb': 'both explore smartly; Thompson often has the lowest regret.',
  'eps-greedy|thompson': 'Thompson stops wasting pulls on clearly bad arms, so regret flattens out.',
};

// ---------------------------------------------------------------------------------- charts

function ChartsPanel({ main, other }: { main: Side | null; other: Side | null }) {
  const ep = useStore((s) => s.ep);
  const t = useStore((s) => s.t);
  const [tab, setTab] = useState(0);
  const st = useStore.getState();
  if (!main) return <section className="panel"><h2 className="section-title">Learning curves</h2><p className="muted small">Computing…</p></section>;
  const r = main.res;
  const o = other?.res.kind === r.kind ? other : null;
  let tabs: { name: string; render: () => JSX.Element }[] = [];
  if (r.kind === 'bandit') {
    const ob = o?.res as BanditRunResult | undefined;
    const sm = (x: Float32Array) => movingAverage(x, Math.max(1, Math.floor(x.length / 60)));
    const mk = (pick: (b: BanditRunResult) => Float32Array, smooth: boolean): Series[] => [
      ...(ob ? [{ values: smooth ? sm(pick(ob)) : pick(ob), color: o!.color, label: o!.label, dashed: true }] : []),
      { values: smooth ? sm(pick(r)) : pick(r), color: main.color, label: main.label },
    ];
    const note = r.seedsUsed > 1 ? `mean of ${r.seedsUsed} runs` : 'single run';
    tabs = [
      { name: 'Reward', render: () => <Chart series={mk((b) => b.avgReward, true)} fillBetween={ob ? [1, 0] : undefined} cursor={t} xLabel="pull" yLabel={`avg reward (${note})`} onPick={(x) => st.setPlayhead(0, x)} ariaLabel="Average reward per pull" /> },
      { name: '% best arm', render: () => <Chart series={mk((b) => b.pctOptimal, true)} fillBetween={ob ? [1, 0] : undefined} cursor={t} xLabel="pull" yLabel="% optimal action" onPick={(x) => st.setPlayhead(0, x)} ariaLabel="Percent optimal action" /> },
      { name: 'Regret', render: () => <Chart series={mk((b) => b.cumRegret, false)} fillBetween={ob ? [1, 0] : undefined} higherIsBetter={false} cursor={t} xLabel="pull" yLabel="cumulative regret" onPick={(x) => st.setPlayhead(0, x)} ariaLabel="Cumulative regret" /> },
    ];
  } else {
    const og = o?.res as GridRunResult | undefined;
    const markers = main.exp.tweaks.map((tw) => tw.at);
    if (r.planner) {
      tabs = [
        {
          name: 'Convergence',
          render: () => (
            <Chart
              series={[...(og?.planner ? [{ values: og.deltas, color: o!.color, label: o!.label, dashed: true }] : []), { values: r.deltas, color: main.color, label: main.label }]}
              logY
              cursor={ep}
              xLabel="frame"
              yLabel="max |ΔV|"
              onPick={(x) => st.setPlayhead(x, 0)}
              ariaLabel="Largest value change per frame"
            />
          ),
        },
        {
          name: 'V(start)',
          render: () => (
            <Chart series={[...(og ? [{ values: og.returns, color: o!.color, label: o!.label, dashed: true }] : []), { values: r.returns, color: main.color, label: main.label }]} cursor={ep} xLabel="frame" yLabel="V(start)" onPick={(x) => st.setPlayhead(x, 0)} ariaLabel="Value of the start state" />
          ),
        },
      ];
    } else {
      const w = Math.max(1, Math.floor(r.E / 40));
      const sm = (x: ArrayLike<number>) => movingAverage(x, w);
      const ogAgent = og && !og.planner ? og : undefined;
      const note = r.seedsUsed > 1 ? `, mean of ${r.seedsUsed} seeds` : '';
      tabs = [
        {
          name: 'Return',
          render: () => (
            <Chart
              series={[
                { values: r.avgReturns, color: main.color, label: 'raw', alpha: 0.25, width: 1 },
                ...(ogAgent ? [{ values: sm(ogAgent.avgReturns), color: o!.color, label: o!.label, dashed: true }] : []),
                { values: sm(r.avgReturns), color: main.color, label: main.label },
              ]}
              fillBetween={ogAgent ? [2, 1] : undefined}
              cursor={ep}
              markers={markers}
              xLabel="episode"
              yLabel={`return (avg ${w}${note})`}
              onPick={(x) => st.setPlayhead(x, 0)}
              ariaLabel="Return per episode"
            />
          ),
        },
        {
          name: 'Steps',
          render: () => (
            <Chart
              series={[...(ogAgent ? [{ values: sm(ogAgent.avgLengths), color: o!.color, label: o!.label, dashed: true }] : []), { values: sm(r.avgLengths), color: main.color, label: main.label }]}
              fillBetween={ogAgent ? [1, 0] : undefined}
              higherIsBetter={false}
              cursor={ep}
              markers={markers}
              xLabel="episode"
              yLabel="steps per episode"
              onPick={(x) => st.setPlayhead(x, 0)}
              ariaLabel="Steps per episode"
            />
          ),
        },
      ];
    }
  }
  const cur = Math.min(tab, tabs.length - 1);
  return (
    <section className="panel">
      <h2 className="section-title">
        Learning curves <span className="tag">click to jump</span>
      </h2>
      <div className="tabs" role="tablist">
        {tabs.map((tb, i) => (
          <button key={tb.name} type="button" role="tab" aria-selected={i === cur} className={`btn small${i === cur ? ' on' : ''}`} onClick={() => setTab(i)}>
            {tb.name}
          </button>
        ))}
      </div>
      {tabs[cur].render()}
    </section>
  );
}

// ---------------------------------------------------------------------------------- pseudocode with live line

function PseudoPanel({ meta, from, showDiff, hotParam, slot }: { meta: AlgoMeta; from: AlgoMeta | null; showDiff: boolean; hotParam: string | null; slot: 0 | 1 }) {
  const ep = useStore((s) => s.ep);
  const t = useStore((s) => s.t);
  const r = useStore((s) => s.slots[slot].res);
  let current = -1;
  if (r && r.algoId === meta.id) {
    if (r.kind === 'bandit') current = t > 0 ? r.lines[t - 1] : -1;
    else if (r.planner) {
      const ph = r.phases[Math.min(ep, r.E - 1)];
      const id = meta.id === 'policy-iteration' ? (ph === 1 ? 'improve' : 'bellman-pi') : 'bellman';
      current = meta.pseudocode.findIndex((l) => l.id === id);
    } else if (ep < r.E && t > 0) {
      const j = r.epStart[ep] + Math.min(t, r.lengths[ep]) - 1;
      current = r.stLine[j] ?? -1;
      if (current === 255) current = -1;
    }
  }
  return <Pseudocode from={from} to={meta} current={current} hotParam={hotParam} showDiff={showDiff} />;
}

// ---------------------------------------------------------------------------------- sensitivity sweep

function SweepPanel() {
  const sweep = useStore((s) => s.sweep);
  const slots = useStore((s) => s.slots);
  const st = useStore.getState();
  if (!sweep) return null;
  const exp = slots[sweep.slot].exp;
  const meta = algoMeta(exp.algoId);
  const d = meta.params.find((p) => p.id === sweep.paramId);
  if (!d) return null;
  const cur = effectiveParams(exp)[d.id];
  const scores = sweep.points.map((p) => p.result.score);
  return (
    <section className="panel">
      <h2 className="section-title">
        Sweep <span className="tag">{d.symbol ?? d.label}</span>
        <button type="button" className="mini" onClick={() => st.set({ sweep: null })}>
          Close
        </button>
      </h2>
      {sweep.status === 'running' && (
        <p className="small muted">
          Running {sweep.done}/{sweep.total} settings…
        </p>
      )}
      {sweep.status === 'error' && <p className="error">The sweep failed: {sweep.message}</p>}
      {sweep.status === 'done' && (
        <>
          <Chart
            series={[{ values: scores, color: 'var(--neon-yellow)', label: `score by ${d.label}` }]}
            xFormat={(x) => show(sweep.points[Math.round(x)]?.value)}
            cursor={sweep.points.findIndex((p) => p.value === cur) >= 0 ? sweep.points.findIndex((p) => p.value === cur) : undefined}
            xLabel={d.symbol ?? d.label}
            yLabel={exp.env.kind === 'bandit' ? 'avg reward' : 'final return'}
            height={140}
            onPick={(x) => {
              const p = sweep.points[x];
              if (p) st.setParam(sweep.slot, d.id, p.value);
            }}
            ariaLabel={`Score for each value of ${d.label}`}
          />
          <p className="small muted" style={{ margin: '4px 0 0' }}>
            {exp.env.kind === 'grid' ? 'Each setting averaged over 3 seeds. ' : ''}Click a tile to try that value.
          </p>
          <div className="sweep-grid">
            {sweep.points.map((p) => (
              <button key={String(p.value)} type="button" className={`sweep-cell${p.value === cur ? ' cur' : ''}`} onClick={() => st.setParam(sweep.slot, d.id, p.value)}>
                {exp.env.kind === 'grid' && p.result.finalV && <Thumb exp={exp} V={p.result.finalV} best={p.result.finalBest} />}
                <div>{show(p.value)}</div>
                <div style={{ color: 'var(--neon-yellow)' }}>{fmt(p.result.score)}</div>
              </button>
            ))}
          </div>
        </>
      )}
    </section>
  );
}

function Thumb({ exp, V, best }: { exp: Experiment; V: Float32Array; best: Uint8Array | null }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || exp.env.kind !== 'grid') return;
    const L = exp.env.layout;
    const C = Math.max(2, Math.floor(64 / Math.max(L.w, L.h)));
    c.width = L.w * C;
    c.height = L.h * C;
    const ctx = c.getContext('2d')!;
    const th = readTheme();
    let m = 1e-6;
    for (let i = 0; i < V.length; i++) m = Math.max(m, Math.abs(V[i]));
    for (let s = 0; s < L.cells.length; s++) {
      const cell = L.cells[s];
      const x = (s % L.w) * C;
      const y = Math.floor(s / L.w) * C;
      ctx.fillStyle = cell.t === 'wall' ? rgba(th.wall) : cell.t === 'goal' ? rgba(th.good) : cell.t === 'lava' ? rgba(th.orange) : cell.t === 'cliff' ? rgba(th.bg) : rgba(diverging(th, V[s], m));
      ctx.fillRect(x, y, C, C);
      if (best && C >= 6 && best[s] && best[s] !== 15 && cell.t !== 'wall' && cell.t !== 'goal' && cell.t !== 'lava' && cell.t !== 'cliff') {
        ctx.fillStyle = rgba(th.yellow);
        const a = [0, 1, 2, 3].find((k) => best[s] & (1 << k)) ?? 0;
        const cx = x + C / 2 + [0, 1, 0, -1][a] * C * 0.25;
        const cy = y + C / 2 + [-1, 0, 1, 0][a] * C * 0.25;
        ctx.fillRect(cx - 1, cy - 1, 2, 2);
      }
    }
  }, [exp, V, best]);
  return <canvas ref={ref} aria-hidden="true" />;
}
