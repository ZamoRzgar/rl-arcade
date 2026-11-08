import { useState } from 'react';
import { ALGORITHMS, FAMILY_LABELS, algoMeta } from '../core/registry';
import type { Family } from '../core/types';
import { effectiveParams, useStore, type Tool } from '../state/store';
import { ParamControl } from './ParamControl';
import { getLevel } from '../core/envs/presets';

const FAMILIES: Family[] = ['bandit', 'dp', 'tabular', 'policy-gradient'];

const TOOLS: { id: Tool; label: string; color: string }[] = [
  { id: 'empty', label: 'Floor', color: 'var(--screen)' },
  { id: 'wall', label: 'Wall', color: 'var(--wall)' },
  { id: 'start', label: 'Start', color: 'var(--neon-yellow)' },
  { id: 'goal', label: 'Goal', color: 'var(--good)' },
  { id: 'lava', label: 'Lava', color: 'var(--neon-orange)' },
  { id: 'cliff', label: 'Cliff', color: 'var(--neon-pink)' },
  { id: 'ice', label: 'Ice', color: 'var(--ice)' },
  { id: 'wind', label: 'Wind', color: 'var(--text-dim)' },
  { id: 'teleport', label: 'Portal', color: 'var(--neon-cyan)' },
  { id: 'erase', label: 'Erase', color: 'transparent' },
];

export function ControlDeck() {
  const mode = useStore((s) => s.mode);
  const editSlot = useStore((s) => s.editSlot);
  const slots = useStore((s) => s.slots);
  const levelId = useStore((s) => s.levelId);
  const hoverParam = useStore((s) => s.hoverParam);
  const changeMode = useStore((s) => s.changeMode);
  const st = useStore.getState();
  const slotIdx = mode === 'versus' ? editSlot : 0;
  const exp = slots[slotIdx].exp;
  const meta = algoMeta(exp.algoId);
  const envKind = exp.env.kind;
  const params = effectiveParams(exp);
  const level = getLevel(levelId);
  const preset = slotIdx === 0 ? level.A : level.B;
  const [inserted, setInserted] = useState<string | null>(null);

  return (
    <aside className="deck col" aria-label="Controls">
      <section className="panel">
        {mode === 'versus' && (
          <div className="slot-tabs" role="group" aria-label="Which player to edit">
            <button type="button" className="slot-tab p1" aria-pressed={editSlot === 0} onClick={() => st.setEditSlot(0)}>
              P1 · {algoMeta(slots[0].exp.algoId).short}
            </button>
            <button type="button" className="slot-tab p2" aria-pressed={editSlot === 1} onClick={() => st.setEditSlot(1)}>
              P2 · {algoMeta(slots[1].exp.algoId).short}
            </button>
          </div>
        )}
        <h2 className="section-title">
          Cartridge <span className="tag">{meta.short}</span>
        </h2>
        {FAMILIES.map((fam) => {
          const algos = ALGORITHMS.filter((a) => a.meta.family === fam);
          return (
            <div className="family" key={fam}>
              <div className="family-name">{FAMILY_LABELS[fam]}</div>
              <div className="carts">
                {algos.map((a) => {
                  const ok = a.meta.envKind === envKind;
                  const why = ok ? a.meta.name : a.meta.envKind === 'bandit' ? `${a.meta.name}: needs a bandit level (try Bandit Alley)` : `${a.meta.name}: needs a grid level`;
                  return (
                    <button
                      key={a.meta.id}
                      type="button"
                      className={`cart${inserted === a.meta.id ? ' inserted' : ''}`}
                      aria-pressed={exp.algoId === a.meta.id}
                      disabled={!ok}
                      title={why}
                      onClick={() => {
                        st.setAlgo(slotIdx, a.meta.id);
                        setInserted(a.meta.id);
                        window.setTimeout(() => setInserted(null), 300);
                      }}
                    >
                      {a.meta.short}
                    </button>
                  );
                })}
              </div>
            </div>
          );
        })}
      </section>

      <section className="panel">
        <h2 className="section-title">
          Knobs
          <span className="seg" role="group" aria-label="What a change does">
            <button
              type="button"
              className="btn small"
              aria-pressed={changeMode === 'rerun'}
              title="Rerun: a change replays the whole run from episode 0 with the same seed"
              onClick={() => st.set({ changeMode: 'rerun' })}
            >
              Rerun
            </button>
            <button
              type="button"
              className="btn small"
              aria-pressed={changeMode === 'live'}
              title="Live: a change applies from the current episode onward"
              onClick={() => st.set({ changeMode: 'live' })}
            >
              Live
            </button>
          </span>
        </h2>
        {meta.params.map((d) => (
          <ParamControl
            key={`${meta.id}-${d.id}`}
            def={d}
            value={params[d.id] ?? d.default}
            changed={preset.algoId === meta.id && preset.params[d.id] !== undefined ? preset.params[d.id] !== params[d.id] : params[d.id] !== d.default}
            hot={hoverParam === d.id}
            onChange={(v) => st.setParam(slotIdx, d.id, v)}
            onReset={() => st.resetParam(slotIdx, d.id)}
            onHover={(on) => st.set({ hoverParam: on ? d.id : null })}
            onSweep={() => st.startSweep(slotIdx, d.id)}
          />
        ))}
        {exp.tweaks.length > 0 && (
          <p className="small muted">
            {exp.tweaks.length} live tweak{exp.tweaks.length > 1 ? 's' : ''} applied (dashed markers on the charts).{' '}
            <button className="mini" type="button" onClick={() => st.clearTweaks(slotIdx)}>
              Remove tweaks
            </button>
          </p>
        )}
      </section>

      <RunPanel />
      {envKind === 'grid' ? <EditorPanel /> : <BanditPanel />}
    </aside>
  );
}

function RunPanel() {
  const exp = useStore((s) => s.slots[0].exp);
  const st = useStore.getState();
  const planner = algoMeta(exp.algoId).family === 'dp';
  const bandit = exp.env.kind === 'bandit';
  return (
    <section className="panel">
      <h2 className="section-title">Run settings</h2>
      <div className="kv">
        {!bandit && (
          <>
            <label htmlFor="run-episodes">{planner ? 'Max sweeps' : 'Episodes'}</label>
            <input id="run-episodes" type="number" min={1} max={5000} value={exp.run.episodes} onChange={(e) => st.setRun({ episodes: Math.max(1, Math.min(5000, Math.round(Number(e.target.value) || 1))) })} />
            <label htmlFor="run-max">Max steps / episode</label>
            <input id="run-max" type="number" min={1} max={5000} value={exp.run.maxSteps} onChange={(e) => st.setRun({ maxSteps: Math.max(1, Math.min(5000, Math.round(Number(e.target.value) || 1))) })} />
          </>
        )}
        <label htmlFor="run-seed">Seed</label>
        <span className="row">
          <input id="run-seed" type="number" value={exp.run.seed} onChange={(e) => st.setRun({ seed: Math.round(Number(e.target.value) || 0) })} />
          <button type="button" className="btn small" title="New random seed" onClick={() => st.setRun({ seed: Math.floor(Math.random() * 100000) })}>
            Reroll
          </button>
        </span>
        <label htmlFor="run-avg" title="Learning curves show the mean over this many seeds">
          Average curves over
        </label>
        <span className="row">
          <input id="run-avg" type="number" min={1} max={bandit ? 500 : 30} value={exp.run.avgSeeds} onChange={(e) => st.setRun({ avgSeeds: Math.max(1, Math.min(bandit ? 500 : 30, Math.round(Number(e.target.value) || 1))) })} />
          <span className="small muted">seeds</span>
        </span>
      </div>
    </section>
  );
}

function BanditPanel() {
  const exp = useStore((s) => s.slots[0].exp);
  const showTruth = useStore((s) => s.showTruth);
  const st = useStore.getState();
  if (exp.env.kind !== 'bandit') return null;
  const b = exp.env.bandit;
  return (
    <section className="panel">
      <h2 className="section-title">Slot machines</h2>
      <div className="kv">
        <label htmlFor="b-k">Arms</label>
        <input id="b-k" type="number" min={2} max={20} value={b.k} onChange={(e) => st.setBanditK(Math.max(2, Math.min(20, Math.round(Number(e.target.value) || 2))))} />
        <label htmlFor="b-steps">Pulls</label>
        <input id="b-steps" type="number" min={50} max={5000} value={b.steps} onChange={(e) => st.setBanditSteps(Math.max(50, Math.min(5000, Math.round(Number(e.target.value) || 50))))} />
        <label htmlFor="b-dist">Payouts</label>
        <select id="b-dist" value={b.dist} onChange={(e) => st.setBanditDist(e.target.value as 'gaussian' | 'bernoulli')} style={{ background: 'var(--screen)', border: '2px solid var(--line)', padding: 4 }}>
          <option value="gaussian">Gaussian (testbed)</option>
          <option value="bernoulli">Win / lose (Bernoulli)</option>
        </select>
        <label htmlFor="b-truth">Reveal true means</label>
        <input id="b-truth" type="checkbox" checked={showTruth} onChange={(e) => st.set({ showTruth: e.target.checked })} />
      </div>
    </section>
  );
}

function EditorPanel() {
  const tool = useStore((s) => s.tool);
  const windDir = useStore((s) => s.windDir);
  const exp = useStore((s) => s.slots[0].exp);
  const undo = useStore((s) => s.undo.length);
  const redo = useStore((s) => s.redo.length);
  const st = useStore.getState();
  if (exp.env.kind !== 'grid') return null;
  const L = exp.env.layout;
  const setNum = (k: 'stepReward' | 'goalReward' | 'lavaReward' | 'slip' | 'iceSlip', v: number) => st.setLayout({ ...L, [k]: v });
  return (
    <section className="panel">
      <h2 className="section-title">
        Level editor
        {tool && (
          <button type="button" className="mini" onClick={() => st.set({ tool: null, teleportFrom: null })}>
            Done
          </button>
        )}
      </h2>
      <div className="tools" role="group" aria-label="Paint tools">
        {TOOLS.map((t) => (
          <button key={t.id} type="button" className="tool" aria-pressed={tool === t.id} onClick={() => st.set({ tool: tool === t.id ? null : t.id, teleportFrom: null })}>
            <span className="sw" style={{ background: t.color, borderStyle: t.id === 'erase' ? 'dashed' : 'solid', borderColor: 'var(--text-dim)' }} />
            {t.label}
          </button>
        ))}
      </div>
      {tool === 'wind' && (
        <div className="row" style={{ marginTop: 8 }}>
          <span className="small muted">Direction</span>
          {['↑', '→', '↓', '←'].map((a, i) => (
            <button key={a} type="button" className="btn small" aria-pressed={windDir === i} onClick={() => st.set({ windDir: i })}>
              {a}
            </button>
          ))}
          <span className="small muted">Click twice for strength 2</span>
        </div>
      )}
      <p className="small muted" style={{ margin: '8px 0' }}>
        {tool ? 'Click or drag on the arena to paint. Every edit reruns both players.' : 'Pick a tool, then paint on the arena.'}
      </p>
      <div className="row">
        <button type="button" className="btn small" disabled={!undo} onClick={() => st.undoEdit()}>
          Undo
        </button>
        <button type="button" className="btn small" disabled={!redo} onClick={() => st.redoEdit()}>
          Redo
        </button>
        <span className="small muted">Size</span>
        <button type="button" className="btn small" onClick={() => st.resize(L.w - 1, L.h)} aria-label="Narrower">
          W−
        </button>
        <button type="button" className="btn small" onClick={() => st.resize(L.w + 1, L.h)} aria-label="Wider">
          W+
        </button>
        <button type="button" className="btn small" onClick={() => st.resize(L.w, L.h - 1)} aria-label="Shorter">
          H−
        </button>
        <button type="button" className="btn small" onClick={() => st.resize(L.w, L.h + 1)} aria-label="Taller">
          H+
        </button>
      </div>
      <div className="kv" style={{ marginTop: 10 }}>
        <label htmlFor="r-step">Reward per step</label>
        <input id="r-step" type="number" step={0.5} value={L.stepReward} onChange={(e) => setNum('stepReward', Number(e.target.value) || 0)} />
        <label htmlFor="r-goal">Goal bonus</label>
        <input id="r-goal" type="number" step={1} value={L.goalReward} onChange={(e) => setNum('goalReward', Number(e.target.value) || 0)} />
        <label htmlFor="r-lava">Lava penalty</label>
        <input id="r-lava" type="number" step={1} value={L.lavaReward} onChange={(e) => setNum('lavaReward', Number(e.target.value) || 0)} />
        <label htmlFor="r-slip">Slip everywhere</label>
        <input id="r-slip" type="number" step={0.05} min={0} max={1} value={L.slip} onChange={(e) => setNum('slip', Math.max(0, Math.min(1, Number(e.target.value) || 0)))} />
        <label htmlFor="r-ice">Slip on ice</label>
        <input id="r-ice" type="number" step={0.05} min={0} max={1} value={L.iceSlip} onChange={(e) => setNum('iceSlip', Math.max(0, Math.min(1, Number(e.target.value) || 0)))} />
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>
        {L.w}×{L.h} grid · {L.cells.filter((c) => c.t === 'wall').length} walls
      </p>
    </section>
  );
}
