import { useEffect, useMemo, useRef, useState } from 'react';
import { fromAscii, LEVELS } from '../core/envs/presets';
import { runGrid, type GridRunResult } from '../core/engine';
import type { Experiment, GridLayout } from '../core/types';
import { ALGORITHMS, FAMILY_LABELS } from '../core/registry';
import { PLAIN } from '../core/plain';
import { buildGridFrame } from '../viz/frame';
import { drawGrid } from '../viz/drawGrid';
import { readTheme } from '../viz/theme';
import type { Family } from '../core/types';

// A tiny level for the live demo: lava pits between the robot and the flag.
const DEMO_LAYOUT: GridLayout = fromAscii(['S..L....', '.#.L.##.', '.#...#..', '...L...G'], {
  stepReward: -1,
  bumpReward: -1,
  goalReward: 10,
  lavaReward: -20,
});
const DEMO_EXP: Experiment = {
  env: { kind: 'grid', layout: DEMO_LAYOUT },
  algoId: 'qlearning',
  params: { alpha: 0.5, gamma: 0.95, epsilon: 0.1 },
  run: { episodes: 120, maxSteps: 60, seed: 12, avgSeeds: 1 },
  tweaks: [],
};
const SHOWCASE = [0, 3, 12, 40, 119];

function LiveDemo() {
  const res = useMemo<GridRunResult>(() => runGrid(DEMO_EXP), []);
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const [caption, setCaption] = useState({ k: 0, steps: 0, ret: 0, done: false });

  useEffect(() => {
    const c = cv.current;
    const w = wrap.current;
    if (!c || !w) return;
    const reduce = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
    const ctx = c.getContext('2d')!;
    let raf = 0;
    let k = reduce ? SHOWCASE.length - 1 : 0;
    let t = reduce ? 1e9 : 0;
    let last = performance.now();
    let acc = 0;
    let hold = 0;
    let lastCaption = '';
    const L = DEMO_LAYOUT;
    let scale = Math.max(1, Math.max(...Array.from(res.snapV, (v) => Math.abs(v))));
    scale = Math.max(scale, 1);
    const loop = (now: number) => {
      const dt = Math.min(100, now - last);
      last = now;
      const ep = SHOWCASE[k];
      const len = res.lengths[ep];
      if (!reduce) {
        if (t < len) {
          // each try takes at most ~3 s, however long it is
          acc += (dt / 1000) * Math.max(8, len / 3);
          const n = Math.floor(acc);
          acc -= n;
          t = Math.min(len, t + n);
        } else {
          hold += dt;
          if (hold > 1400) {
            hold = 0;
            t = 0;
            k = (k + 1) % SHOWCASE.length;
          }
        }
      }
      const cell = Math.max(18, Math.min(58, Math.floor(w.clientWidth / L.w)));
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      const W = cell * L.w;
      const H = cell * L.h;
      if (c.width !== Math.round(W * dpr)) {
        c.width = Math.round(W * dpr);
        c.height = Math.round(H * dpr);
        c.style.width = `${W}px`;
        c.style.height = `${H}px`;
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const frame = buildGridFrame(res, L, SHOWCASE[k], t);
      drawGrid(ctx, {
        frame,
        ghost: null,
        diff: null,
        overlays: { values: true, q: false, arrows: true, visits: false, trail: true, ghost: false, diff: false, numbers: false },
        theme: readTheme(),
        cell,
        time: now,
        frac: 1,
        policyGlyphs: false,
        traces: null,
        nWindow: 0,
        horizon: 0,
        hoverCell: -1,
        reducedMotion: !!reduce,
        valueScale: scale,
        playerColor: readTheme().text,
      });
      const ep2 = SHOWCASE[k];
      const tt = Math.min(t, res.lengths[ep2]);
      let ret = 0;
      for (let j = res.epStart[ep2]; j < res.epStart[ep2] + tt; j++) ret += res.stR[j];
      const key = `${k}|${tt}`;
      if (key !== lastCaption) {
        lastCaption = key;
        setCaption({ k, steps: tt, ret, done: tt >= res.lengths[ep2] });
      }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [res]);

  const ep = SHOWCASE[caption.k];
  const words = ['knows nothing yet and wanders around', 'is starting to avoid the lava', 'has found a way to the flag', 'takes a fairly direct route', 'goes almost straight to the flag'];
  return (
    <figure className="demo">
      <div className="demo-head">
        <span className="demo-try">TRY #{ep + 1}</span>
        <span className="demo-dots" aria-hidden="true">
          {SHOWCASE.map((_, i) => (
            <i key={i} className={i === caption.k ? 'on' : i < caption.k ? 'past' : ''} />
          ))}
        </span>
      </div>
      <div ref={wrap} className="demo-screen">
        <canvas ref={cv} role="img" aria-label="A robot learning to reach the flag while avoiding lava" />
      </div>
      <figcaption className="demo-caption" aria-live="polite">
        The robot {words[caption.k]}. {caption.steps} {caption.steps === 1 ? 'step' : 'steps'}, {caption.ret >= 0 ? '+' : ''}
        {caption.ret} {Math.abs(caption.ret) === 1 ? 'point' : 'points'}
        {caption.done ? '' : ' so far'}.
      </figcaption>
    </figure>
  );
}

const FAMILIES: { id: Family; plain: string }[] = [
  { id: 'bandit', plain: 'Slot machines: which one pays best?' },
  { id: 'dp', plain: 'Planners that already know the map' },
  { id: 'tabular', plain: 'Learners that figure out the map by moving around' },
  { id: 'policy-gradient', plain: 'Learners that tune their habits directly' },
];

export function Landing({ onPlay }: { onPlay: (levelId?: string) => void }) {
  useEffect(() => {
    window.scrollTo(0, 0);
  }, []);
  return (
    <div className="landing">
      <header className="land-top">
        <span className="logo" aria-label="RL Arcade">
          RL <span>ARCADE</span>
        </span>
        <nav className="land-nav" aria-label="Page sections">
          <a href="#what">What is it?</a>
          <a href="#how">How to use</a>
          <a href="#methods">Methods</a>
          <button type="button" className="btn primary" onClick={() => onPlay()}>
            Start playing
          </button>
        </nav>
      </header>

      <section className="hero">
        <div className="hero-text">
          <p className="eyebrow">Reinforcement learning you can play with</p>
          <h1>Watch a computer learn by trial and error</h1>
          <p className="lead">
            RL Arcade lets you watch learning programs play small games. They start knowing nothing. They earn points for good moves, lose points for bad ones,
            and slowly work out how to win.
          </p>
          <p className="lead">Change any setting and you see exactly what your change did.</p>
          <div className="row">
            <button type="button" className="btn primary big" onClick={() => onPlay()}>
              Start playing
            </button>
            <a className="btn big ghost-btn" href="#how">
              How it works
            </a>
          </div>
        </div>
        <LiveDemo />
      </section>

      <section className="land-section" id="what">
        <h2>What is reinforcement learning?</h2>
        <p className="section-lead">It’s like training a dog with treats, or getting better at a video game by playing it over and over. Three ideas are all you need:</p>
        <div className="cards3">
          <article className="land-card">
            <h3>The player</h3>
            <p>A little robot, called the agent. It can move up, down, left or right. At first it moves at random.</p>
          </article>
          <article className="land-card">
            <h3>Points</h3>
            <p>Reaching the flag earns points. Lava and wasted steps cost points. Nobody shows it the right path; points are its only clue.</p>
          </article>
          <article className="land-card">
            <h3>Learning</h3>
            <p>After each move it updates its notes on which squares and moves led to points. After enough tries, its notes are good enough to win.</p>
          </article>
        </div>
      </section>

      <section className="land-section" id="how">
        <h2>How to use RL Arcade</h2>
        <ol className="steps">
          <li>
            <p><b>Pick a level.</b> Use the Level menu at the top: slot machines, a cliff edge, a windy valley, a maze, and more.</p>
          </li>
          <li>
            <p><b>Pick a cartridge.</b> Each cartridge on the left is a different learning method. The panel on the right explains it in simple words.</p>
          </li>
          <li>
            <p><b>Press Start.</b> Watch the robot try again and again. Use the speed slider, or drag the timeline to jump to any try.</p>
          </li>
          <li>
            <p><b>Change a knob.</b> Move any slider. Your previous run stays on the board as a faded ghost, so you can see what your change did.</p>
          </li>
          <li>
            <p><b>Try 2P Versus.</b> Put two methods side by side on the same level, with the same luck, and see which one learns better.</p>
          </li>
        </ol>
      </section>

      <section className="land-section">
        <h2>Reading the board</h2>
        <ul className="legend-list">
          <li>
            <span className="sw" style={{ background: 'var(--val-pos)' }} />
            <p>
              <b>Blue squares</b> are good places to be: the robot expects to earn points from there.
            </p>
          </li>
          <li>
            <span className="sw" style={{ background: 'var(--val-neg)' }} />
            <p>
              <b>Pink squares</b> are bad places: it expects to lose points.
            </p>
          </li>
          <li>
            <span className="sw arrow" aria-hidden="true">
              ➜
            </span>
            <p>
              <b>Amber arrows</b> show the move it would choose right now.
            </p>
          </li>
          <li>
            <span className="sw spark" aria-hidden="true">
              ✚
            </span>
            <p>
              <b>Pink sparks</b> mark random moves it made just to explore.
            </p>
          </li>
          <li>
            <span className="sw" style={{ background: 'var(--ghost)', opacity: 0.6 }} />
            <p>
              <b>Faded robot and arrows</b> are your previous run, kept for comparison.
            </p>
          </li>
          <li>
            <span className="sw outline" />
            <p>
              <b>Outlined squares</b> are where the new run would act differently from the old one.
            </p>
          </li>
        </ul>
      </section>

      <section className="land-section">
        <h2>Pick a level to start</h2>
        <div className="level-cards">
          {LEVELS.map((l) => (
            <button key={l.id} type="button" className="level-card" onClick={() => onPlay(l.id)}>
              <span className="lc-name">{l.name}</span>
              <span className="lc-text">{l.challenge}</span>
              <span className="lc-go">Play ›</span>
            </button>
          ))}
        </div>
      </section>

      <section className="land-section" id="methods">
        <h2>The learning methods</h2>
        <p className="section-lead">Fifteen methods in four families. Each one is explained like this inside the app too.</p>
        {FAMILIES.map((f) => (
          <div key={f.id} className="family-block">
            <h3>
              {FAMILY_LABELS[f.id]} <span className="muted">· {f.plain}</span>
            </h3>
            <div className="method-grid">
              {ALGORITHMS.filter((a) => a.meta.family === f.id).map((a) => (
                <article key={a.meta.id} className="method">
                  <h4>{a.meta.name}</h4>
                  <p>{PLAIN[a.meta.id]?.simple}</p>
                </article>
              ))}
            </div>
          </div>
        ))}
      </section>

      <section className="land-cta">
        <h2>Ready?</h2>
        <p>Start with Cliff Walk: two methods, one cliff, two very different personalities.</p>
        <button type="button" className="btn primary big" onClick={() => onPlay('cliff-walk')}>
          Play Cliff Walk
        </button>
      </section>

      <footer className="land-foot">
        <p>
          Every number is computed live in your browser. Nothing is pre-recorded. Built with TypeScript and React, following Sutton &amp; Barto’s{' '}
          <i>Reinforcement Learning: An Introduction</i>.
        </p>
        <p>
          <a href="https://github.com/ZamoRzgar/rl-arcade" target="_blank" rel="noreferrer">
            Source code on GitHub
          </a>
        </p>
      </footer>
    </div>
  );
}
