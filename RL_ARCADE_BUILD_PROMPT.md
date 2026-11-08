# Build Prompt: RL Arcade — an interactive Reinforcement Learning sandbox

> Paste everything below this line into your AI coding agent (Claude Code, Cursor, etc.) inside an **empty project folder**. Save a copy as `SPEC.md` in the repo so the agent can re-read it in later sessions.

---

## 0. Your role

You are a senior AI/ML software engineer and frontend architect. You have shipped production TypeScript apps and you know reinforcement learning at the level of Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.). You write correct, tested, typed code. You plan before you build, you build in small verified increments, and you never present unfinished or unverified work as done.

You are building **RL Arcade**, a browser-based sandbox where people can run reinforcement learning algorithms, change any parameter or switch algorithms, and **see exactly what changed and why**, all in an original retro-arcade visual style.

---

## 1. Product context

- **Owner and purpose:** a personal portfolio project. It must impress a technical recruiter or ML engineer within 30 seconds of opening the link, and hold up when an expert pokes at it.
- **Audience:** ML engineers, students of RL, recruiters. Assume visitors know some ML but maybe not RL.
- **The page's single job:** let a visitor build intuition for RL by changing things and *watching the consequence*.
- **Core promise (the differentiator):** every change the user makes is compared against a baseline:
  1. **Change a number** (γ, α, ε, n, λ, …) → see the old run as a ghost next to the new run, plus a visual diff of values, policy and learning curve.
  2. **Switch the algorithm** → see the update equation and pseudocode *morph* from one to the other (like a code diff), and both algorithms running side by side on the same environment with the same random seed.
- **Delivery:** a static web app (no backend), deployed to a public URL, installable as a PWA.

---

## 2. Non-negotiable principles

1. **Correctness first.** A beautiful visualization of a subtly wrong algorithm is worse than nothing. Every algorithm has reference tests (section 13).
2. **Determinism.** Same experiment config + same seed → bit-identical results, always.
3. **Fair comparisons.** When two runs are compared, the *only* difference between them must be what the user changed (see "common random numbers", section 7).
4. **Never block the UI.** All simulation runs in a Web Worker. The main thread only renders.
5. **One interface, many algorithms.** Adding an algorithm = one file + metadata + tests. It then appears automatically in the UI, the diff system, and the comparison mode.
6. **Show internals, not just behavior.** Each algorithm must expose its *signature quantity* (the thing that makes it different) and the UI must visualize it.
7. **Explain in plain language.** Every parameter has a one-line explanation of what raising or lowering it does.
8. **Original art only.** The arcade look must be our own. Do not copy characters, sprites, logos, fonts-as-logos, sounds, or level designs from any existing game.

---

## 3. Tech stack (use exactly this unless you have a blocking reason; if so, stop and explain)

| Concern | Choice | Why |
|---|---|---|
| Language | TypeScript, `strict: true`, no `any` | Safety for math-heavy code |
| Build | Vite | Fast, simple, static output |
| UI | React 18 | Ecosystem, familiarity |
| State | Zustand | Small, no boilerplate, works well with workers |
| Simulation | Web Worker + Comlink | Keeps UI at 60fps |
| Arena rendering | HTML Canvas 2D (with `devicePixelRatio` handling) | Fast for grids, pixel-art friendly |
| Diagrams & transitions | D3 (scales, interpolation, transitions) + SVG | Equation/backup-diagram morphs |
| Charts | uPlot | Very fast streaming time series |
| Math typesetting | KaTeX | Equations |
| Text diffing | `diff` (jsdiff) | Pseudocode line diffs |
| Styling | CSS Modules + CSS custom properties (design tokens) | Full control for a custom arcade look |
| Tests | Vitest (unit/reference), Playwright (e2e + screenshots) | |
| Quality | ESLint, Prettier, `tsc --noEmit` in CI | |
| PWA | `vite-plugin-pwa` | Installable, offline |
| CI/CD | GitHub Actions → Vercel (or GitHub Pages) | Free, automatic |
| Package manager | pnpm | |

Use the latest stable version of each dependency at the time you start and commit the lockfile. Do **not** add a backend, database, or auth. Do **not** use Python or Pyodide in the browser.

---

## 4. Architecture

### 4.1 Folder structure

```
rl-arcade/
  SPEC.md                    # this document
  CLAUDE.md                  # conventions + "add an algorithm" recipe (you maintain it)
  README.md                  # portfolio-quality: GIFs, live link, architecture diagram
  src/
    core/                    # PURE TypeScript. No DOM, no React. 100% unit-testable.
      rng.ts                 # seeded PRNG with labeled forkable streams
      types.ts               # Env, Agent, Planner, Experiment, events, snapshots
      envs/
        gridworld.ts         # editable grid (walls, lava, goal, wind, ice, teleport)
        presets.ts           # "levels" (section 8.2)
        bandit.ts            # k-armed bandit (Gaussian + Bernoulli)
      agents/                # one file per algorithm
        bandit/epsilonGreedy.ts, ucb1.ts, thompson.ts, gradientBandit.ts
        dp/valueIteration.ts, policyIteration.ts
        tabular/monteCarlo.ts, sarsa.ts, qLearning.ts, expectedSarsa.ts,
                nStepSarsa.ts, sarsaLambda.ts, dynaQ.ts
        policyGradient/reinforce.ts, actorCritic.ts
      registry.ts            # algorithm + environment registries
      metrics.ts             # returns, convergence detection, regret, diff metrics
    runner/
      worker.ts              # runs experiments, streams snapshots/events
      protocol.ts            # typed message definitions (discriminated unions)
      snapshotStore.ts       # ring buffer of snapshots for timeline scrubbing
    state/
      store.ts               # Zustand: experiments, runs, baseline, UI state
      urlState.ts            # encode/decode experiment config in the URL
    viz/                     # visualizer plugins (section 10)
    ui/                      # layout, panels, controls, editor, HUD
    design/
      tokens.css             # all design tokens (section 11)
      sfx.ts                 # synthesized 8-bit sound effects (WebAudio)
      sprites.ts             # pixel-art sprites drawn in code
    content/
      algorithms/*.ts        # plain-English explainers per algorithm
  tests/
    reference/               # textbook reference tests (section 13)
    e2e/                     # Playwright
```

### 4.2 Data flow

```
UI control change
   → store creates new immutable Experiment config (hash = id)
   → previous run becomes Baseline (unless user pinned one)
   → worker.run(experiment)            [cached by hash: identical config = no rerun]
   → worker streams: snapshots (Transferable Float32Arrays), events, metrics
   → store keeps latest + ring buffer
   → requestAnimationFrame renderer draws the latest snapshot
   → diff layer compares Current vs Baseline snapshots at the same time index
```

### 4.3 Core interfaces (implement these, refine names if needed, keep the shape)

```ts
// rng.ts
export interface Rng {
  next(): number;                 // [0,1)
  int(n: number): number;         // 0..n-1
  normal(): number;               // N(0,1)
  beta(a: number, b: number): number;
  fork(label: string | number): Rng; // independent, deterministic child stream
}
export function createRng(seed: number): Rng; // use a well-known PRNG (e.g. PCG32 or xoshiro128**) + a hash for fork()

// types.ts
export interface EnvSpec {
  id: string;
  kind: 'grid' | 'bandit';
  numStates: number;
  numActions: number;
  actionLabels: string[];        // e.g. ['↑','→','↓','←']
  layout?: GridLayout;           // for rendering
  hasModel: boolean;             // true if P(s'|s,a) and R are available (needed for DP)
}

export interface Transition { next: number; reward: number; done: boolean; info?: Record<string, unknown> }

export interface Env {
  readonly spec: EnvSpec;
  reset(rng: Rng): number;                          // start state
  step(s: number, a: number, rng: Rng): Transition;
  model?(s: number, a: number): Array<{ p: number; next: number; reward: number; done: boolean }>;
}

export type ParamScale = 'linear' | 'log' | 'int' | 'bool' | 'enum';
export interface ParamDef {
  id: string; label: string; symbol?: string;       // e.g. 'γ'
  scale: ParamScale; min?: number; max?: number; step?: number;
  options?: string[]; default: number | boolean | string;
  explainUp: string;   // "Higher γ: the agent cares more about far-future reward."
  explainDown: string;
  affects: string[];   // ids of equation tokens / pseudocode lines / viz layers this param touches
}

export interface AlgoMeta {
  id: string; name: string;
  family: 'bandit' | 'dp' | 'tabular' | 'policy-gradient' | 'deep';
  needsModel: boolean;
  params: ParamDef[];
  equation: EquationToken[];       // structured, for token-level diff (section 9.2)
  pseudocode: PseudoLine[];        // { id, text, indent }
  backupDiagram: BackupSpec;       // for the backup-diagram morph
  signatureViz: string[];          // viz plugin ids that best show this algorithm
  summary: string;                 // 2–3 plain-English sentences
  watchFor: string[];              // "Watch the Q-values along the cliff edge…"
  reference: string;               // e.g. "Sutton & Barto §6.5"
}

export type UpdateEvent =
  | { kind: 'td'; s: number; a: number; target: number; delta: number; line: string }
  | { kind: 'mc-return'; s: number; a: number; G: number; line: string }
  | { kind: 'sweep'; s: number; oldV: number; newV: number; line: string }
  | { kind: 'policy-improve'; s: number; oldA: number; newA: number; line: string }
  | { kind: 'planning'; s: number; a: number; delta: number; line: string }
  | { kind: 'trace'; traces: Float32Array; line: string }
  | { kind: 'bandit-pull'; arm: number; reward: number; line: string };

export interface Agent<Snap = AgentSnapshot> {
  readonly meta: AlgoMeta;
  init(env: EnvSpec, params: ParamValues, rng: Rng): void;
  act(s: number): number;
  update(s: number, a: number, t: Transition): UpdateEvent[];
  endEpisode?(): UpdateEvent[];
  inspect(): Snap;   // returns COPIES of typed arrays (Q, V, policy probs, traces, counts…)
}

// Model-based planners (Value/Policy Iteration) don't interact via episodes
export interface Planner<Snap = AgentSnapshot> {
  readonly meta: AlgoMeta;
  init(env: Env, params: ParamValues): void;
  iterate(): { events: UpdateEvent[]; done: boolean }; // one sweep (or one state update in "fine" mode)
  inspect(): Snap;
}

export interface Experiment {
  envId: string; envConfig: GridLayout | BanditConfig;
  algoId: string; params: ParamValues;
  seed: number; episodes: number; maxStepsPerEpisode: number;
}
```

### 4.4 Worker protocol

- Typed discriminated-union messages in `protocol.ts`: `run`, `pause`, `resume`, `step`, `stepEpisode`, `setSpeed`, `liveTweak`, `sweep` (sensitivity strip), `cancel`.
- The worker runs in **time-sliced chunks** (~8 ms of work, then yields) so it stays responsive to `pause`/`cancel`.
- Snapshots are sent as **Transferable** `Float32Array`s. Never JSON-encode big arrays.
- The renderer always draws the *latest* snapshot. Simulation speed is decoupled from render rate (turbo mode can run thousands of steps per frame).
- Snapshot ring buffer for timeline scrubbing: store a snapshot every N steps/episodes with a memory cap (default 50 MB). Document the policy.

---

## 5. Determinism and fair comparison (critical)

1. One root seed per experiment. Derive **separate streams**: `rng.fork('env')`, `rng.fork('agent')`.
2. Fork the env stream **per episode**: `envRng.fork(episodeIndex)`. Episode *k* then gets the same start state and the same environment randomness (slips, wind gusts, bandit rewards) in every run, whatever the agent did in earlier episodes. This is the **common random numbers** technique. It makes "baseline vs current" differences come from the change, not from luck.
3. Bandits: pre-draw the reward for arm *a* at time *t* from `envRng.fork(t).fork(a)`, so two algorithms pulling the same arm at the same step get the same reward.
4. Tests must prove: (a) same config → identical outputs; (b) switching algorithm keeps identical start states per episode.
5. UI exposes the seed (editable, plus a "REROLL" button with a pixel-dice icon) and an optional "average over N seeds" mode for curves (shaded variance band).

---

## 6. Algorithms — v1 scope

For each algorithm: implement it, give it full `AlgoMeta`, write a plain-English explainer, and write tests. Parameter ranges below are defaults; adjust only with justification.

### 6.1 Bandits (environment: k-armed, Gaussian or Bernoulli)

| Algorithm | Params | Signature visualization |
|---|---|---|
| ε-greedy | ε [0–1, 0.1], initial Q₀ [−5–5, 0] (optimistic init demo), step: sample-average or constant α | Estimated value bars vs hidden true values; exploration sparks |
| UCB1 | c [0–5, 2] | Value bars with **confidence-bound whiskers** that shrink as arms get pulled |
| Thompson sampling | prior α, β [0.1–10, 1] | **Posterior distribution curves** per arm sharpening over time; sampled values shown as dots |
| Gradient bandit | α [0.01–1, 0.1], baseline on/off | Preference bars + softmax probability bars |

Bandit charts: average reward, % optimal action, cumulative regret.

### 6.2 Dynamic programming (needs model)

| Algorithm | Params | Signature visualization |
|---|---|---|
| Value iteration | γ [0–0.999, 0.9], θ [1e-6–1e-1, log], sweep mode: synchronous / in-place, granularity: per-sweep / per-state | Value heatmap filling in sweep by sweep; Δ per sweep chart (convergence) |
| Policy iteration | γ, θ | Alternating phases clearly labeled: **EVALUATE** (values settle) → **IMPROVE** (arrows flip, flips highlighted) |

### 6.3 Tabular model-free

| Algorithm | Params | Signature visualization |
|---|---|---|
| Monte Carlo control | γ, ε, first-visit / every-visit, exploring starts on/off | Whole episode trail lights up, **then** returns propagate backward along it |
| SARSA | α [0.01–1, 0.5], γ, ε, ε-decay | Q-triangles per cell (4 triangles = 4 actions); the update uses the action actually taken next (highlight a′) |
| Q-learning | α, γ, ε, ε-decay | Same view, but highlight the **max** action in s′ (it may differ from the action taken) |
| Expected SARSA | α, γ, ε | Highlight all actions in s′ weighted by π (opacity = probability) |
| n-step SARSA | n [1–32, 4], α, γ, ε | n-step window drawn as a bracket over the trail |
| SARSA(λ) | λ [0–1, 0.9], trace type: accumulating / replacing, α, γ, ε | **Eligibility-trace heat trail** fading behind the agent |
| Dyna-Q | planning steps n [0–100, 10], α, γ, ε | Real updates in solid color; **imagined (planning) updates** as ghost flashes across the grid |

### 6.4 Policy gradient (tabular softmax)

| Algorithm | Params | Signature visualization |
|---|---|---|
| REINFORCE | α_θ [1e-4–1, log], γ, baseline on/off | Per-cell policy probability glyph (4 arrows, length ∝ π(a\|s)); gradient direction pulse after each episode |
| Actor-critic (one-step) | α_actor, α_critic, γ | Two layers: critic V heatmap + actor arrows; TD error δ shown as a pulse at the updated cell |

### 6.5 Phase 2 (do NOT build now, but keep the architecture ready)

DQN (replay buffer, target network, Double DQN toggle, Dueling toggle), A2C, PPO (clip ε, GAE λ, epochs, entropy coef) on CartPole and MountainCar, using a small hand-written MLP in TypeScript (Float32Array, Adam). Verify it with a finite-difference gradient check. Deep RL code must be lazy-loaded so it doesn't grow the initial bundle.

---

## 7. Environments

### 7.1 GridWorld (editable)

- Size from 3×3 to 16×16.
- Cell types: `empty`, `wall`, `start`, `goal` (terminal, +reward), `lava` (terminal, −reward), `cliff` (−100, teleports to start, not terminal), `wind` (pushes the agent 1–2 cells in a direction), `ice` (slip probability), `teleport` (A → A′ with reward, as in Sutton & Barto Example 3.5).
- Global settings: step reward (default −1 or 0 depending on preset), slip probability, max steps per episode, discount comes from the algorithm.
- Exposes `model()` so DP algorithms work on any user-edited grid (including stochastic cells).
- **Editor:** paint cell types with the mouse/touch, drag start/goal, a palette of tools, undo/redo, clear. Editing creates a new experiment (and therefore a diff against the previous one).

### 7.2 Presets ("Levels")

Each level is a preset environment plus a suggested algorithm pair and a one-line "challenge" that hints at what to look for.

1. **Bandit Alley:** 10-armed Gaussian testbed. *Challenge: beat ε-greedy with UCB.*
2. **Classic 5×5:** Sutton & Barto Example 3.5 grid with teleport cells A (+10) and B (+5). *Challenge: watch value iteration find the far-sighted policy.*
3. **Cliff Walk:** 4×12 cliff walking. *Challenge: SARSA vs Q-learning, safe path vs optimal path.*
4. **Windy Valley:** windy gridworld. *Challenge: how does SARSA cope with wind?*
5. **Ice Rink:** slippery maze. *Challenge: compare DP on a stochastic model vs Q-learning.*
6. **Long Hallway:** a long corridor with a single delayed reward. *Challenge: raise n or λ and watch credit travel faster.*
7. **Shifting Maze:** a maze where a wall moves after N episodes (blocking/shortcut maze idea). *Challenge: Dyna-Q with n=0 vs n=50.*

---

## 8. The sandbox UX

### 8.1 Layout (desktop ≥ 1280px; adapt down to phone width ~400px with stacked panels and no horizontal page scroll)

```
┌───────────────────────────── TOP BAR ─────────────────────────────┐
│ RL ARCADE   [LEVEL ▾]   [1P SOLO | 2P VERSUS]   [SHARE] [♪] [⚙]   │
├────────────┬──────────────────────────────────┬───────────────────┤
│ CONTROL    │            ARENA (canvas)        │   INSIGHT         │
│ DECK       │  ┌────────────────────────────┐  │  Update equation  │
│            │  │ grid / bandit cabinet      │  │  (with diff)      │
│ Env + edit │  │ overlay toggles: V | Q |   │  │  Pseudocode       │
│ tools      │  │ π | visits | traces | DIFF │  │  (current line lit│
│            │  └────────────────────────────┘  │   + diff marks)   │
│ Algorithm  │  HUD: EP 0042  STEP 013  RET −13 │  Charts (uPlot)   │
│ cartridges │  ◀◀ ▶ ▮▮ ▶| ▶▶| speed ═══ turbo  │  Diff summary     │
│            │  timeline scrubber ──●─────────  │  chips            │
│ Parameters │                                  │                   │
├────────────┴──────────────────────────────────┴───────────────────┤
│ HIGH SCORES: run history table (algo, changed params, score, ep-to-converge) │
└───────────────────────────────────────────────────────────────────┘
```

### 8.2 Controls

- **Algorithm picker:** "cartridges" grouped by family (Bandits, Planning, Tabular, Policy Gradient). Disabled cartridges (e.g. DP on an env without a model) show *why* they are disabled.
- **Parameter controls:** slider + numeric input + reset-to-default for each `ParamDef`. Log scale where specified.
  - **Hover/focus a parameter:** highlight every place it acts (`affects[]`): its symbol in the equation, the related pseudocode lines, and the related viz layer. Show `explainUp` / `explainDown`.
  - γ also shows an "effective horizon ≈ 1/(1−γ) steps" ring around the agent on hover.
- **Playback:** play/pause, step one update, step one episode, speed slider (1×…turbo), timeline scrubber over stored snapshots, reset.
- **Two change modes**, selectable in the parameter panel:
  - **Rerun (default):** a change restarts the run from episode 0 with the same seed. This is the mode the diff system uses.
  - **Live tweak:** a change applies from the current step onward. A vertical marker on all charts shows where the change happened.
- **Keyboard:** Space play/pause, → step, Shift+→ step episode, R reset, B pin baseline, 1 / 2 solo / versus, ? shortcut sheet. Every control is reachable by keyboard and shows a visible focus ring.

### 8.3 Explainers

Each algorithm has a compact card: 2–3 sentence summary, "watch for" bullets that point to its signature visualization, and the textbook reference. Plain language, no jargon without a tooltip definition.

---

## 9. The diff system (the heart of the project; build it with the most care)

### 9.1 Changing a parameter

When any parameter, seed, environment cell, or preset changes in Rerun mode:

1. The previous run becomes the **baseline** automatically, unless the user pinned one ("PIN BASELINE" button with a pixel-pin icon, key B). Use original pixel-art icons drawn in code, never emoji. A pinned baseline stays until unpinned.
2. The new run starts with the same seed (common random numbers).
3. Show all of the following:
   - **Ghost overlay:** baseline policy arrows / agent drawn faded (≈35% opacity, desaturated) under the current ones.
   - **Diff heatmap layer (DIFF toggle):** ΔV = V_current − V_baseline per cell on a symmetric diverging scale centered at 0 (colorblind-safe; see tokens). Legend with real values.
   - **Policy flip markers:** cells where the greedy action differs get a pulsing outline. Count shown in a chip: "Policy differs in 7 / 48 cells".
   - **Curve diff:** baseline and current learning curves overlaid, with the area between them shaded by sign (better / worse).
   - **Diff summary chips** (computed in `metrics.ts`): "Converged 312 episodes faster", "Final avg return +8.4", "Δ max |V| 3.2", "Regret −14%". Define convergence precisely (e.g. greedy policy unchanged for K consecutive episodes, or moving-average return within ε of its final value) and document it in the code.
4. **Sensitivity strip (on demand: "SWEEP" button next to a parameter):** run K = 8 values across the parameter's range in the worker, averaged over S seeds (default 5). Show small multiples: tiny final-policy thumbnail + mini learning curve for each value, plus a sensitivity curve (final return vs parameter value) with a variance band. Must finish in under ~2 s for tabular presets. Show progress while it runs.

### 9.2 Switching the algorithm

When the user switches algorithm A → B:

1. **Equation diff morph.** Each algorithm's update rule is stored as structured tokens (`EquationToken { id, latex, role }`), not as one opaque string. Diff A's and B's tokens by `id`:
   - shared tokens slide to their new positions (FLIP animation),
   - removed tokens fade out in the "removed" color,
   - added tokens fade in in the "added" color and keep a glow for a few seconds.

   Example, SARSA → Q-learning: the `γ·Q(s′,a′)` token turns into `γ·maxₐ Q(s′,a)`. Render with KaTeX per token.
2. **Pseudocode diff.** Line-level diff (jsdiff) between A's and B's pseudocode, shown inline with +/− gutter markers, then collapsing into B's normal view. During runs, the currently executing line (from `UpdateEvent.line`) is highlighted.
3. **Backup diagram morph.** Each algorithm declares its backup diagram (state/action nodes, depth, branching, whether it samples or takes an expectation/max). Animate between them with D3 transitions. Examples: MC → TD shrinks the full-episode chain to one step; SARSA → Expected SARSA fans the next action out into a weighted expectation; TD → Dyna-Q adds simulated branches.
4. **"What changed" card:** 1–3 sentences generated from metadata differences (e.g. "Q-learning bootstraps from the best next action instead of the action it will actually take. Expect riskier, more optimal paths.").
5. **2P VERSUS mode** (also reachable directly from the top bar): split-screen arenas for A and B on the same environment and seed, with synchronized playback and timeline. A third **disagreement map** shows the cells where their greedy policies differ, colored by which algorithm values that cell higher. Overlaid curves. Either side's algorithm and params can be changed independently.

### 9.3 Run history ("HIGH SCORES")

A table of the last N runs (default 20). Columns: algorithm, params that differ from the previous run (as chips), seed, score (final moving-average return or negative regret), episodes-to-converge. Click a row to restore it, Shift-click two rows to compare them. The current best score per level gets a small crown. Store in memory, and optionally in `localStorage` (wrapped in try/catch; the app must work without it).

### 9.4 Sharing

The full experiment config (env layout, algorithm, params, seed, mode, baseline config) is encoded compactly in the URL (`#` fragment, compressed + base64url). Opening the link reproduces the exact runs. A "SHARE" button copies the link (clipboard API with a fallback that selects the text).

---

## 10. Visualizer plugins

Each plugin is `{ id, label, appliesTo(envSpec, algoMeta), draw(ctx, snapshot, baseline?, theme) }` and draws into a layer of the arena canvas or into its own panel. Required v1 plugins:

`GridBase` (cells, walls, special tiles in pixel-art style), `AgentSprite` (animated, original pixel character), `ValueHeatmap`, `QTriangles`, `PolicyArrows`, `PolicyProbGlyphs`, `VisitCounts`, `EligibilityTrail`, `EpisodeTrail`, `PlanningFlashes`, `DiffHeatmap`, `PolicyFlipMarkers`, `GhostBaseline`, `BanditCabinet` (arms as arcade levers with value bars), `ConfidenceWhiskers`, `PosteriorCurves`, `LearningCurve`, `ConvergenceChart`, `RegretChart`, `EquationDiff`, `PseudocodePanel`, `BackupDiagram`, `SensitivityStrip`, `DisagreementMap`.

Every color-coded layer has a legend with real values. Numbers on cells appear only when the cell is large enough to read them (≥ 40 px), otherwise on hover.

---

## 11. Arcade design system

### 11.1 Direction

An **original** neon-arcade cabinet: a dark indigo screen glowing inside a bezel, chunky pixel borders, HUD-style readouts, and playful arcade naming ("PRESS START", "1P SOLO / 2P VERSUS", "HIGH SCORES", "LEVEL SELECT"), while every label stays clear about what it does. Fun on the surface, rigorous underneath.

### 11.2 Tokens (`design/tokens.css`). Start from these, tune for contrast.

```css
/* Layout: three-panel cabinet; arena is the bright "screen", side panels are the darker cabinet body */
:root {
  --bg:          #0c0a1f;  /* cabinet body */
  --screen:      #120f2e;  /* arena background */
  --panel:       #1a1640;
  --line:        #2e2866;  /* grid lines, pixel borders */
  --text:        #e9e6ff;
  --text-dim:    #9a93d1;
  --neon-cyan:   #2de2e6;  /* primary accent, positive values, "added" */
  --neon-pink:   #ff3caa;  /* negative values, "removed" */
  --neon-yellow: #ffd319;  /* agent, highlights, current pseudocode line */
  --neon-orange: #ff8a3d;  /* lava, danger */
  --ghost:       #8f86c9;  /* baseline ghost (used at reduced opacity) */
  --good:        #5cf28c;  --bad: #ff5470;   /* semantic only, in chips */

  --font-display: "Press Start 2P", "Courier New", monospace; /* titles + HUD labels ONLY, small sizes, uppercase */
  --font-data:    "VT323", "Courier New", monospace;          /* big HUD numbers */
  --font-body:    "IBM Plex Mono", ui-monospace, monospace;   /* explanations, controls, tooltips */
}
```

- Load fonts from Google Fonts with `display=swap` and real fallbacks. **Never** use the pixel display font for paragraphs or small body text.
- Diverging value scale: cyan (positive) ↔ dark neutral (0) ↔ pink (negative). Also offer a colorblind-safe alternative palette in settings (e.g. blue ↔ orange), and test both.
- The app is dark-first by design. Optionally provide a "daylight cabinet" light theme from the same token names; if you do, every token must be redefined for it.

### 11.3 Effects and motion

- CRT overlay on the arena only: subtle scanlines (repeating gradient, ≤ 6% opacity), slight vignette, gentle glow (`text-shadow` / `box-shadow`). Toggle "CRT FX" in settings.
- Pixel borders via stepped `box-shadow`. Sprites drawn on canvas with `imageSmoothingEnabled = false`.
- Small, meaningful animations: cartridge "insert" when switching algorithm, equation morph, policy-flip pulse, "LEVEL CLEAR!" banner when a run converges.
- `prefers-reduced-motion`: disable CRT flicker, pulses, and morph animations (jump to the end state).
- **Photosensitivity:** nothing may flash more than 3 times per second (WCAG 2.3.1). Turbo mode must not strobe; throttle visual flashes independently of the simulation speed.

### 11.4 Sound

Synthesized 8-bit sound effects with WebAudio (no audio files): step blip, goal chime, lava buzz, cartridge insert, level clear. **Muted by default**, toggle in the top bar, starts only after a user gesture. Auto-silence step sounds above 4× speed.

### 11.5 Accessibility

WCAG AA contrast for all text; visible focus rings (neon outline); every control has a label; HUD values are mirrored in an `aria-live="polite"` region (throttled); keyboard shortcuts listed in a help sheet (`?`); the canvas has a text summary for screen readers (current episode, return, policy differences).

---

## 12. Performance budgets (measure and report them)

- 60 fps rendering on a mid-range laptop during normal playback.
- Turbo mode: ≥ 100k environment steps/second for tabular agents on a 10×10 grid.
- Rerun after a slider change: debounce 150 ms. Show the first frame of the new run within 300 ms, and finish 500 episodes on a 10×10 grid within ~1 s.
- Initial JS ≤ 300 KB gzipped (lazy-load charts/diagrams if needed; deep RL later must be its own chunk).
- Lighthouse performance and accessibility ≥ 90.
- No memory growth beyond the snapshot cap during long runs.

---

## 13. Testing (non-negotiable)

### 13.1 Unit tests (Vitest), `src/core` ≥ 90% line coverage
- RNG: determinism, fork independence, distribution sanity checks.
- Each environment: transitions, rewards, terminal handling, `model()` probabilities sum to 1.
- Each agent: one hand-computed update step matches the expected number exactly (e.g. one Q-learning update with given α, γ, r, Q-values).

### 13.2 Reference tests against textbook results (Sutton & Barto, 2nd ed.)
- **Classic 5×5 (Example 3.5), γ = 0.9:**
  - Policy evaluation of the equiprobable random policy reproduces the textbook value grid (top row ≈ 3.3, 8.8, 4.4, 5.3, 1.5) within 0.05.
  - Value iteration reproduces the optimal values (top row ≈ 22.0, 24.4, 22.0, 19.4, 17.5) within 0.05.
- **Value iteration and policy iteration** converge to the same V* (max abs diff < 1e-6) on every preset with a model.
- **Cliff Walk (Example 6.6), ε = 0.1, α = 0.5, averaged over ≥ 20 seeds:**
  - Q-learning's greedy policy after training follows the cliff edge (greedy return = −13).
  - SARSA's online average return over episodes 400–500 is higher than Q-learning's.
- **10-armed testbed:** averaged over ≥ 500 runs × 1000 steps, UCB (c = 2) achieves higher average reward than ε-greedy (ε = 0.1) over the last 100 steps; ε = 0.1 beats ε = 0 (greedy) on % optimal action.
- **Thompson sampling (Bernoulli):** cumulative regret grows sublinearly (regret(10k)/regret(1k) < 5).
- **Dyna-Q:** on a maze preset, n = 50 reaches the goal in a stable short path in fewer episodes than n = 0 (averaged over seeds).
- **n-step / λ:** on Long Hallway, value information reaches the start state in fewer episodes for n = 8 than n = 1.
- **Determinism:** the same Experiment twice gives identical snapshots; switching algorithm with the same seed gives identical per-episode start states.

Use fixed seeds and averaging so tests are not flaky. **Never weaken a reference test to make it pass. If one fails, the implementation is wrong until proven otherwise.**

### 13.3 E2E (Playwright)
- Load each level, press start, assert the HUD episode count increases.
- Change γ → the baseline ghost and diff chips appear.
- Switch SARSA → Q-learning → the equation diff shows the added `max` token.
- 2P VERSUS renders two arenas and a disagreement map.
- A share URL round-trips to the same config.
- Screenshot tests at 1440px and 400px widths.

---

## 14. Engineering standards

- `tsc --noEmit`, ESLint, Prettier, unit tests, and build all pass in CI on every push. Deploy on merge to `main`.
- No `any`, no `// @ts-ignore` without a written reason, no dead code, no commented-out blocks.
- `src/core` stays pure: no imports from React, DOM, or `ui/`.
- Small commits with clear messages (Conventional Commits).
- Errors are shown to the user in plain language with how to fix them (e.g. "Value iteration needs a model of the environment; this grid has none. Pick a tabular algorithm instead.").
- Keep `CLAUDE.md` current. It must contain:
  - project conventions,
  - **"How to add an algorithm"**: create `core/agents/<family>/<name>.ts` implementing `Agent` or `Planner` → fill in `AlgoMeta` (params with `explainUp`/`explainDown`/`affects`, equation tokens, pseudocode lines, backup diagram, signature viz) → register in `registry.ts` → add an explainer in `content/algorithms` → add unit + reference tests. Then it works in the UI, the diff system, sweeps and 2P mode without further changes,
  - "How to add an environment" and "How to add a visualizer plugin".

---

## 15. Milestones — build in this order, and verify at the end of each

At the end of **every** milestone: run typecheck, lint, unit tests, e2e (when present), and a production build. Then report: what was built, what you verified and how, what isn't done or isn't verified, and any deviations from this spec with reasons. Do not start the next milestone while the current one has failing checks.

- **M0. Foundation:** scaffold, tooling, CI, deploy pipeline, design tokens, fonts, empty arcade shell (top bar, three panels, bezel, CRT overlay). *Done when:* the shell is live at a public URL.
- **M1. Core engine (headless):** RNG with streams, GridWorld + model, presets, bandit env, value iteration, policy iteration, Q-learning, SARSA, metrics. *Done when:* the reference tests for these pass.
- **M2. Worker + arena:** runner, protocol, snapshot store, canvas layers (grid, sprite, values, Q-triangles, arrows), HUD, playback controls, timeline, learning-curve chart. *Done when:* every level runs smoothly at 60 fps and turbo meets its budget.
- **M3. Sandbox controls:** algorithm cartridges, parameter panel with hover-highlighting, grid editor with undo/redo, level select, explainers, keyboard shortcuts.
- **M4. Diff system:** baseline/ghost/pin, diff heatmap, policy flips, curve diff, summary chips, equation token diff, pseudocode diff, "what changed" card, 2P VERSUS + disagreement map, high scores, URL sharing. *Done when:* the e2e diff tests pass.
- **M5. Full v1 algorithm set:** remaining bandits, MC, Expected SARSA, n-step, SARSA(λ), Dyna-Q, REINFORCE, actor-critic, with their signature visualizations, backup-diagram morphs, and the sensitivity strip.
- **M6. Polish and ship:** sound, PWA, accessibility audit, performance audit against budgets, colorblind palette, mobile layout, README (live link, GIFs of the diff features, architecture diagram, "how to add an algorithm"), and an About panel explaining the project and its tech.
- **Phase 2 (later, separate request):** deep RL (section 6.5).

---

## 16. How you work with me

1. **First response:** don't write code yet. Restate the goal in 3–5 sentences, list any ambiguities that block you (only real blockers, each with your recommended default), and give a short plan for M0–M1. Then wait for my go-ahead.
2. When a decision is minor, pick the sensible default, note it in the milestone report, and keep going.
3. When RL math is involved, cite the Sutton & Barto section you follow in a code comment.
4. Never fake results: no hard-coded "demo" numbers posing as computed ones, no placeholder functions reported as done, no skipped tests reported as passing.
5. If something in this spec is wrong or a much better option exists, say so with reasons before deviating.

---

## 17. Definition of done (v1)

- Live public URL, installable as a PWA, works offline after first load.
- All v1 algorithms and levels work, with their signature visualizations.
- Changing any parameter shows ghost + diff heatmap + curve diff + summary chips.
- Switching algorithms shows the equation morph, pseudocode diff, backup-diagram morph, and "what changed" card. 2P VERSUS works with the disagreement map.
- Share links reproduce runs exactly.
- All tests pass in CI. Performance and accessibility budgets are met and reported.
- The README is portfolio-ready.
