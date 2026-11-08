# RL Arcade: notes for AI agents and contributors

The full product spec is in `RL_ARCADE_BUILD_PROMPT.md` (also referred to as SPEC). Read it before large changes.

## Commands

- `npm run dev`: dev server on :5173
- `npm test`: Vitest (unit + textbook reference tests). Must pass before every commit.
- `npm run typecheck`, `npm run build`: build runs the typecheck first.

## Rules

- `src/core` is pure TypeScript: no React, no DOM, no imports from `ui/`, `viz/`, `state/`.
- Determinism is sacred. All randomness goes through `Rng` (`src/core/rng.ts`). Environment randomness uses `root.fork('env').fork(episode)`; agent randomness uses `root.fork('agent')`. The environment draws exactly one uniform per step. Never use `Math.random()` in `src/core`.
- Never weaken a reference test to make it pass. If one fails, the implementation is wrong until proven otherwise.
- Strict TS, no `any`. Cite the Sutton & Barto section in a comment when implementing RL math.
- Copy is plain language from the user's side ("Settles 12 episodes sooner", not internal names).
- Colors come from CSS tokens in `src/styles/app.css`. Canvas code reads them through `readTheme()`.

## Architecture in one paragraph

A slider change creates a new immutable `Experiment` → `state/store.ts` keeps the old run as the baseline (solo) and asks a worker (`runner/client.ts`, cached by experiment hash) to run it → `core/engine.ts` runs the whole experiment headlessly and records per-episode snapshots, the trajectory, and the table row updated at each step → `viz/frame.ts` rebuilds the exact state at any (episode, step) → `viz/drawGrid.ts` / `drawBandit.ts` draw it, with the baseline frame as a ghost and diff. Live tweaks are stored as `tweaks: [{at, params}]` and replayed deterministically.

## How to add an algorithm

1. Create `src/core/agents/<family>/<name>.ts` implementing `GridAgent` (sample-based), `GridPlanner` (needs the model), or `BanditAgent`.
   - `update()` returns a `StepInfo` with the pseudocode `line` id and the state `s` whose row changed. That is what the UI replays between snapshots.
   - Expose live tables through `tables()`: `Q` and/or `V` and/or `P` (policy probabilities).
2. Write its `AlgoMeta`:
   - `params`: each with range, default, `explainUp` / `explainDown`, `affects` (viz layers).
   - `equation`: **tokens with stable ids**. Reuse ids from related algorithms (`lhs`, `alpha`, `gamma`, `boot-*`, …) so the equation diff shows what actually changed.
   - `pseudocode` lines (ids referenced by `StepInfo.line`, `params` for hover highlighting).
   - `backup` diagram (reuse node ids `s0`, `a0`, `s1`, … so it morphs nicely).
   - `traits` (drives the "what changed" text), `summary`, `watchFor`, `reference`.
3. Register it in `src/core/registry.ts`.
4. Add tests: one hand-computed update in `tests/reference/units.test.ts`, and a behavioral reference test if a textbook result exists. The "every algorithm runs on every level" test picks it up automatically.

The UI, the diff system, sweeps, versus mode, share links and high scores need no changes.

## How to add a level

Add an entry to `LEVELS` in `src/core/envs/presets.ts`. ASCII legend is in `fromAscii`. Give it a challenge sentence and a suggested P1/P2 pair.

## Known limits

- Inside an episode, the board replays the single row each step updated. For SARSA(λ) and Dyna-Q, the other cells they touch update visibly at the episode boundary (snapshots there are exact).
- Recording is capped at 1.5M steps per run. The UI says so when a run hits it.
- Deep RL (DQN, A2C, PPO on CartPole) is Phase 2 in the spec and not built yet.
