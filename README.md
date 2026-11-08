# RL Arcade

An interactive reinforcement learning sandbox with an arcade look. Turn any knob or swap the algorithm, and see exactly what changes: on the board, in the learning curves, and in the math.

<!-- Add a screenshot or GIF here, e.g. docs/screenshot.png (the Cliff Walk SARSA → Q-learning swap makes a good one). -->

**Live demo:** https://zamorzgar.github.io/rl-arcade/

## What it does

- **15 algorithms, all running live in your browser:** ε-greedy, UCB1, Thompson sampling, gradient bandit · value iteration, policy iteration · Monte Carlo control, SARSA, Q-learning, Expected SARSA, n-step SARSA, SARSA(λ), Dyna-Q · REINFORCE (with baseline), one-step actor-critic.
- **7 levels**, including Sutton & Barto's 5×5 teleport grid, Cliff Walk, Windy Gridworld and a maze whose wall moves mid-run. A level editor covers walls, lava, cliffs, ice, wind and portals.
- **The diff system** (the point of the project):
  - **Change a number:** the previous run becomes a *baseline*. It shows as a faded ghost agent and ghost arrows, a ΔV heatmap, outlined policy flips, dashed learning curves with the gap shaded, and chips such as "settles 255 episodes sooner".
  - **Swap the algorithm:** the update equation morphs token by token (shared terms slide, new terms glow, removed terms fade out). The pseudocode shows a line diff, the backup diagram animates into the new shape, and a card explains what changed in plain language.
  - **2P Versus:** two algorithms on the same level with the *same luck*, plus a disagreement map of where their policies differ.
  - **Sweep:** run any parameter across its whole range (averaged over seeds) and click a result to try it.
- **Exact replay:** every run is fully deterministic. Share links reproduce what you saw.

## Engineering notes

| | |
|---|---|
| Stack | Vite · React 18 · TypeScript (strict) · Zustand · Canvas 2D · KaTeX · Vitest |
| Simulation | Pure TypeScript core in `src/core`, run in Web Workers. A whole tabular run takes milliseconds, so every slider change recomputes the entire run. |
| Playback | The engine records per-episode snapshots, the trajectory, and the table row updated at every step. The UI rebuilds the exact state at any (episode, step), which makes scrubbing, ghosts and diffs exact. |
| Fair comparisons | Separate seeded random streams for the environment and the agent, forked per episode (common random numbers). Two runs that differ only in γ see the same start states, slips and bandit payouts. |
| Correctness | Reference tests against textbook results: Fig. 3.2 and 3.5 values on the 5×5 grid, VI ≡ PI on every level, Q-learning's cliff-edge path (return −13), SARSA beating Q-learning online, UCB beating ε-greedy on the 10-armed testbed, sublinear Thompson regret, Dyna-Q and n-step speedups. |
| Accessibility | Keyboard shortcuts (`?` in the app), visible focus, a colorblind-safe palette, reduced-motion support, and no flashing faster than 3 Hz. |
| Offline | Installable PWA with a small service worker. |

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
npm test           # unit + reference tests
npm run build      # typecheck + production build to dist/
```

## Project layout

```
src/core/      simulation engine: rng, environments, agents, metrics, engine (no DOM, no React)
src/runner/    Web Worker + typed protocol + result cache
src/state/     Zustand store, share-link encoding
src/viz/       frame reconstruction and canvas renderers
src/ui/        React components
tests/         unit and textbook reference tests
```

To add an algorithm, see [CLAUDE.md](CLAUDE.md#how-to-add-an-algorithm).

## References

Sutton, R. S. & Barto, A. G. *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. Section references appear in the app next to each algorithm.
