import type { Cell, EnvConfig, GridLayout, ParamValues, RunSettings } from '../types';
import { emptyLayout } from './gridworld';

/**
 * Build a layout from ASCII rows.
 *   .  empty      #  wall      S  start     G  goal      L  lava     C  cliff
 *   I  ice        w  wind ↑1   W  wind ↑2   A/B teleport sources, a/b their targets
 */
export function fromAscii(rows: string[], overrides: Partial<GridLayout> = {}, teleports: Record<string, number> = {}): GridLayout {
  const h = rows.length;
  const w = rows[0].length;
  const base = emptyLayout(w, h);
  const cells: Cell[] = [];
  const targets: Record<string, number> = {};
  rows.forEach((row, r) => {
    [...row].forEach((ch, c) => {
      if (ch >= 'a' && ch <= 'z' && ch !== 'w') targets[ch] = r * w + c;
    });
  });
  rows.forEach((row) => {
    for (const ch of row) {
      switch (ch) {
        case '#':
          cells.push({ t: 'wall' });
          break;
        case 'S':
          cells.push({ t: 'start' });
          break;
        case 'G':
          cells.push({ t: 'goal' });
          break;
        case 'L':
          cells.push({ t: 'lava' });
          break;
        case 'C':
          cells.push({ t: 'cliff' });
          break;
        case 'I':
          cells.push({ t: 'ice' });
          break;
        case 'w':
          cells.push({ t: 'wind', dir: 0, strength: 1 });
          break;
        case 'W':
          cells.push({ t: 'wind', dir: 0, strength: 2 });
          break;
        default:
          if (ch >= 'A' && ch <= 'Z') {
            const tgt = targets[ch.toLowerCase()];
            cells.push({ t: 'teleport', target: tgt, reward: teleports[ch] ?? 0, label: ch });
          } else if (ch >= 'a' && ch <= 'z') {
            cells.push({ t: 'empty', label: ch.toUpperCase() + '′' });
          } else cells.push({ t: 'empty' });
      }
    }
  });
  return { ...base, ...overrides, w, h, cells };
}

export interface SlotPreset {
  algoId: string;
  params: ParamValues;
}

export interface Level {
  id: string;
  name: string;
  challenge: string;
  env: EnvConfig;
  run: RunSettings;
  A: SlotPreset;
  B: SlotPreset;
}

const grid = (layout: GridLayout): EnvConfig => ({ kind: 'grid', layout });

const blockingRow = (gapLeft: boolean) => (gapLeft ? '.########' : '########.');
const mazeRows = (gapLeft: boolean) => ['........G', '.........', '.........', blockingRow(gapLeft), '.........', '...S.....'];
const shiftingMaze = (() => {
  const L = fromAscii(mazeRows(false), { stepReward: -1, bumpReward: -1, goalReward: 0 });
  const after = fromAscii(mazeRows(true));
  const set: [number, Cell][] = [];
  for (let c = 0; c < 9; c++) set.push([3 * 9 + c, after.cells[3 * 9 + c]]);
  return { ...L, events: [{ atEpisode: 40, set }] };
})();

export const LEVELS: Level[] = [
  {
    id: 'bandit-alley',
    name: 'Bandit Alley',
    challenge: 'Ten slot machines with hidden payouts. Beat ε-greedy with UCB.',
    env: { kind: 'bandit', bandit: { k: 10, dist: 'gaussian', steps: 1000 } },
    run: { episodes: 1, maxSteps: 1000, seed: 7, avgSeeds: 100 },
    A: { algoId: 'eps-greedy', params: { epsilon: 0.1 } },
    B: { algoId: 'ucb', params: { c: 2 } },
  },
  {
    id: 'classic-5x5',
    name: 'Classic 5×5',
    challenge: 'Teleports A (+10) and B (+5), no exit. Watch value iteration find the far-sighted policy.',
    env: grid(
      fromAscii(['.A.B.', '.....', '...b.', '.....', '.a...'], { stepReward: 0, bumpReward: -1, randomStart: true }, { A: 10, B: 5 }),
    ),
    run: { episodes: 150, maxSteps: 60, seed: 1, avgSeeds: 1 },
    A: { algoId: 'value-iteration', params: { gamma: 0.9 } },
    B: { algoId: 'policy-iteration', params: { gamma: 0.9 } },
  },
  {
    id: 'cliff-walk',
    name: 'Cliff Walk',
    challenge: 'SARSA vs Q-learning: the safe path or the optimal one along the edge?',
    env: grid(fromAscii(['............', '............', '............', 'SCCCCCCCCCCG'], { stepReward: -1, bumpReward: -1, goalReward: 0, cliffReward: -100 })),
    run: { episodes: 500, maxSteps: 300, seed: 3, avgSeeds: 1 },
    A: { algoId: 'sarsa', params: { alpha: 0.5, epsilon: 0.1, gamma: 0.99 } },
    B: { algoId: 'qlearning', params: { alpha: 0.5, epsilon: 0.1, gamma: 0.99 } },
  },
  {
    id: 'windy-valley',
    name: 'Windy Valley',
    challenge: 'An upward wind pushes the agent off course. How does SARSA cope?',
    env: grid(
      fromAscii(['...wwwWWw.', '...wwwWWw.', '...wwwWWw.', 'S..wwwWGw.', '...wwwWWw.', '...wwwWWw.', '...wwwWWw.'], {
        stepReward: -1,
        bumpReward: -1,
        goalReward: 0,
      }),
    ),
    run: { episodes: 200, maxSteps: 2000, seed: 5, avgSeeds: 1 },
    A: { algoId: 'sarsa', params: { alpha: 0.5, epsilon: 0.1, gamma: 0.99 } },
    B: { algoId: 'expected-sarsa', params: { alpha: 0.5, epsilon: 0.1, gamma: 0.99 } },
  },
  {
    id: 'ice-rink',
    name: 'Ice Rink',
    challenge: 'Ice cells make you slide at random. Planning with the true model vs learning by falling.',
    env: grid(
      fromAscii(['S.II..', '.#II.L', '..I...', 'L.II#.', '..II..', '.L..IG'], { stepReward: -1, bumpReward: -1, goalReward: 10, lavaReward: -20, iceSlip: 0.5 }),
    ),
    run: { episodes: 400, maxSteps: 100, seed: 11, avgSeeds: 1 },
    A: { algoId: 'value-iteration', params: { gamma: 0.95 } },
    B: { algoId: 'qlearning', params: { alpha: 0.2, epsilon: 0.1, gamma: 0.95 } },
  },
  {
    id: 'long-hallway',
    name: 'Long Hallway',
    challenge: 'One reward at the far end. Raise n (or λ) and watch credit travel back faster.',
    env: grid(fromAscii(['S.................G'], { stepReward: 0, bumpReward: 0, goalReward: 10 })),
    run: { episodes: 60, maxSteps: 400, seed: 2, avgSeeds: 1 },
    A: { algoId: 'nstep-sarsa', params: { n: 1, alpha: 0.5, epsilon: 0.1, gamma: 0.95 } },
    B: { algoId: 'nstep-sarsa', params: { n: 8, alpha: 0.5, epsilon: 0.1, gamma: 0.95 } },
  },
  {
    id: 'shifting-maze',
    name: 'Shifting Maze',
    challenge: 'At episode 40 the wall moves and the short route closes. Dyna-Q with n = 0 vs n = 50.',
    env: grid(shiftingMaze),
    run: { episodes: 120, maxSteps: 600, seed: 4, avgSeeds: 1 },
    A: { algoId: 'dyna-q', params: { planning: 0, alpha: 0.1, epsilon: 0.1, gamma: 0.95 } },
    B: { algoId: 'dyna-q', params: { planning: 50, alpha: 0.1, epsilon: 0.1, gamma: 0.95 } },
  },
];

export const getLevel = (id: string) => LEVELS.find((l) => l.id === id) ?? LEVELS[2];
