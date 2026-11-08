/** Shared types for the pure simulation core. Nothing in src/core may import React or the DOM. */

export type ParamValue = number | boolean | string;
export type ParamValues = Record<string, ParamValue>;

export type ParamScale = 'linear' | 'log' | 'int' | 'bool' | 'enum';

export interface ParamDef {
  id: string;
  label: string;
  /** Math symbol used in the equation, e.g. 'γ'. Equation tokens reference the param by id. */
  symbol?: string;
  scale: ParamScale;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  default: ParamValue;
  explainUp: string;
  explainDown: string;
  /** Visual layers this parameter acts on, e.g. 'viz:horizon', 'viz:explore'. */
  affects: string[];
}

/** One piece of an update equation. Tokens with the same id across algorithms are "the same idea". */
export interface EqToken {
  id: string;
  tex: string;
  param?: string;
}

export interface PseudoLine {
  id: string;
  text: string;
  indent: number;
  params?: string[];
}

export type BackupNodeKind = 'state' | 'action' | 'terminal' | 'dots' | 'model';
export interface BackupNode {
  id: string;
  kind: BackupNodeKind;
  x: number;
  y: number;
  label?: string;
}
export interface BackupArc {
  id: string;
  /** Node ids the arc spans (drawn under their common parent). */
  over: string[];
  label: string;
}
export interface BackupSpec {
  nodes: BackupNode[];
  edges: [string, string][];
  arcs?: BackupArc[];
  dashed?: [string, string][];
  caption: string;
}

export type Family = 'bandit' | 'dp' | 'tabular' | 'policy-gradient';
export type EnvKind = 'grid' | 'bandit';

/** Qualitative traits used to generate the "what changed" text when switching algorithms. */
export interface AlgoTraits {
  /** Finishes the sentence "<name> updates toward …". */
  target: string;
  onPolicy: boolean;
  model: 'none' | 'learned' | 'given';
  /** Finishes "… and updates …". */
  timing: string;
  learns: string;
}

export interface AlgoMeta {
  id: string;
  name: string;
  short: string;
  family: Family;
  envKind: EnvKind;
  needsModel: boolean;
  params: ParamDef[];
  equation: EqToken[];
  pseudocode: PseudoLine[];
  backup: BackupSpec;
  summary: string;
  watchFor: string[];
  reference: string;
  traits: AlgoTraits;
}

// ---------------------------------------------------------------- environments

export type CellType =
  | 'empty'
  | 'wall'
  | 'start'
  | 'goal'
  | 'lava'
  | 'cliff'
  | 'wind'
  | 'ice'
  | 'teleport';

export interface Cell {
  t: CellType;
  /** wind direction (0 up, 1 right, 2 down, 3 left) */
  dir?: number;
  /** wind strength in cells */
  strength?: number;
  /** teleport destination cell index */
  target?: number;
  /** teleport reward, or goal reward override */
  reward?: number;
  label?: string;
}

export interface GridEvent {
  atEpisode: number;
  set: [number, Cell][];
}

export interface GridLayout {
  w: number;
  h: number;
  cells: Cell[];
  /** reward for every move */
  stepReward: number;
  /** reward when bumping into a wall or the edge (replaces stepReward) */
  bumpReward: number;
  /** added on entering a goal */
  goalReward: number;
  /** added on entering lava */
  lavaReward: number;
  /** replaces the step reward when falling off a cliff (agent returns to start) */
  cliffReward: number;
  /** probability that any move is replaced by a uniformly random one */
  slip: number;
  /** slip probability on ice cells */
  iceSlip: number;
  /** start each episode in a random non-terminal cell */
  randomStart: boolean;
  /** layout changes that happen at a given episode (e.g. a moving wall) */
  events?: GridEvent[];
}

export interface BanditConfig {
  k: number;
  dist: 'gaussian' | 'bernoulli';
  steps: number;
}

export type EnvConfig = { kind: 'grid'; layout: GridLayout } | { kind: 'bandit'; bandit: BanditConfig };

export interface RunSettings {
  /** episodes for agents, max sweeps for planners (ignored by bandits, which use bandit.steps) */
  episodes: number;
  maxSteps: number;
  seed: number;
  /** number of seeds averaged for the learning curves (1 = only the displayed run) */
  avgSeeds: number;
}

/** A parameter change applied from a given episode (grid) or step (bandit) onward: "live tweak". */
export interface Tweak {
  at: number;
  params: ParamValues;
}

export interface Experiment {
  env: EnvConfig;
  algoId: string;
  params: ParamValues;
  run: RunSettings;
  tweaks: Tweak[];
}

export interface Transition {
  next: number;
  reward: number;
  done: boolean;
}

export interface Outcome {
  p: number;
  next: number;
  reward: number;
  done: boolean;
}

// ---------------------------------------------------------------- agents

/** Live references to an agent's tables (the engine copies them into snapshots). */
export interface Tables {
  Q?: Float32Array;
  V?: Float32Array;
  P?: Float32Array;
}

export interface StepInfo {
  /** pseudocode line id that best describes this step's update */
  line: string;
  /** state whose row was updated this step (-1 = none) */
  s: number;
  target?: number;
  delta?: number;
  /** states touched by planning (Dyna) this step, at most 4 */
  plan?: number[];
}

export interface GridAgent {
  setParams(p: ParamValues): void;
  beginEpisode(ep: number, s: number): void;
  act(s: number): number;
  /** true if the last act() was exploratory (not the current greedy choice) */
  lastExplore: boolean;
  update(s: number, a: number, tr: Transition): StepInfo | null;
  endEpisode(truncated: boolean, lastState: number): void;
  tables(): Tables;
}

export interface PlannerFrame {
  /** 0 = evaluate / sweep, 1 = improve */
  phase: number;
  delta: number;
  done: boolean;
}

export interface GridPlanner {
  frame(): PlannerFrame;
  tables(): Tables;
}

export type BanditExtraKind = 'none' | 'ucb' | 'beta' | 'normal' | 'softmax';

export interface BanditTables {
  Q: Float32Array;
  N: Float32Array;
  /** two floats per arm, meaning depends on xKind (ucb: bonus; beta: a,b; normal: mean,sd; softmax: π,H) */
  X: Float32Array;
}

export interface BanditAgent {
  readonly xKind: BanditExtraKind;
  setParams(p: ParamValues): void;
  act(t: number): number;
  lastExplore: boolean;
  update(arm: number, reward: number): string;
  tables(): BanditTables;
}
