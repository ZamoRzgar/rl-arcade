import type { AlgoMeta, BanditAgent, BanditConfig, GridAgent, GridPlanner, ParamValues } from './types';
import type { Rng } from './rng';
import type { GridEnv } from './envs/gridworld';
import { defaults } from './agents/common';
import { expectedSarsaMeta, qLearningMeta, sarsaMeta, TDAgent } from './agents/tabular/td';
import {
  DynaQAgent,
  dynaQMeta,
  MonteCarloAgent,
  monteCarloMeta,
  NStepSarsaAgent,
  nStepSarsaMeta,
  SarsaLambdaAgent,
  sarsaLambdaMeta,
} from './agents/tabular/advanced';
import { PolicyIteration, policyIterationMeta, ValueIteration, valueIterationMeta } from './agents/dp/dp';
import { ActorCriticAgent, actorCriticMeta, ReinforceAgent, reinforceMeta } from './agents/pg/pg';
import {
  EpsGreedyBandit,
  epsGreedyBanditMeta,
  GradientBandit,
  gradientBanditMeta,
  ThompsonBandit,
  thompsonMeta,
  UcbBandit,
  ucbMeta,
} from './agents/bandit/bandits';

export interface AlgoDef {
  meta: AlgoMeta;
  grid?: (env: GridEnv, p: ParamValues, rng: Rng) => GridAgent;
  planner?: (env: GridEnv, p: ParamValues) => GridPlanner;
  bandit?: (cfg: BanditConfig, p: ParamValues, rng: Rng) => BanditAgent;
}

/**
 * Every algorithm the app knows. To add one: implement GridAgent / GridPlanner / BanditAgent,
 * write its AlgoMeta, and add a line here. The UI, diff system, sweeps and 2P mode pick it up.
 */
export const ALGORITHMS: AlgoDef[] = [
  { meta: epsGreedyBanditMeta, bandit: (c, p, r) => new EpsGreedyBandit(c.k, p, r) },
  { meta: ucbMeta, bandit: (c, p, r) => new UcbBandit(c.k, p, r) },
  { meta: thompsonMeta, bandit: (c, p, r) => new ThompsonBandit(c.k, c.dist === 'bernoulli', p, r) },
  { meta: gradientBanditMeta, bandit: (c, p, r) => new GradientBandit(c.k, p, r) },
  { meta: valueIterationMeta, planner: (e, p) => new ValueIteration(e, p) },
  { meta: policyIterationMeta, planner: (e, p) => new PolicyIteration(e, p) },
  { meta: monteCarloMeta, grid: (e, p, r) => new MonteCarloAgent(e.S, p, r) },
  { meta: sarsaMeta, grid: (e, p, r) => new TDAgent(e.S, 'sarsa', p, r) },
  { meta: qLearningMeta, grid: (e, p, r) => new TDAgent(e.S, 'qlearning', p, r) },
  { meta: expectedSarsaMeta, grid: (e, p, r) => new TDAgent(e.S, 'expected', p, r) },
  { meta: nStepSarsaMeta, grid: (e, p, r) => new NStepSarsaAgent(e.S, p, r) },
  { meta: sarsaLambdaMeta, grid: (e, p, r) => new SarsaLambdaAgent(e.S, p, r) },
  { meta: dynaQMeta, grid: (e, p, r) => new DynaQAgent(e.S, p, r) },
  { meta: reinforceMeta, grid: (e, p, r) => new ReinforceAgent(e.S, p, r) },
  { meta: actorCriticMeta, grid: (e, p, r) => new ActorCriticAgent(e.S, p, r) },
];

const byId = new Map(ALGORITHMS.map((a) => [a.meta.id, a]));

export function getAlgo(id: string): AlgoDef {
  const a = byId.get(id);
  if (!a) throw new Error(`Unknown algorithm "${id}"`);
  return a;
}

export const algoMeta = (id: string): AlgoMeta => getAlgo(id).meta;

/** Algorithm defaults merged with the given values (unknown keys are dropped). */
export function resolveParams(id: string, p: ParamValues): ParamValues {
  const meta = algoMeta(id);
  const out = defaults(meta.params);
  for (const d of meta.params) if (p[d.id] !== undefined && typeof p[d.id] === typeof d.default) out[d.id] = p[d.id];
  return out;
}

export const FAMILY_LABELS: Record<AlgoMeta['family'], string> = {
  bandit: 'Bandits',
  dp: 'Planning (DP)',
  tabular: 'Tabular TD & MC',
  'policy-gradient': 'Policy gradient',
};
