/**
 * Plain-language explanations shown under each algorithm and under its formula.
 * Written for someone who has never heard of reinforcement learning. No jargon.
 * Every registered algorithm must have an entry (a test checks this).
 */

export interface PlainWords {
  /** what the method does, in one or two everyday sentences */
  simple: string;
  /** what its formula says, in words */
  formula: string;
}

export const PLAIN: Record<string, PlainWords> = {
  'eps-greedy': {
    simple: 'Usually plays the slot machine that has paid best so far, but now and then picks one at random, just in case another machine is secretly better.',
    formula: 'After each pull, move that machine’s score a little toward the reward you just got.',
  },
  ucb: {
    simple: 'Gives a bonus to machines it has not tried much, as if to say: “I’m not sure about this one yet, so let me check it.”',
    formula: 'Pick the machine with the best score plus an “unsure” bonus. The bonus shrinks the more often you try that machine.',
  },
  thompson: {
    simple: 'Keeps a rough guess of how good each machine might be. Each turn it imagines one possible version of the truth and plays the best machine in it.',
    formula: 'Draw a random guess for every machine from what you currently believe, then play the highest guess.',
  },
  'gradient-bandit': {
    simple: 'Builds up a liking for each machine. When a machine pays more than usual, it likes that machine a bit more and picks it more often.',
    formula: 'If the reward beat your average, like the chosen machine a bit more and the others a bit less (and the reverse if it was worse).',
  },
  'value-iteration': {
    simple: 'It already knows the map and the rules. It works out how good every square is by looking one step ahead, again and again, until the numbers stop changing.',
    formula: 'A square’s value = the reward of its best move now, plus a slightly shrunk value of the square that move leads to.',
  },
  'policy-iteration': {
    simple: 'Starts with a random plan, checks how good that plan is, then improves the plan. It repeats until the plan stops changing.',
    formula: 'Score every square under the current plan, then switch each square to the move that scores best.',
  },
  'monte-carlo': {
    simple: 'Plays a whole game first, then looks back and gives credit to every move based on the total score that game actually got.',
    formula: 'Move the score of each move you made toward the total reward that came after it.',
  },
  sarsa: {
    simple: 'Learns after every step, using the move it will really make next, even its random ones. That makes it careful around danger.',
    formula: 'New score = old score + a small step toward (reward now + value of the move I will actually make next).',
  },
  qlearning: {
    simple: 'Learns after every step, but always assumes its next move will be the best one, even while it is still experimenting. That makes it bold.',
    formula: 'New score = old score + a small step toward (reward now + value of the best possible next move).',
  },
  'expected-sarsa': {
    simple: 'Like SARSA, but instead of betting on one next move, it averages over all the moves it might make. Its learning is steadier.',
    formula: 'New score = old score + a small step toward (reward now + the average value of my possible next moves).',
  },
  'nstep-sarsa': {
    simple: 'Waits a few steps and uses those real rewards before guessing the rest, so good news travels back along the path faster.',
    formula: 'Add up the next n real rewards, then add a guess for everything after that, and move the old score toward this total.',
  },
  'sarsa-lambda': {
    simple: 'Leaves a fading trail of breadcrumbs behind it. When something good or bad happens, every square on the recent trail learns from it at once.',
    formula: 'Every recent move gets a share of the surprise. The newest moves get the biggest share.',
  },
  'dyna-q': {
    simple: 'Learns from each real move, then replays memories of earlier moves in its head to keep learning without taking extra steps.',
    formula: 'The same update as Q-learning, done once for the real step and n more times on remembered steps.',
  },
  reinforce: {
    simple: 'Does not score squares. It directly changes how likely it is to make each move: moves from good games become more likely, moves from bad games less likely.',
    formula: 'Make the moves you took more likely, in proportion to how much better than usual that game went.',
  },
  'actor-critic': {
    simple: 'Two parts work together: a critic guesses how good each square is, and an actor chooses the moves. After each step the critic says “better or worse than I expected”, and the actor adjusts.',
    formula: 'Surprise = reward + value of the next square − value of this square. Then push the chosen move’s chance in the direction of the surprise.',
  },
};

export const plainWords = (id: string): PlainWords | null => PLAIN[id] ?? null;
