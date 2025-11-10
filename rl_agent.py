%%writefile rl_agent.py
# rl_agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.98, epsilon=0.2, min_epsilon=0.01, decay=0.9995):
        self.n_states = n_states
        self.n_actions = n_actions
        self.Q = np.zeros((n_states, n_actions))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.decay = decay

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        else:
            # break ties randomly
            row = self.Q[state]
            maxv = np.max(row)
            candidates = np.where(row == maxv)[0]
            return random.choice(candidates.tolist())

    def learn(self, s, a, r, s_next):
        best_next = np.max(self.Q[s_next])
        td = r + self.gamma * best_next - self.Q[s, a]
        self.Q[s, a] += self.alpha * td
        # decay epsilon
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.decay
            self.epsilon = max(self.epsilon, self.min_epsilon)