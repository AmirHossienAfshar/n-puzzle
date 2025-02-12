from AbstractAgent import Agent
import numpy as np
from enum import Enum

class Move(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class QLearningAgent(Agent):
    def __init__(self, game_env, learning_rate, discount_factor, exploration_rate, epsilon_decay_rate=0.95, min_epsilon=0.01):
        super().__init__(game_env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon = min_epsilon
        self.action_space = list(Move)
        self.q_table = {} # Q-table to store Q-values for state-action pairs

    def select_action(self, state):
        """Epsilon-greedy action selection with valid move restriction."""
        possible_moves = self.env.get_possible_moves()

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in possible_moves}

        if np.random.rand() < self.exploration_rate:
            return np.random.choice(list(possible_moves.keys()))

        q_values = self.q_table[state]

        max_q = max(q_values[action] for action in possible_moves)

        best_actions = [action for action in possible_moves if q_values[action] == max_q]

        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        """Update Q-table using the Q-learning formula."""
        if state not in self.q_table:
            self.q_table[state] = {a: 0.0 for a in range(len(self.action_space))}

        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0.0 for a in range(len(self.action_space))}

        max_q = self.q_table[next_state][max(self.q_table[next_state], key=lambda x: self.q_table[next_state][x])]

        td_error = reward + self.discount_factor * max_q - self.q_table[state][action]

        self.q_table[state][action] += self.learning_rate * td_error

        return td_error

    def adjust_exploration_rate(self):
        """Decay exploration rate after each episode."""
        self.exploration_rate *= self.epsilon_decay_rate if self.exploration_rate * self.epsilon_decay_rate >= self.min_epsilon \
            else self.min_epsilon

    def train(self, episodes):
        pass