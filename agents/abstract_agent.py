from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, env):
        self.env = env

    @abstractmethod
    def select_action(self, state):
        """Select an action based on the current policy or value function."""
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the agent's policy or value function.
        For Policy Iteration, this could include Policy Evaluation and Policy Improvement.
        For Q-Learning and SARSA, this would update the Q-values.
        """
        pass

    def train(self, episodes):
        """
        Train the agent over multiple episodes.
        The child classes can override this method to handle specific training processes.
        """
        pass