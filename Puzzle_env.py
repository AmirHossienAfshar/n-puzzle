import numpy as np
import gym
from gym import spaces
from enum import Enum

class Move(Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class SlidingPuzzleEnv(gym.Env):
    def __init__(self, size=3):
        super(SlidingPuzzleEnv, self).__init__()
        self.size = size
        self.goal_state = self._generate_goal_state()
        self.state = self._shuffle_board()

        # 4 possible moves: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4)

        # Observation space (flattened board representation)
        self.observation_space = spaces.Box(low=0, high=self.size**2-1, shape=(self.size**2,), dtype=np.int32)

    def _generate_goal_state(self):
        """Generate the solved puzzle as a NumPy array."""
        goal = np.arange(1, self.size**2).tolist() + [0]
        return np.array(goal).reshape(self.size, self.size)

    def _shuffle_board(self):
        """Shuffle while ensuring the board is solvable."""
        board = self.goal_state.flatten()
        np.random.shuffle(board)
        while not self.is_solvable(board):
            np.random.shuffle(board)
        return board.reshape(self.size, self.size)

    def is_solvable(self, board):
        """Check solvability using inversion count. even inversion counts means solvable."""
        board_list = board[board != 0]
        inv_count = sum(
            board_list[i] > board_list[j]
            for i in range(len(board_list))
            for j in range(i + 1, len(board_list))
        )
        return inv_count % 2 == 0

    def get_possible_moves(self):
        """Returns possible moves as action indices (0: Up, 1: Down, 2: Left, 3: Right)."""
        x, y = np.argwhere(self.state == 0)[0]
        moves = {}
        if x > 0:
            moves[Move.UP] = (-1, 0)
        if x < self.size - 1:
            moves[Move.DOWN] = (1, 0)
        if y > 0:
            moves[Move.LEFT] = (0, -1)
        if y < self.size - 1:
            moves[Move.RIGHT] = (0, 1)
        return moves

    def _apply_action(self, action):
        """Move the empty tile based on action if valid."""
        moves = self.get_possible_moves()
        if action in moves:
            dx, dy = moves[action]
            x, y = np.argwhere(self.state == 0)[0]
            nx, ny = x + dx, y + dy
            self.state[x, y], self.state[nx, ny] = self.state[nx, ny], self.state[x, y]
            return True
        return False

    def step(self, action):
        """Execute an action, return new state, reward, and done flag."""
        if self._apply_action(action):
            reward = -1
            done = np.array_equal(self.state, self.goal_state)  # Check if solved
            if done:
                reward = 100
            return self.state.flatten(), reward, done, {}

        return self.state.flatten(), -5, False, {}

    def reset(self):
        """Reset the board to a new shuffled state."""
        self.state = self._shuffle_board()
        return self.state.flatten()

    def render(self, mode="human"):
        """Render the puzzle board."""
        # print(self.state)
        return self.state

# env = SlidingPuzzleEnv(size=3)
# state = env.reset()
# env.render()