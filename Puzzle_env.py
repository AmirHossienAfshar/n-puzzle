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
        self.puzzle_to_solve = None

        # 4 possible moves: Up, Down, Left, Right
        self.action_space = spaces.Discrete(4) ########################################### is this part consistent to the other explanations?
        
        self.visited_states = [] #?

        # Observation space (flattened board representation)
        self.observation_space = spaces.Box(low=0, high=self.size**2-1, shape=(self.size**2,), dtype=np.int32)
        
    def set_size(self, size): # this part has to be double checked, to make sure that this generate goal test and suffle board are neccecary
        self.size = size
        self.goal_state = self._generate_goal_state()
        self.state = self._shuffle_board()
        
    def set_puzzle_to_solve(self, value):
        self.puzzle_to_solve = value
        
    def get_puzzle_to_solve(self):
        return self.puzzle_to_solve

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

    # def get_possible_moves(self):
    #     """Returns possible moves as action indices (0: Up, 1: Down, 2: Left, 3: Right)."""
    #     x, y = np.argwhere(self.state == 0)[0]
    #     moves = {}
    #     if x > 0:
    #         moves[Move.UP] = (-1, 0)
    #     if x < self.size - 1:
    #         moves[Move.DOWN] = (1, 0)
    #     if y > 0:
    #         moves[Move.LEFT] = (0, -1)
    #     if y < self.size - 1:
    #         moves[Move.RIGHT] = (0, 1)
    #     return moves
    
    def get_possible_moves(self):
        """Returns possible moves as action indices (0: Up, 1: Down, 2: Left, 3: Right)."""
        x, y = np.argwhere(self.state == 0)[0]
        moves = {}
        if x > 0:
            moves[0] = (-1, 0)  # Up
        if x < self.size - 1:
            moves[1] = (1, 0)   # Down
        if y > 0:
            moves[2] = (0, -1)  # Left
        if y < self.size - 1:
            moves[3] = (0, 1)   # Right
        return moves

    # def _apply_action(self, action):
    #     """Move the empty tile based on action if valid."""
    #     # moves = self.get_possible_moves()
    #     # if action in moves:
    #     #     dx, dy = moves[action]
    #     #     x, y = np.argwhere(self.state == 0)[0]
    #     #     nx, ny = x + dx, y + dy
    #     #     self.state[x, y], self.state[nx, ny] = self.state[nx, ny], self.state[x, y]
    #     #     return True
    #     # return False # this ond one had inconsiency with the enumeration. 
    #     action_map = {
    #     0: Move.UP,
    #     1: Move.DOWN,
    #     2: Move.LEFT,
    #     3: Move.RIGHT
    #     }
    #     moves = self.get_possible_moves()
    #     if action_map[action] in moves:
    #         dx, dy = moves[action_map[action]]
    #         x, y = np.argwhere(self.state == 0)[0]
    #         nx, ny = x + dx, y + dy
    #         self.state[x, y], self.state[nx, ny] = self.state[nx, ny], self.state[x, y]
    #         return True
    #     return False
    
    def _apply_action(self, action):
        """Move the empty tile based on action if valid."""
        moves = self.get_possible_moves()  # keys are integers: 0, 1, 2, 3
        if action in moves:
            dx, dy = moves[action]
            x, y = np.argwhere(self.state == 0)[0]
            nx, ny = x + dx, y + dy
            self.state[x, y], self.state[nx, ny] = self.state[nx, ny], self.state[x, y]
            return True
        return False
    
    def manhattan_distance(self, state, goal_state):
        """Compute the Manhattan distance between state and goal_state."""
        distance = 0
        for num in range(1, state.size):  # Ignore 0 (empty space)
            x, y = np.where(state == num)  # Find current position
            gx, gy = np.where(goal_state == num)  # Find goal position
            distance += abs(x - gx) + abs(y - gy)
        return distance
    
    def step(self, action):
        """Execute an action and return state, reward, and done flag."""
        prev_distance = self.manhattan_distance(self.state, self.goal_state)

        if self._apply_action(action):
            new_distance = self.manhattan_distance(self.state, self.goal_state)
            # Reward based on improvement plus a small constant penalty per step.
            reward = (prev_distance - new_distance) - 0.5

            # If the puzzle is solved, override with a large reward.
            if np.array_equal(self.state, self.goal_state):
                reward = 100

            return self.state.flatten(), reward, np.array_equal(self.state, self.goal_state), {}

        # Return heavy penalty for invalid moves.
        return self.state.flatten(), -5, False, {}
    
    # def step(self, action):
    #     # Use an episode-level visited set (stored in the environment or agent).
    #     state_key = tuple(self.state.flatten())

    #     prev_distance = self.manhattan_distance(self.state, self.goal_state)
    #     if self._apply_action(action):
    #         new_distance = self.manhattan_distance(self.state, self.goal_state)
    #         reward = 2.0 * (prev_distance - new_distance) - 0.5  # Weighted improvement + step penalty

    #         if state_key in self.visited_states:
    #             reward -= 2.0  # Extra penalty for revisiting a known state
    #         else:
    #             self.visited_states.append(state_key)

    #         if np.array_equal(self.state, self.goal_state):
    #             reward = 100

    #         return self.state.flatten(), reward, np.array_equal(self.state, self.goal_state), {}

    #     return self.state.flatten(), -5, False, {}


    # def step(self, action): # old version, that didn't make sense
    #     """Execute an action, return new state, reward, and done flag."""
    #     if self._apply_action(action):
    #         reward = -1
    #         done = np.array_equal(self.state, self.goal_state)
    #         if done:
    #             reward = 100
    #         return self.state.flatten(), reward, done, {}

    #     return self.state.flatten(), -5, False, {}
    
    def generate_puzzle(self):
        """this part is for the initial puzzle, not the one that is going to learn form"""
        self.state = self._shuffle_board()
        self.puzzle_to_solve = self.state.flatten()
        return self.state.flatten()

    def reset(self):
        """Reset the board to a new shuffled state."""
        self.state = self._shuffle_board()
        return self.state.flatten()

    def render(self, mode="human"): # this part has to be make sure, how that this render function is used in the main.py
        """Render the puzzle board."""
        # print(self.state)
        return self.state

# env = SlidingPuzzleEnv(size=3)
# state = env.reset()
# env.render()

