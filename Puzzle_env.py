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
        
        # self.state_history = []
        # self.max_history_length = 50  # Number of steps to track
        # self.loop_threshold = 10       # Number of repeats to consider as a loop

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
    
    # def is_solvable(self, board: np.ndarray) -> bool:
    #     """
    #     Determines if a given sliding puzzle configuration is solvable based on inversion count and 
    #     the position of the empty tile.

    #     Solvability Rules:
    #     - If N (grid size) is odd, the puzzle is solvable if the number of inversions is even.
    #     - If N is even:
    #     - If the blank tile is on an even row from the bottom (2nd-last, 4th-last, ...), 
    #         the puzzle is solvable if the number of inversions is odd.
    #     - If the blank tile is on an odd row from the bottom (last, 3rd-last, ...), 
    #         the puzzle is solvable if the number of inversions is even.

    #     Parameters:
    #     -----------
    #     board : np.ndarray
    #         A flattened 1D representation of the puzzle grid.

    #     Returns:
    #     --------
    #     bool
    #         True if the puzzle is solvable, False otherwise.
    #     """
    #     board_list = board[board != 0]  # Remove the empty tile (0) for inversion counting
    #     inv_count = sum(
    #         board_list[i] > board_list[j]
    #         for i in range(len(board_list))
    #         for j in range(i + 1, len(board_list))
    #     )

    #     n = self.size  # Grid size (N x N)
        
    #     # Locate the empty tile (0)
    #     empty_tile_index = np.where(board == 0)[0][0]
    #     empty_tile_row = empty_tile_index // n  # Row index (0-based)
    #     blank_row_from_bottom = n - empty_tile_row  # Row position from bottom (1-based)

    #     if n % 2 == 1:  # Odd grid size
    #         return inv_count % 2 == 0
    #     else:  # Even grid size
    #         if blank_row_from_bottom % 2 == 0:  # Blank tile on even row from bottom
    #             return inv_count % 2 == 1  # Must have odd inversions
    #         else:  # Blank tile on odd row from bottom
    #             return inv_count % 2 == 0  # Must have even inversions
    
    def is_solvable(self, board: np.ndarray) -> bool:
        """
        Determine whether the given board is solvable.

        Args:
            board (np.ndarray): 2D NumPy array representing the puzzle.
                                The blank is represented by 0.

        Returns:
            bool: True if the puzzle is solvable, False otherwise.
        """
        n = self.size  # Grid size (n x n)
        flat = board.flatten()
        tiles = [tile for tile in flat if tile != 0]

        # Count inversions
        inversions = 0
        for i in range(len(tiles)):
            for j in range(i + 1, len(tiles)):
                if tiles[i] > tiles[j]:
                    inversions += 1

        # Get row index (0-based) of blank from top
        blank_index = flat.tolist().index(0)
        blank_row = blank_index // n

        if n % 2 == 1:
            return inversions % 2 == 0
        else:
            return (inversions + blank_row) % 2 == 1



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
        # print(state)
        # print(goal_state)
        distance = 0
        for num in range(1, state.size):  # Ignore 0 (empty space)
            x, y = np.where(state == num)  # Find current position
            gx, gy = np.where(goal_state == num)  # Find goal position
            distance += abs(x - gx) + abs(y - gy)
        return distance
    
    def step(self, action): # this version uses menhattan distance for the reward managment
        """Execute an action and return state, reward, and done flag."""
        prev_distance = self.manhattan_distance(self.state, self.goal_state)

        if self._apply_action(action):
            new_distance = self.manhattan_distance(self.state, self.goal_state)

            reward = 5 * (prev_distance - new_distance) - 0.5

            if np.array_equal(self.state, self.goal_state):
                reward = 1000

            return self.state.flatten(), reward, np.array_equal(self.state, self.goal_state), {}

        # Return heavy penalty for invalid moves.
        return self.state.flatten(), -5, False, {}
    
    # def step(self, action):
    #     prev_distance = self.manhattan_distance(self.state, self.goal_state)
    #     if self._apply_action(action):
    #         new_distance = self.manhattan_distance(self.state, self.goal_state)
    #         reward = 5 * (prev_distance - new_distance) - 0.5

    #         # Check if goal state is reached
    #         if np.array_equal(self.state, self.goal_state):
    #             reward = 1000
    #             done = True
    #         else:
    #             done = False

    #         # Update state history
    #         self.state_history.append(self.state.flatten())
    #         if len(self.state_history) > self.max_history_length:
    #             self.state_history.pop(0)

    #         # Check for loops
    #         repeats = self.check_for_loops()
    #         if repeats >= self.loop_threshold:
    #             reward -= 20  # Penalize for getting stuck
    #             done = True   # Terminate the episode

    #         return self.state.flatten(), reward, done, {}
    #     else:
    #         # Invalid move
    #         return self.state.flatten(), -5, False, {}

    # def check_for_loops(self):
    #     """Count the maximum number of times any state appears in the history."""
    #     state_counts = {}
    #     for s in self.state_history:
    #         s_key = tuple(s)
    #         state_counts[s_key] = state_counts.get(s_key, 0) + 1
    #     max_repeats = max(state_counts.values())
    #     return max_repeats
    
    # def step(self, action): # this one is the one that uses a list, for privous visited states. doesn't make sence for the memory managment
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


    # def step(self, action): # old version, that didn't make sense because of the simplecity.
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
        # print(f"[LOG TO CHECK] {self.state}")
        return self.state.flatten()

    def reset(self):
        """Reset the board to a new shuffled state."""
        self.state = self._shuffle_board()
        # self.state_history = []
        return self.state.flatten()

    def render(self, mode="human"): # this part has to be make sure, how that this render function is used in the main.py
        """Render the puzzle board."""
        # print(self.state)
        return self.state

# env = SlidingPuzzleEnv(size=3)
# state = env.reset()
# env.render()



# import numpy as np

# def is_solvable(board):
#     """Check solvability of an N×N sliding puzzle."""
#     N = int(np.sqrt(len(board)))  # Determine the grid size (assuming square grid)
#     board = np.array(board).reshape((N, N))  # Reshape into N×N format
    
#     board_list = board.flatten()  # Convert to 1D list
#     board_list = board_list[board_list != 0]  # Remove the blank (0)

#     # Count inversions
#     inv_count = sum(
#         board_list[i] > board_list[j]
#         for i in range(len(board_list))
#         for j in range(i + 1, len(board_list))
#     )

#     if N % 2 == 1:  # Odd grid (3x3, 5x5, etc.)
#         return inv_count % 2 == 0  # Solvable if even inversion count

#     else:  # Even grid (4x4, 6x6, etc.)
#         blank_row = N - np.where(board == 0)[0][0]  # Row index from bottom
#         if (blank_row % 2 == 0):  # Blank is on an even row from bottom
#             return inv_count % 2 == 1  # Inversion count must be odd
#         else:  # Blank is on an odd row from bottom
#             return inv_count % 2 == 0  # Inversion count must be even

# # Given 4x4 puzzle board
# puzzle_board = [1, 9, 13, 10, 12, 2, 14, 7, 11, 15, 3, 6, 0, 4, 5, 8]

# # Check solvability
# solvable = is_solvable(puzzle_board)
# print(solvable)
