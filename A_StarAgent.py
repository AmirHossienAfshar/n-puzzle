import heapq
import numpy as np
from Puzzle_env import SlidingPuzzleEnv

class AStarSolver:
    def __init__(self, env, heuristic="manhattan", max_explored_states=10000):
        self.env = env
        self.log_interval = 1000
        self.max_explored_states = max_explored_states
        
        if heuristic == "manhattan":
            self.heuristic_func = self.manhattan_distance
        elif heuristic == "misplaced_tiles":
            self.heuristic_func = self.misplaced_tiles
        else:
            raise ValueError("Unknown heuristic: {}".format(heuristic))
    
    def manhattan_distance(self, state, goal_state, size):
        """
        Compute the Manhattan distance between state and goal_state.
        Both are expected as tuples of length size*size.
        """
        distance = 0
        for index, tile in enumerate(state):
            if tile == 0:  # Ignore the blank.
                continue
            row, col = divmod(index, size)
            goal_index = goal_state.index(tile)
            goal_row, goal_col = divmod(goal_index, size)
            distance += abs(row - goal_row) + abs(col - goal_col)
        return distance

    def misplaced_tiles(self, state, goal_state):
        """
        Count the number of misplaced tiles (excluding the blank).
        """
        count = 0
        for s, g in zip(state, goal_state):
            if s != 0 and s != g:
                count += 1
        return count

    def get_neighbors(self, state, size):
        """
        Given a state (as a tuple) and puzzle size, return all neighboring states.
        Each neighbor is a tuple: (new_state, move) where move is (dr, dc).
        """
        neighbors = []
        state_list = list(state)
        zero_index = state_list.index(0)
        row, col = divmod(zero_index, size)
        
        moves = []
        if row > 0:
            moves.append((-1, 0))  # Up
        if row < size - 1:
            moves.append((1, 0))   # Down
        if col > 0:
            moves.append((0, -1))  # Left
        if col < size - 1:
            moves.append((0, 1))   # Right
        
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            new_index = new_row * size + new_col
            new_state = state_list.copy()

            new_state[zero_index], new_state[new_index] = new_state[new_index], new_state[zero_index]
            neighbors.append((tuple(new_state), (dr, dc)))
        return neighbors

    def solve(self):
        """
        Perform A* search to solve the puzzle.
        The initial state is taken from env.puzzle_to_solve (or generated if None),
        and the goal state is created using env._generate_goal_state().
        
        Returns:
            list: A list of tuples, where each tuple contains:
                  - A state (as a list) representing the puzzle configuration.
                  - The move that led to this state.

        Attributes:
            visited (set): A set of unique states that have already been expanded. 
                           This prevents revisiting the same state and speeds up the search.
        
            explored_states (int): The total number of states removed from the open set and processed. 
                                   This counts how many states the algorithm has explored in total.
        """
        # Get the initial state.
        if self.env.puzzle_to_solve is None:
            print("No puzzle found")
            exit(1)
        else:
            initial_np_state = self.env.puzzle_to_solve
        initial_state = tuple(int(x) for x in initial_np_state.tolist())
        
        # Get the goal state from the environment.
        goal_np_state = self.env._generate_goal_state().flatten()
        goal_state = tuple(int(x) for x in goal_np_state.tolist())
        
        size = self.env.size
        
        # Open set: each item is (f_score, g_score, state, path)
        start_h = self.heuristic_func(initial_state, goal_state, size)
        # open_set = [(start_h, 0, initial_state, [initial_state])]
        open_set = [(start_h, 0, initial_state, [(initial_state, None)])]  # Fix: Store moves
        visited = set()
        explored_states = 0
        
        while open_set:
            explored_states += 1
            f_score, g_score, current, path = heapq.heappop(open_set)
            
            if explored_states > self.max_explored_states:
                print(f"Terminating search after {self.max_explored_states} explored_states (limit reached).")
                return None
            
            if current == goal_state:
                print(f"Solution found after {explored_states} explore states, steps: {len(path)}")
                # return [list(state) for state in path]
                return [(list(state), move) for state, move in path]
            
            if current in visited:
                continue
            visited.add(current)
            
            # Log progress every log_interval iterations.
            if explored_states % self.log_interval == 0:
                print(f"explored states {explored_states}: open_set size = {len(open_set)}, current f_score = {f_score}, visited = {len(visited)}")
            
            for neighbor, move in self.get_neighbors(current, size):
                if neighbor in visited:
                    continue
                tentative_g = g_score + 1  # each move costs 1
                h = self.heuristic_func(neighbor, goal_state, size)
                f = tentative_g + h
                # heapq.heappush(open_set, (f, tentative_g, neighbor, path + [neighbor]))
                heapq.heappush(open_set, (f, tentative_g, neighbor, path + [(neighbor, move)]))

        
        print("No solution found!")
        return None