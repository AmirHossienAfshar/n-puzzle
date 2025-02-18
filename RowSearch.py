import numpy as np
import heapq

class EnhancedSearch:
    def __init__(self, env):
        self.env = env
        self.current_goal = np.full((env.size, env.size), -1)  
        # important: if it is considerd that the object kept alive, then be setted to defualt each time a new puzzle is made
        self.g_n = 0
        self.h_n = 0
                           
    def create_goal_array_by_row(self, n, row): #this one is fine
        arr = self.current_goal
        start_num = (row - 1) * n + 1
        arr[row - 1] = np.arange(start_num, start_num + n)
        
        goals = []

        if n == row:
            arr[-1, -1] = 0
            goals.append(arr)
            return goals
            
        for i in range(row, n):      # unsolved row indices
            for j in range(n):       # all columns in that row
                new_goal = arr.copy()
                new_goal[i, j] = 0
                goals.append(new_goal)
        
        return goals
    
    def mask_by_cumulative_rows(self, arr, row): #this one is fine
        arr = np.array(arr).reshape(self.env.size, self.env.size)
        masked = np.full(arr.shape, -1)
        
        sorted_arr = np.sort(arr[arr != 0])
        cumulative_elements = sorted_arr[:row * arr.shape[1]]
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] in cumulative_elements or arr[i, j] == 0:
                    masked[i, j] = arr[i, j]
        
        return masked
    
    def manhattan_distance(self, puzzle, goal_puzzle): #this one is fine
        """
        Compute the Manhattan distance between puzzle and goal_puzzle.
        Both are expected as np.array (2D).
        """
        size = self.env.size
        distance = 0
        
        puzzle_flat = puzzle.ravel()
        heuristic_goal = goal_puzzle[-1]
        goal_flat = heuristic_goal.ravel()
        
        for index, tile in enumerate(puzzle_flat):
            if tile == 0 or tile == -1:  # Ignore the blank or masked tiles
                continue
            
            # Find tile position in goal
            goal_index = np.where(goal_flat == tile)[0][0]
            
            # Convert indices to row/col
            row, col = divmod(index, size)
            goal_row, goal_col = divmod(goal_index, size)
            
            # Add Manhattan distance
            distance += abs(row - goal_row) + abs(col - goal_col)
        
        return distance

    
    def get_neighbors(self, puzzle): # I have to make sure that those dr and dc going to be needed at all. perhaps on reshaping the array?
                                        # ps: yes, they are going to be needed to re-implement the -1 -1 kind array
        '''
        new_puzzle (np.array): The 2D numpy array representing the puzzle state after making the move.
        (dr, dc) (tuple): The move made to reach that state, where:
        dr: Change in row (-1 for up, +1 for down)
        dc: Change in column (-1 for left, +1 for right)
        '''
        size = self.env.size
        neighbors = []
        
        # Find the position of the empty tile (0)
        zero_position = np.argwhere(puzzle == 0)[0]
        row, col = zero_position
        
        # Possible moves: Up, Down, Left, Right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < size and 0 <= new_col < size:
                # Create a copy to modify (using np.copy)
                new_puzzle = puzzle.copy()
                # Swap the zero with the neighbor
                new_puzzle[row, col], new_puzzle[new_row, new_col] = new_puzzle[new_row, new_col], new_puzzle[row, col]
                neighbors.append((new_puzzle, (dr, dc)))
        
        return neighbors

    
    def apply_steps(self, puzzle, steps):
        """
        Applies a sequence of moves to the given puzzle state.
        
        Each move in `steps` is expected to be a tuple (state, move),
        where `move` is itself a tuple (dr, dc) indicating the direction
        to slide the blank (0). The function returns the puzzle state after
        all moves have been applied.
        
        :param puzzle: np.array representing the current puzzle state.
        :param steps: List of moves [(state, (dr, dc))] to apply.
        :return: np.array representing the updated puzzle state.
        """
        applied_puzzle = puzzle.copy()
        
        for step in steps:
            # Extract the move (dr, dc); step is assumed to be (state, move)
            _, move = step
            dr, dc = move
            
            # Find the position of the blank tile (value 0)
            zero_pos = np.argwhere(applied_puzzle == 0)[0]
            r, c = zero_pos
            
            # Compute new position for the blank tile
            new_r, new_c = r + dr, c + dc
            
            # Swap the blank with the adjacent tile
            applied_puzzle[r, c], applied_puzzle[new_r, new_c] = (
                applied_puzzle[new_r, new_c],
                applied_puzzle[r, c],
            )
        
        return applied_puzzle

    
    def solve_row_by_row(self, puzzle, goal, steps_so_far):
        """
        Solve a simplified puzzle using A* search.
        :param puzzle: np.array (Simplified puzzle)
        :param goal: List of np.array (Possible goal states)
        :return: List of moves [(state, move)]
        """
        if steps_so_far:
            puzzle = self.apply_steps(puzzle, steps_so_far)

        print(f"Solving row for puzzle:\n{puzzle}")
        print(f"Target goal:\n{goal}")

        path = self.a_star_search(puzzle, goal)

        if path is None:
            print("No solution found for this row.")
            return []

        if len(path) == 0:
            print("Already solved! No moves needed.")
            return []

        print(f"Solved row with {len(path)} steps!")
        for step, (state, move) in enumerate(path, 1):
            print(f"Step {step} (Move {move}):\n{state}\n{'-' * 20}")

        return path

            
    def solve(self):
        puzzle = self.env.puzzle_to_solve
        n = self.env.size
        solve_steps = []
        
        for i in range(1, n + 1):
            goal_state_n = self.create_goal_array_by_row(n, i)
            # if i == 1:
            simplified_puzzle = self.mask_by_cumulative_rows(puzzle, i)
            # Solve for this simplified puzzle (using A*).
            steps = self.solve_row_by_row(simplified_puzzle, goal_state_n, solve_steps)
            solve_steps += steps
            # Update puzzle to the newly solved state.
            # puzzle = goal_state_n
            
        return solve_steps
                
 
    def a_star_search(self, start, goal):
        """
        Perform A* search from start to goal using Manhattan distance.
        :param start: np.array (start puzzle state)
        :param goal: List of np.array (goal puzzle states)
        :return: List of moves [(puzzle, move)] from start to goal
        """
        
        if any(np.array_equal(start, g) for g in goal):
            return []
        
        open_list = []
        
        goal_bytes_set = {g.tobytes() for g in goal}

        g_score = {start.tobytes(): 0}
        f_score = {start.tobytes(): self.manhattan_distance(start, goal)}

        heapq.heappush(open_list, (f_score[start.tobytes()], 0, start.tobytes(), []))

        visited = set()

        while open_list:
            _, cost, current_bytes, path = heapq.heappop(open_list)
            current_state = np.frombuffer(current_bytes, dtype=start.dtype).reshape(start.shape)

            if current_bytes in goal_bytes_set:
                return path

            if current_bytes in visited:
                continue
            visited.add(current_bytes)

            for neighbor, move in self.get_neighbors(current_state):
                neighbor_bytes = neighbor.tobytes()
                tentative_g = cost + 1

                if neighbor_bytes not in g_score or tentative_g < g_score[neighbor_bytes]:
                    g_score[neighbor_bytes] = tentative_g
                    f_score[neighbor_bytes] = tentative_g + self.manhattan_distance(neighbor, goal)
                    new_path = path + [(neighbor, move)]
                    heapq.heappush(open_list, (f_score[neighbor_bytes], tentative_g, neighbor_bytes, new_path))

        return []
        
            
from puzzle_env import SlidingPuzzleEnv

env = SlidingPuzzleEnv(size=3)
agent = EnhancedSearch(env)

env.generate_puzzle()
solve_steps = agent.solve()

print("\n===== Solution Steps =====\n")
print(agent.env.puzzle_to_solve)

for step, (state, move) in enumerate(solve_steps, 1):
    print(f"Step {step}: Move {move}\n")
    print(state)
    print("\n" + "-" * 20 + "\n")