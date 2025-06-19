import heapq
import numpy as np
from collections import deque

class Search:
    def __init__(self, env, heuristic="manhattan", max_explored_states=10000000):
        self.env = env
        self.log_interval = 1000
        self.max_explored_states = max_explored_states
        
        if heuristic == "manhattan":
            self.heuristic_func = self.manhattan_distance
        elif heuristic == "misplaced_tiles":
            self.heuristic_func = self.misplaced_tiles
        else:
            raise ValueError("Unknown heuristic: {}".format(heuristic))
        
        # this two below belong to the row greedy algorithm. don't mind that
        self.solve_states = [] 
        self.current_goal = None
            
    def manhattan_distance(self, state, goal_state):
        """
        Compute the Manhattan distance between state and goal_state.
        Both are expected as tuples of length size*size.
        """
        size = self.env.size
        distance = 0
        for index, tile in enumerate(state):
            if tile == 0 or tile == -1:  # Ignore the blank. -1 is for the row greedy algorithm
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

    def get_neighbors(self, state):
        """
        Given a state (as a tuple) and puzzle size, return all neighboring states.
        Each neighbor is a tuple: (new_state, move) where move is (dr, dc).
        """
        size = self.env.size
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

    def solve_A_start(self, initial_state=None, goal_state=None):
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
        if initial_state is None:
            if self.env.puzzle_to_solve is None:
                print("No puzzle found")
                exit(1)
            else:
                initial_np_state = self.env.puzzle_to_solve
            initial_state = tuple(int(x) for x in initial_np_state.tolist())
        
        # Get the goal state from the environment.
        if goal_state is None:
            goal_np_state = self.env._generate_goal_state().flatten()
            goal_state = tuple(int(x) for x in goal_np_state.tolist())
        
        size = self.env.size
        
        # Open set: each item is (f_score, g_score, state, path)
        start_h = self.heuristic_func(initial_state, goal_state)
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

            for neighbor, move in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                tentative_g = g_score + 1  # each move costs 1
                h = self.heuristic_func(neighbor, goal_state)
                f = tentative_g + h
                # heapq.heappush(open_set, (f, tentative_g, neighbor, path + [neighbor]))
                heapq.heappush(open_set, (f, tentative_g, neighbor, path + [(neighbor, move)]))

        
        print("No solution found!")
        return None
    
    def solve_bfs(self):
        """
        Perform Breadth-First Search (BFS) to solve the puzzle.
        
        Returns:
            list: A list of tuples, where each tuple contains:
                - A state (as a list) representing the puzzle configuration.
                - The move that led to this state.

        Logs:
            - Explored states count
            - Depth of the current node
            - Queue size
        """
        # Get the initial state
        if self.env.puzzle_to_solve is None:
            print("No puzzle found")
            exit(1)
        else:
            initial_np_state = self.env.puzzle_to_solve
        initial_state = tuple(int(x) for x in initial_np_state.tolist())

        # Get goal state
        goal_np_state = self.env._generate_goal_state().flatten()
        goal_state = tuple(int(x) for x in goal_np_state.tolist())

        size = self.env.size

        # BFS uses a queue
        queue = deque()
        queue.append((initial_state, [(initial_state, None)], 0))  # (state, path, depth)

        visited = set()
        explored_states = 0

        while queue:
            current, path, depth = queue.popleft()
            explored_states += 1

            if explored_states > self.max_explored_states:
                print(f"Terminating BFS after {self.max_explored_states} explored_states (limit reached).")
                return None

            if current == goal_state:
                print(f"Solution found at depth {depth} after {explored_states} explored states.")
                return [(list(state), move) for state, move in path]

            if current in visited:
                continue
            visited.add(current)

            if explored_states % self.log_interval == 0:
                print(f"explored states {explored_states}: depth = {depth}, queue size = {len(queue)}, visited = {len(visited)}")

            for neighbor, move in self.get_neighbors(current):
                if neighbor in visited:
                    continue
                queue.append((neighbor, path + [(neighbor, move)], depth + 1))

        print("No solution found!")
        return None


    def solve_dfs(self):
        """
        Perform Depth-First Search (DFS) to solve the puzzle.

        Returns:
            list: A list of tuples, where each tuple contains:
                - A state (as a list) representing the puzzle configuration.
                - The move that led to this state.

        Logs:
            - Explored states count
            - Depth of the current node
            - Stack size
        """
        # Get the initial state
        if self.env.puzzle_to_solve is None:
            print("No puzzle found")
            exit(1)
        else:
            initial_np_state = self.env.puzzle_to_solve
        initial_state = tuple(int(x) for x in initial_np_state.tolist())

        # Get goal state
        goal_np_state = self.env._generate_goal_state().flatten()
        goal_state = tuple(int(x) for x in goal_np_state.tolist())

        size = self.env.size

        # DFS uses a stack (LIFO)
        stack = []
        stack.append((initial_state, [(initial_state, None)], 0))  # (state, path, depth)

        visited = set()
        explored_states = 0

        while stack:
            current, path, depth = stack.pop()
            explored_states += 1

            if explored_states > self.max_explored_states:
                print(f"Terminating DFS after {self.max_explored_states} explored_states (limit reached).")
                return None

            if current == goal_state:
                print(f"Solution found at depth {depth} after {explored_states} explored states.")
                return [(list(state), move) for state, move in path]

            if current in visited:
                continue
            visited.add(current)

            if explored_states % self.log_interval == 0:
                print(f"explored states {explored_states}: depth = {depth}, stack size = {len(stack)}, visited = {len(visited)}")

            for neighbor, move in reversed(self.get_neighbors(current)):
                if neighbor in visited:
                    continue
                stack.append((neighbor, path + [(neighbor, move)], depth + 1))

        print("No solution found!")
        return None
    
        
    def solve_ids(self, max_depth_limit=3000):
        """
        Perform Iterative Deepening Search (IDS) with a maximum depth limit.

        Args:
            max_depth_limit (int): Maximum depth limit for IDS to avoid infinite depth.

        Returns:
            list: A list of tuples (state as list, move) representing the solution path,
                or None if no solution found within the depth limit.

        Logs:
            - Current depth limit per iteration
            - Total explored states
            - Stack size and visited count periodically
        """

        if self.env.puzzle_to_solve is None:
            print("No puzzle found")
            exit(1)
        else:
            initial_np_state = self.env.puzzle_to_solve
        initial_state = tuple(int(x) for x in initial_np_state.tolist())

        goal_np_state = self.env._generate_goal_state().flatten()
        goal_state = tuple(int(x) for x in goal_np_state.tolist())

        size = self.env.size
        explored_states_total = 0

        def dls(limit):
            nonlocal explored_states_total
            stack = [(initial_state, [(initial_state, None)], 0)]
            visited = set()

            while stack:
                current, path, depth = stack.pop()
                explored_states_total += 1

                if explored_states_total > self.max_explored_states:
                    print("Terminating IDS: max_explored_states reached.")
                    return None

                if current == goal_state:
                    print(f"Solution found at depth {depth} after {explored_states_total} explored states.")
                    return [(list(state), move) for state, move in path]

                if depth < limit and current not in visited:
                    visited.add(current)

                    if explored_states_total % self.log_interval == 0:
                        print(f"[depth {limit}] explored = {explored_states_total}, current depth = {depth}, stack size = {len(stack)}, visited = {len(visited)}")

                    for neighbor, move in reversed(self.get_neighbors(current)):
                        if neighbor not in visited:
                            stack.append((neighbor, path + [(neighbor, move)], depth + 1))

            return None

        # Iterative deepening loop
        for depth_limit in range(max_depth_limit + 1):
            print(f"Starting DLS iteration with depth limit: {depth_limit}")
            result = dls(depth_limit)
            if result is not None:
                return result

        print(f"No solution found up to depth {max_depth_limit}.")
        return None
    
    
    def solve_row_greedy(self):
        puzzle = self.env.puzzle_to_solve
        n = self.env.size
        
        # initial setup for each run
        self.current_goal = np.full((self.env.size, self.env.size), -1)
        self.solve_states = [] 
        
        solve_steps = []
        for i in range(1, n + 1):
            goal_state_n = self.create_goal_array_by_row(n, i)
            simplified_puzzle = self.mask_by_cumulative_rows(puzzle, i)
            steps = self.solve_row_by_row(simplified_puzzle, goal_state_n, solve_steps)
            solve_steps += steps
                  
        self.solve_states = solve_steps
        
        return self.reconstruct_solution_steps()
        
        
    # this function is used for row greedy. that differes from the other A*, because that is using 
    # multiple goals to check, and for a better readability, I used this seperated custum A*.
    def custum_A_star_search(self, start, goal):
        """
        Perform A* search from start to goal using Manhattan distance.
        :param start: np.array (start puzzle state)
        :param goal: List of np.array (goal puzzle states)
        :return: List of moves [(puzzle, move)] from start to goal
        """
        # print(f"start is {start}")
        # print(f"goal is {goal}")
        if any(np.array_equal(start, g) for g in goal):
            return []
        
        open_list = []
        
        
        goal_bytes_set = {g.tobytes() for g in goal}
        
        # print(f"goal_bytes_set is{goal_bytes_set}")
        
        puzzle_flat = start.ravel()
        heuristic_goal = goal[-1]
        goal_flat = heuristic_goal.ravel()

        g_score = {start.tobytes(): 0}
        f_score = {start.tobytes(): self.heuristic_func(puzzle_flat.tolist(), goal_flat.tolist())}

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
            
            # print(f"current_state is{current_state.tolist()}")
            # print(f"current_state type is{type(current_state.tolist())}")
            
            # print(f"current_state is{tuple(current_state.tolist())}")
            # print(f"current_state type is{type(tuple(current_state.tolist()))}")
            
            
            current_state_ = []
            for i in current_state:
                for j in i:
                    current_state_.append(int(j))
            current_state_ = tuple(current_state_)
            current_state_ = self.get_neighbors(current_state_)
            
            for neighbor, move in current_state_:
                neighbor = np.array(neighbor).reshape(self.env.size, self.env.size)

                neighbor_flat = neighbor.ravel()
                neighbor_bytes = neighbor.tobytes()
                tentative_g = cost + 1

                if neighbor_bytes not in g_score or tentative_g < g_score[neighbor_bytes]:
                    g_score[neighbor_bytes] = tentative_g
                    f_score[neighbor_bytes] = tentative_g + self.heuristic_func(neighbor_flat.tolist(), goal_flat.tolist())
                    new_path = path + [(neighbor, move)]
                    heapq.heappush(open_list, (f_score[neighbor_bytes], tentative_g, neighbor_bytes, new_path))

        return []
    
    def reconstruct_solution_steps(self):
        """
        Reconstructs the sequence of solved steps in a 1D flattened format by applying 
        each recorded move to the initial puzzle state.

        Returns:
        --------
        list[list[int]]
            A list where each entry represents the puzzle state (flattened 1D) at each step.
        """
        if not self.solve_states:
            return []

        puzzle = self.env.puzzle_to_solve.copy()  # Start from the initial puzzle state
        puzzle = np.array(puzzle).reshape(self.env.size, self.env.size)
        reconstructed_steps = [(puzzle.flatten().tolist(), None)]  # Store the initial state as a list
        
        for step, (state, move) in enumerate(self.solve_states):
            dr, dc = move  # Extract move direction
            zero_pos = np.argwhere(puzzle == 0)[0]  # Find empty tile position
            r, c = zero_pos

            # Compute the new position for the empty tile
            new_r, new_c = r + dr, c + dc

            # Apply the move (swap)
            puzzle[r, c], puzzle[new_r, new_c] = puzzle[new_r, new_c], puzzle[r, c]

            # Store the new state as a regular Python list
            reconstructed_steps.append((puzzle.flatten().tolist(), move))

        return reconstructed_steps
    
    def solve_row_by_row(self, puzzle, goal, steps_so_far):
        """
        Solve a simplified puzzle using A* search.
        :param puzzle: np.array (Simplified puzzle)
        :param goal: List of np.array (Possible goal states)
        :return: List of moves [(state, move)]
        """
        applied_puzzle = puzzle.copy()
        if steps_so_far:        
            for step in steps_so_far:
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
            
        path = self.custum_A_star_search(applied_puzzle, goal)

        if path is None:
            return []

        if len(path) == 0:
            return []

        return path
    
    def mask_by_cumulative_rows(self, arr: list[list[int]], row: int) -> np.ndarray:
        """
        Masks an input `n x n` puzzle grid by revealing only the smallest `row * n` elements 
        in a row-wise cumulative fashion, while replacing other elements with `-1`.

        The function sorts all non-zero elements, selects the smallest `row * n` values, and 
        creates a masked version of the grid where only those selected values and zeros remain 
        visible.

        Parameters:
        -----------
        arr : list[list[int]]
            The `n x n` grid of integers representing a puzzle state.
        row : int
            The number of rows to cumulatively reveal based on the smallest values.

        Returns:
        --------
        np.ndarray
            A masked version of the input grid, where only `row * n` smallest elements 
            (plus any zeros) are retained, and all other elements are set to `-1`.

        Example:
        --------
        Input:
        ```
        arr = [[8, 3, 7, 4],
            [2, 1, 6, 5],
            [12, 11, 9, 10],
            [16, 15, 14, 13]]
        row = 2
        ```

        Sorted non-zero values:
        ```
        [1, 2, 3, 4, 5, 6, 7, 8]
        ```

        Output:
        ```
        [[-1,  3, -1,  4]
        [ 2,  1, -1,  5]
        [-1, -1, -1, -1]
        [-1, -1, -1, -1]]
        ```

        Notes:
        ------
        - `self.env.size` determines the grid size (`n`).
        - The function ensures that `0` values remain unchanged.
        - Useful for progressively revealing parts of a solved puzzle while maintaining hidden information.
        """
        arr = np.array(arr).reshape(self.env.size, self.env.size)
        masked = np.full(arr.shape, -1)
        
        sorted_arr = np.sort(arr[arr != 0])
        cumulative_elements = sorted_arr[:row * arr.shape[1]]
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] in cumulative_elements or arr[i, j] == 0:
                    masked[i, j] = arr[i, j]
        
        return masked
    
    def create_goal_array_by_row(self, n: int, row: int) -> list[np.ndarray]:
        """
        Generates multiple goal states for an n x n sliding puzzle by modifying a specific row 
        and allowing variations of the empty tile (0) in the remaining unsolved rows.

        The function updates `self.current_goal` by setting the specified row (`row - 1`) 
        to a sequential number sequence. It then creates multiple variations of the goal state 
        by placing the empty tile (`0`) in different locations in the remaining rows.

        Parameters:
        -----------
        n : int
            The size of the sliding puzzle (n x n grid).
        row : int
            The row (1-based index) that should be finalized in the goal state.

        Returns:
        --------
        list[np.ndarray]
            A list of `n x n` NumPy arrays representing possible goal states.
            If `row == n`, only one goal state is returned with the last tile set to 0.
            Otherwise, multiple goal states are generated by placing the empty tile in different 
            positions in the unsolved rows.

        Example:
        --------
        Suppose `self.current_goal` is a `4x4` grid and `row = 2`:

        ```
        [[ 1  2  3  4]
        [ 5  6  7  8]
        [ 9 10 11 12]
        [13 14 15  0]]
        ```

        The function will first ensure that row 2 is `[5, 6, 7, 8]`. Then it will return 
        multiple goal states where the empty tile (`0`) is placed at different positions 
        in rows 3 and 4, like:

        ```
        [[ 1  2  3  4]     [[ 1  2  3  4]
        [ 5  6  7  8]      [ 5  6  7  8]
        [ 9 10 11  0]      [ 9 10 11 12]
        [13 14 15 12]]     [13 14  0 15]]
        ```

        Notes:
        ------
        - If `row == n`, the function only sets the last tile in the bottom-right to `0` and returns.
        - This function is useful for defining goal states at different levels of puzzle solving.
        """
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
