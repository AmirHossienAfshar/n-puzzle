from agents import Agent
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
        # self.action_space = list(Move) # this part will probebly make inconsistancy, so I made the line below:
        self.action_space = list(range(self.env.action_space.n))
        self.q_table = {} # Q-table to store Q-values for state-action pairs
        
    def get_state_key(self, state):
        """Convert a state (flattened array) into a hashable key (tuple)."""
        return tuple(state)
    
    def select_action(self, state):
        """Epsilon-greedy action selection with valid move restriction."""
        # print(f"the state to put action on, is: {state}")
        state_key = self.get_state_key(state)  # Convert state to a hashable key.
        possible_moves = self.env.get_possible_moves()  # This returns a dict with valid moves.
        # print(f"possible moves are {possible_moves}")

        if state_key not in self.q_table:
            self.q_table[state_key] = {action: 0.0 for action in possible_moves}

        if np.random.rand() < self.exploration_rate:
            # Edge case: if possible_moves is empty, choose randomly from full action space.
            if possible_moves:
                return np.random.choice(list(possible_moves.keys()))
            else:
                print("no possible moves found")
                exit(0)
                return np.random.choice(self.action_space)

        q_values = self.q_table[state_key]
        max_q = max(q_values[action] for action in possible_moves)
        best_actions = [action for action in possible_moves if q_values[action] == max_q]
        # print(f"best action to choose is {np.random.choice(best_actions)}")
        return np.random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0.0 for a in self.action_space}

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0.0 for a in self.action_space}

        current_q = self.q_table[state_key][action]
        max_next_q = max(self.q_table[next_state_key].values()) if not done else 0.0

        td_error = reward + self.discount_factor * max_next_q - current_q
        self.q_table[state_key][action] += self.learning_rate * td_error

        return td_error

    def adjust_exploration_rate(self):
        """Decay exploration rate after each episode."""
        self.exploration_rate *= self.epsilon_decay_rate if self.exploration_rate * self.epsilon_decay_rate >= self.min_epsilon \
            else self.min_epsilon
    
    def train(self, episodes):
        """Train the agent over a number of episodes."""
        i = 0
        for episode in range(episodes):
            # Reset environment to start a new episode.
            state = self.env.reset()  # state is a flattened NumPy array
            done = False
            step_count = 0
            total_reward = 0
            i+=1
            # print(i)

            while not done and step_count < 10000:
                # Choose an action using epsilon-greedy.
                action = self.select_action(state)
                # Execute the action.
                next_state, reward, done, _ = self.env.step(action)
                # Update Q-table.
                self.update(state, action, reward, next_state, done)
                state = next_state  # move to the next state
                total_reward += reward
                step_count += 1

            self.adjust_exploration_rate()

            # Optionally, print progress every so often.
            if (episode + 1) % 100 == 0:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.exploration_rate}")

        print("Training complete!")
    
    # def train(self, episodes):
    #     """Train the agent over a number of episodes."""
    #     for episode in range(episodes):
    #         state = self.env.reset()
    #         done = False
    #         total_reward = 0

    #         while not done:
    #             # Choose an action using epsilon-greedy.
    #             action = self.select_action(state)
    #             # Execute the action.
    #             next_state, reward, done, _ = self.env.step(action)
    #             # Update Q-table.
    #             self.update(state, action, reward, next_state, done)
    #             state = next_state  # Move to the next state
    #             total_reward += reward

    #         self.adjust_exploration_rate()

    #         # Optionally, print progress every so often.
    #         if (episode + 1) % 100 == 0:
    #             print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.exploration_rate}")

    #     print("Training complete!")

        
    def solve(self, puzzle_to_solve):
        """
        Solve the puzzle using the learned Q-table.
        'puzzle_to_solve' is expected to be a flat array.
        Returns a list of states (each state as a flat list) representing the solution path.
        """
        # Set the environment to the puzzle to solve.
        self.env.state = np.array(puzzle_to_solve).reshape(self.env.size, self.env.size)
        solution_states = [self.env.state.flatten().tolist()]  # record initial state
        done = np.array_equal(self.env.state, self.env.goal_state)
        steps = 0

        # while not done:
        while not done and steps < 10000:
            state = self.env.state.flatten()
            # print(state)
            state_key = self.get_state_key(state)
            
            # If the current state wasn't seen during training, fall back to a random valid move.
            if state_key not in self.q_table:
                action = self.select_action(state)
            else:
                # Greedily choose the best action from Q-table.
                q_values = self.q_table[state_key]
                action = max(q_values, key=q_values.get)
            
            next_state, reward, done, _ = self.env.step(action)
            solution_states.append(next_state.tolist())  # record the new state
            steps += 1

            if done:
                print(f"Solved in {steps} moves!")
                break

        return solution_states
    
    def print_q_table(self):
        table = self.q_table
        for state_key, action_values in table.items():
            # Convert state_key from tuple of np.int64 to tuple of ints
            state = tuple(int(x) for x in state_key)
            print(f"State: {state}")
            for action, q_val in action_values.items():
                # Convert q_val to a float if it is a numpy array
                if isinstance(q_val, np.ndarray):
                    q_val = q_val.item()
                print(f"  Action {action}: {q_val}")
            print("-" * 40)


    def solve_row_by_row(self, puzzle):
        n = self.env.puzzle_size
        for i in range(n):
            goal_state_n = self.create_goal_array_by_row(n, i+1)
            simplified_puzzle = self.mask_by_cumulative_rows(puzzle, i+1)
            # solve for this simplyed puzzles.
            
                 
    def create_goal_array_by_row(self, n, row):
        # Create an n x n array filled with -1
        arr = np.full((n, n), -1)
        
        # Fill the selected row with consecutive numbers starting from 1
        start_num = (row - 1) * n + 1
        arr[row - 1] = np.arange(start_num, start_num + n)
        
        # Set the last element of the last row to 0
        arr[-1, -1] = 0
        
        return arr
    
    def mask_by_cumulative_rows(self, arr, row):
        # Create a masked array filled with -1
        masked = np.full(arr.shape, -1)
        
        # Determine the cumulative row elements from a sorted array (1-counted)
        sorted_arr = np.sort(arr[arr != 0])  # Exclude zero before sorting
        cumulative_elements = sorted_arr[:row * arr.shape[1]]
        
        # Keep only elements from cumulative_elements and 0
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if arr[i, j] in cumulative_elements or arr[i, j] == 0:
                    masked[i, j] = arr[i, j]
        
        return masked




# from Puzzle_env import SlidingPuzzleEnv

# env = SlidingPuzzleEnv(size=3)
# agent = QLearningAgent(
#         game_env=env,
#         learning_rate=0.1,
#         discount_factor=0.99,
#         exploration_rate=1.0,
#         epsilon_decay_rate=0.95,
#         min_epsilon=0.01
#     )


# print(f"goal state is {agent.env.goal_state}")
# print("Generating a new puzzle...")
# agent.env.generate_puzzle()
# print(agent.env.puzzle_to_solve)

# print("Training the agent...")
# agent.train(1000)

# print("Solving the puzzle...")
# solved = agent.solve(agent.env.puzzle_to_solve)
# for i in solved:
#     print(i)
    
# agent.print_q_table(agent.q_table)