from PySide6.QtCore import QObject, Property, Signal, Slot
import threading
import numpy as np
import time
import random
from puzzle_env import SlidingPuzzleEnv
from QlearningAgent import QLearningAgent
from A_StarAgent import AStarSolver

class PuzzleBridge(QObject):
    puzzle_list_changed = Signal()
    puzzle_size_changed = Signal()
    agent_training_progress_changed = Signal()
    invoke_start_btn_changed = Signal()
    invoke_generate_btn_changed = Signal()
    
    _instance = None  # Singleton instance
    
    @classmethod
    def get_instance(cls):
        """Returns the singleton instance of Bridge."""
        if cls._instance is None:
            cls._instance = PuzzleBridge()
        return cls._instance
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        super().__init__()
        self.puzzle_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.puzzle_size = 3
        t = threading.Thread(target=self.main_func, daemon=True)
        t.start()
        self.environment = SlidingPuzzleEnv(self.puzzle_size)
        self.step_per_sec = 0
        self.agent_type = None
        self.agent = None
        self.train_progress = 0.0
        self.train_episode_num = 1000
        self.invoke_start = False
        self.invoke_generate = True

    def set_puzzle_list(self, value):
        self.puzzle_list = value
        self.puzzle_list_changed.emit()
        
    def get_puzzle_list(self):
        return self.puzzle_list
    
    def set_puzzle_size(self, value):
        self.puzzle_size = value
        self.environment.set_size(value) # this part is working, but probebly need's to be considered changing in the future.
        self.puzzle_size_changed.emit()
        
    def get_puzzle_size(self):
        return self.puzzle_size
    
    def set_agent_training_progress(self, value):
        self.train_progress = value
        self.agent_training_progress_changed.emit()
        
    def get_agent_training_progress(self):
        return self.train_progress
    
    def set_invoke_start_btn(self, value): 
        '''
        makes sure to be started only if there is a puzzle generated: on the self.generate_new_puzzle func.
            self.set_invoke_start_btn(True)
        also, when the puzzle is solved, is being setted to false: on the solve func
            self.set_invoke_start_btn(False)
        
        also, for rl agents, is enabled only if the training is done. -> to-do
        '''
        self.invoke_start = value
        self.invoke_start_btn_changed.emit()
        
    def get_invoke_start_btn(self):
        return self.invoke_start
    
    def set_invoke_generate_btn(self, value):
        '''
        while solving the agent, makes sure that new puzzle is not tended to be generated.
        '''
        self.invoke_generate = value
        self.invoke_generate_btn_changed.emit()
        
    def get_invoke_generate_btn(self):
        return self.invoke_generate
    
    pyside_invoke_generate_btn = Property(bool, get_invoke_generate_btn, set_invoke_generate_btn, notify=invoke_generate_btn_changed)
    pyside_invoke_start_btn = Property(bool, get_invoke_start_btn, set_invoke_start_btn, notify=invoke_start_btn_changed)
    pyside_training_progress = Property(float, get_agent_training_progress, set_agent_training_progress, notify=agent_training_progress_changed)
    pyside_puzzle_list = Property(list, get_puzzle_list, set_puzzle_list, notify=puzzle_list_changed)
    pyside_puzzle_size = Property(int, get_puzzle_size, set_puzzle_size, notify=puzzle_size_changed)
            
    def generate_new_puzzle(self):
        state = self.environment.generate_puzzle().flatten()
        self.set_puzzle_list(state.tolist()) # each puzzle that is setted here, is going to be the one that is solved.
        self.set_invoke_start_btn(True)
        
    def train_agent(self): # to-do: all RL agents must contain the same format of training.
        # self.agent = QLearningAgent(self.environment) ### this part has to be handled. 
        self.agent = QLearningAgent(
            game_env=self.environment,
            learning_rate=0.1,
            discount_factor=0.95,   
            exploration_rate=1.0,
            epsilon_decay_rate=0.995,
            min_epsilon=0.01
        )
        self.agent.train(self.train_episode_num)
        
    def sovle_puzzle(self):
        print("start is triggered")
        self.solve_puzzle_A_star()
        
    def solve_puzzle_A_star(self): # to-do: integerate all solvers in a single format, so there wouldn't be a need to do all those in seperated funcs.
        self.agent = AStarSolver(env=self.environment)
        solution_steps = self.agent.solve()
        if solution_steps != None:
            only_states = [state for state, _ in solution_steps]
        else:
            self.invoke_error()
        t = threading.Thread(target=self.render, args=(only_states,), daemon=True)
        t.start()
        
    def render(self, solved_array):
        self.set_invoke_start_btn(False)
        self.set_invoke_generate_btn(False)
        for arr_ in solved_array:
            time.sleep(1 / self.step_per_sec)
            self.set_puzzle_list(arr_)
        self.set_invoke_generate_btn(True)
        print("finished rendering.")
        
    def main_func(self):
        # self.test_progress_bar()
        pass
    
    def test_puzzle(self, steps=10):
        n = self.puzzle_size
        puzzle = np.arange(1, n*n+1).reshape(n, n)
        puzzle[n-1, n-1] = 0  # Place blank (0) in bottom-right corner

        states = []

        states.append(puzzle.flatten().tolist())

        directions = [(-1, 0),  # move up
                    (1,  0),  # move down
                    (0, -1),  # move left
                    (0,  1)]  # move right

        for _ in range(steps):
            blank_row, blank_col = np.where(puzzle == 0)
            blank_row, blank_col = int(blank_row[0]), int(blank_col[0])

            valid_moves = []
            for d_row, d_col in directions:
                new_r, new_c = blank_row + d_row, blank_col + d_col
                if 0 <= new_r < n and 0 <= new_c < n:
                    valid_moves.append((d_row, d_col))

            d_row, d_col = random.choice(valid_moves)
            new_r, new_c = blank_row + d_row, blank_col + d_col

            puzzle[blank_row, blank_col], puzzle[new_r, new_c] = (
                puzzle[new_r, new_c],
                puzzle[blank_row, blank_col],
            )

            states.append(puzzle.flatten().tolist())
        return states
    
    def test_puzzle_flat(self, steps=10):
        n = self.puzzle_size
        initial_state = list(range(1, n*n)) + [0]
        
        states = [initial_state.copy()]
        state = initial_state.copy()
        
        for _ in range(steps):
            index = state.index(0)
            row, col = divmod(index, n)
            
            valid_moves = []
            if row > 0:
                valid_moves.append(index - n)
            if row < n - 1:
                valid_moves.append(index + n)
            if col > 0:
                valid_moves.append(index - 1)
            if col < n - 1:
                valid_moves.append(index + 1)
            
            neighbor_index = random.choice(valid_moves)
            state[index], state[neighbor_index] = state[neighbor_index], state[index]
            
            states.append(state.copy())
        
        return states
    
    def test_progress_bar(self):
        for i in range(101):
            time.sleep(0.1)
            self.set_agent_training_progress(i/100)