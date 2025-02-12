from PySide6.QtCore import QObject, Property, Signal, Slot
import threading
import numpy as np
import time
import random
from puzzle_env import SlidingPuzzleEnv

class PuzzleBridge(QObject):
    puzzle_list_changed = Signal()
    puzzle_size_changed = Signal()
    
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
        self.puzzle_list = []
        self.puzzle_size = 0
        t = threading.Thread(target=self.main_func, daemon=True)
        t.start()
        self.environment = None

    def set_puzzle_list(self, value):
        self.puzzle_list = value
        self.puzzle_list_changed.emit()
        print(value)
        
    def get_puzzle_list(self):
        return self.puzzle_list
    
    def set_puzzle_size(self, value):
        self.puzzle_size = value
        self.puzzle_size_changed.emit()
        
    def get_puzzle_size(self):
        return self.puzzle_size
    
    pyside_puzzle_list = Property(list, get_puzzle_list, set_puzzle_list, notify=puzzle_list_changed)
    pyside_puzzle_size = Property(int, get_puzzle_size, set_puzzle_size, notify=puzzle_size_changed)
    
    def updated_model_elemnt(self, new_puzzle_list): # probebly won't be needed in the final version
        for i in range(self.puzzle_size):
            if self.puzzle_list[i] != new_puzzle_list[i]:
                self.puzzle_list[i] = new_puzzle_list[i]
                self.puzzle_list_changed.emit()
                time.sleep(0.2)
                
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
           
        print("done.")
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
    
    def main_func(self):
        
        self.puzzle_size = 5
        self.environment = SlidingPuzzleEnv(self.puzzle_size)
        state = self.environment.render().flatten()
        self.set_puzzle_list(state.tolist())
        
        # generated_states = self.test_puzzle(10)
        # for state in generated_states:
        #     self.set_puzzle_list(state)
        #     time.sleep(0.5)
        
        # generated_states = self.test_puzzle_flat(1000)
        # for state in generated_states:
        #     self.set_puzzle_list(state)
        #     # print(state)
        #     time.sleep(0.05)