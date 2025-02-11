from PySide6.QtCore import QObject, Property, Signal, Slot
import threading
import numpy as np

class PuzzleBridge(QObject):
    puzzle_list_changed = Signal()
    puzzle_size_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.puzzle_list = []
        self.puzzle_size = 0
        t = threading.Thread(target=self.main_func, daemon=True)
        t.start()

    def set_puzzle_list(self, value):
        self.puzzle_list = value
        self.puzzle_list_changed.emit()
        
    def get_puzzle_list(self):
        return self.puzzle_list
    
    def set_puzzle_size(self, value):
        self.puzzle_size = value
        self.puzzle_size_changed.emit()
        
    def get_puzzle_size(self):
        return self.puzzle_size
    
    pyside_puzzle_list = Property(list, set_puzzle_list, get_puzzle_list, notify=puzzle_list_changed)
    pyside_puzzle_size = Property(int, set_puzzle_size, get_puzzle_size, notify=puzzle_size_changed)
    
    def main_func(self):
        puzzle_3_3 = np.array([[1, 2, 3],
                              [4, 0, 5],
                              [6, 7, 8]])
        puzzle_3_3 = puzzle_3_3.flatten()
        self.set_puzzle_list(puzzle_3_3)
        self.set_puzzle_size(3)