from PySide6.QtCore import QObject, Property, Signal, Slot
import threading
# import numpy as np

class PuzzleBridge(QObject):
    puzzle_list_changed = Signal()
    
    def __init__(self):
        super().__init__()
        self.puzzle_list = []
        t = threading.Thread(target=self.main_func, daemon=True)
        t.start()

    def set_puzzle_list(self, value):
        self.puzzle_list = value
        self.puzzle_list_changed.emit()
        
    def get_puzzle_list(self):
        return self.puzzle_list
    
    pyside_puzzle_list = Property(list, set_puzzle_list, get_puzzle_list, notify=puzzle_list_changed)
    
    def main_func(self):
        puzzle_3_3 = [1, 2, 3, 4, 0, 5, 6, 7, 8]
        self.set_puzzle_list(puzzle_3_3)