from PySide6.QtCore import QObject, Slot
from puzzle_handler import PuzzleBridge

class PuzzleSetting(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = PuzzleBridge.get_instance()
        
    @Slot(int)
    def setting_set_puzzle_size(self, value):
        '''sets the puzzle size, the parameter is the size of the puzzle grid'''
        self.bridge.set_puzzle_size(value)

    @Slot(str)
    def setting_set_agent_type(self, value):
        pass
    
    @Slot(int)
    def setting_set_solver_speed(self, value):
        '''sets the solver speed, the parameter is steps per second'''
        self.bridge.step_per_sec = value
        
    @Slot()
    def setting_initiate_train(self):
        pass
    
    @Slot()
    def setting_initiate_generate_puzzle(self):
        print("generate puzzle")
    
    @Slot()
    def setting_initiate_solve_puzzle(self):
        pass
    