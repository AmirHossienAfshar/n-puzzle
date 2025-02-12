from PySide6.QtCore import QObject, Slot
from puzzle_handler import PuzzleBridge

class PuzzleSetting(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = PuzzleBridge.get_instance()
        
    @Slot(int)
    def setting_set_puzzle_size(self, value):
        self.bridge.set_puzzle_size(value)

    @Slot(str)
    def setting_set_agent_type(self, value):
        pass
    
    @Slot(int)
    def setting_set_solver_speed(self, value):
        pass
        
    @Slot()
    def setting_initiate_train(self):
        pass
    
    @Slot()
    def setting_initiate_generate_puzzle(self):
        pass
    
    @Slot()
    def setting_initiate_solve_puzzle(self):
        pass
    