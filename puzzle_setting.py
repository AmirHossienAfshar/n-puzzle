from PySide6.QtCore import QObject, Slot
from puzzle_handler import PuzzleBridge
from enum import Enum

class AgentType(Enum):
    Q_LEARNING = "Q_Learning"
    SARSA = "Sarsa"
    A_STAR = "A_star"
    BLIND = "Blind"
    HEURISTIC = "Heuristic"

class PuzzleSetting(QObject):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.bridge = PuzzleBridge.get_instance()
        
    @Slot(str)
    def setting_set_puzzle_size(self, value):
        '''sets the puzzle size, the parameter is the size of the puzzle grid'''
        if value == "3x3":
            num = 3
        elif value == "4x4":
            num = 4
        elif value == "5x5":
            num = 5
        elif value == "6x6":
            num = 6
        elif value == "7x7":
            num = 7
        elif value == "8x8":
            num = 8
        self.bridge.set_puzzle_size(num)

    @Slot(str)
    def setting_set_agent_type(self, value):
        '''sets the agent type that is going to solve the puzzle'''
        if value == "Q-Learning":
            agent = AgentType.Q_LEARNING
        elif value == "Sarsa":
            agent = AgentType.SARSA
        elif value == "A*":
            agent = AgentType.A_STAR
        elif value == "Blind":
            agent = AgentType.BLIND
        elif value == "Heuristic":
            agent = AgentType.HEURISTIC
        self.bridge.agent_type = agent
    
    @Slot(int) # working. dont mind.
    def setting_set_solver_speed(self, value):
        '''sets the solver speed, the parameter is steps per second'''
        self.bridge.step_per_sec = value
        
    @Slot(int)
    def setting_set_episode_number(self, value):
        '''sets the number of episodes that the agent is going to train'''
        self.bridge.train_episode_num = value
        
    @Slot()
    def setting_initiate_train(self):
        '''calls the handler's train agent function'''
        self.bridge.train_agent()
        
    @Slot()
    def setting_initiate_generate_puzzle(self):
        '''generates a new puzzle, which is garenteed to be a solvable one.'''
        self.bridge.generate_new_puzzle()
    
    @Slot()
    def setting_initiate_solve_puzzle(self):
        '''calls the handler's solve puzzle function'''
        self.bridge.sovle_puzzle()
        