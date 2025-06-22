from PySide6.QtCore import QObject, Property, Signal, QThreadPool
from puzzle_env import SlidingPuzzleEnv
from QlearningAgent import QLearningAgent
from search import Search
from enum import Enum
from engine.worker import Worker
import threading
import time

class AgentType(Enum):
    Q_LEARNING = "Q_Learning"
    SARSA = "Sarsa"
    A_STAR = "A_star"
    DFS = "DFS"
    BFS = "BFS"
    IDS = "IDS"
    Row_GREEDY_A_STAR = "Row_Greedy_A_Star"

class PuzzleBridge(QObject):
    puzzle_list_changed = Signal()
    puzzle_size_changed = Signal()
    agent_training_progress_changed = Signal()
    invoke_start_btn_changed = Signal()
    invoke_generate_btn_changed = Signal()
    search_btn_is_enable_changed = Signal()
    search_status_is_pending_changed = Signal()
    search_status_is_done_changed = Signal()
    search_status_progress_is_busy_changed = Signal()
    
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
        self.threadpool   = QThreadPool()
        
        self.puzzle_list = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.puzzle_size = 3
        self.solution = None
        t = threading.Thread(target=self.main_func, daemon=True)
        t.start()
        self.environment = SlidingPuzzleEnv(self.puzzle_size)
        self.step_per_sec = 0
        self.agent_type = None
        # self.agent = None
        self.search_methods = Search(self.environment)
        self.train_progress = 0.0
        self.train_episode_num = 1000
        self.invoke_start = False
        self.invoke_generate = True
        self.search_btn_is_enable = False
        self.search_status_is_pending = True
        self.search_status_is_done = False
        self.search_status_progress_is_busy = False
        
    def set_search_button_enable(self, value):
        self.search_btn_is_enable = value
        self.search_btn_is_enable_changed.emit()
        
    def get_search_button_enable(self):
        return self.search_btn_is_enable
        
    def set_search_status_is_pending(self, value):
        self.search_status_is_pending = value
        self.search_status_is_pending_changed.emit()
        
    def get_search_status_is_pending(self):
        return self.search_status_is_pending
        
    def set_search_status_is_done(self, value):
        self.search_status_is_done = value
        self.search_status_is_done_changed.emit()
        
    def get_search_status_is_done(self):
        return self.search_status_is_done
        
    def set_search_status_progress_is_busy(self, value):
        self.search_status_progress_is_busy = value
        self.search_status_progress_is_busy_changed.emit()
        
    def get_search_status_progress_is_busy(self):
        return self.search_status_progress_is_busy

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
    pyside_search_btn_is_enable = Property(bool, get_search_button_enable, set_search_button_enable, notify=search_btn_is_enable_changed)
    pyside_search_status_is_pending = Property(bool, get_search_status_is_pending, set_search_status_is_pending,
                                               notify= search_status_is_pending_changed)
    pyside_search_status_is_done = Property(bool, get_search_status_is_done, set_search_status_is_done,
                                               notify= search_status_is_done_changed)
    pyside_search_status_progress_is_busy = Property(bool, get_search_status_progress_is_busy, set_search_status_progress_is_busy,
                                               notify= search_status_progress_is_busy_changed)
            
    def generate_new_puzzle(self):
        state = self.environment.generate_puzzle().flatten()
        self.set_puzzle_list(state.tolist()) # each puzzle that is setted here, is going to be the one that is solved.
        # self.set_invoke_start_btn(True)
        self.set_search_button_enable(True)
        self.set_invoke_start_btn(False) # disables the solving for the provious puzzle rather than the very new one
        self.set_search_status_is_done(False)
        self.set_search_status_is_pending(True)
        
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
        
    # thread function
    def handle_search_result(self, result):
        self.solution = result
        print("Got solution:", result)

    # thread function
    def handle_search_finished(self):
        print("Search thread finished.")
        self.set_search_status_progress_is_busy(False)
        self.set_invoke_start_btn(True)
        self.set_invoke_generate_btn(True) # after searchin is done, new puzzle is allowed to be maid
        self.set_search_status_is_done(True)

    # thread function
    def handle_search_error(self, error_tuple):
        exctype, value, tb = error_tuple
        print("Search error:", value)
        print(tb)
            
    def search(self):
        self.set_search_button_enable(False) # prevents multiple times of performing the search 
        # to do: implement a way that a puzzle could be solved with different methods...
        self.set_invoke_generate_btn(False) # while searching, no new puzzle must be maid
        self.set_search_status_is_pending(False)
        self.set_search_status_progress_is_busy(True)
    
        worker = Worker(self.search_puzzle)

        # connect signals
        worker.signals.result.connect(self.handle_search_result)
        worker.signals.finished.connect(self.handle_search_finished)
        worker.signals.error.connect(self.handle_search_error)

        # fire off the thread
        self.threadpool.start(worker)
        
        
    def search_puzzle(self, progress_callback=None):
        solution_steps = None
        if self.agent_type == AgentType.A_STAR:
            print("ASTAR is triggred")
            solution_steps = self.search_methods.solve_A_start()
                   
        elif self.agent_type == AgentType.BFS:
            print("BFS is triggred")
            solution_steps = self.search_methods.solve_bfs()
            
        elif self.agent_type == AgentType.DFS:
            print("DFS is triggred")
            solution_steps = self.search_methods.solve_dfs()
            
        elif self.agent_type == AgentType.IDS:
            print("IDS is triggred")
            solution_steps = self.search_methods.solve_ids()
            
        elif self.agent_type == AgentType.Row_GREEDY_A_STAR:
            print("row greedy is triggred")
            solution_steps = self.search_methods.solve_row_greedy()
            
        print(solution_steps)
        if solution_steps != None:
            solution = [state for state, _ in solution_steps]
        else:
            self.invoke_error()
        return solution
        
    
    def solve_puzzle(self): # to-do: integerate with pyside qthreads rather than python, as it cuases glitches
        self.set_search_button_enable(False)
        t = threading.Thread(target=self.render, args=(self.solution,), daemon=True)
        t.start()
        
    def render(self, solved_array):
        print(solved_array)
        self.set_invoke_start_btn(False)
        self.set_invoke_generate_btn(False)
        self.set_search_button_enable(False)
        for arr_ in solved_array:
            time.sleep(1 / self.step_per_sec)
            self.set_puzzle_list(arr_)
        self.set_invoke_generate_btn(True)
        print("finished rendering.")
    
    def main_func(self):
        # self.test_progress_bar()
        pass

    def test_progress_bar(self):
        for i in range(101):
            time.sleep(0.1)
            self.set_agent_training_progress(i/100)