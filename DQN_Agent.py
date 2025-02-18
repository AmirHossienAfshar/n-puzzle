import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# Define the neural network model.
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output raw Q-values (one per action)

# DQNAgent class
class DQNAgent:
    def __init__(self, env, learning_rate=1e-3, discount_factor=0.99,
                 exploration_rate=1.0, epsilon_decay=0.995, min_epsilon=0.01,
                 batch_size=64, replay_buffer_size=10000, target_update_freq=10):
        """
        Initialize the DQN agent.
        
        Parameters:
          env: The environment (must be Gym-compatible).
          learning_rate: Adam optimizer learning rate.
          discount_factor: Gamma in Q-learning.
          exploration_rate: Initial epsilon for epsilon-greedy.
          epsilon_decay: Multiplicative decay factor for epsilon.
          min_epsilon: Minimum value for epsilon.
          batch_size: Size of minibatch samples.
          replay_buffer_size: Maximum number of experiences to store.
          target_update_freq: How many episodes between target network updates.
        """
        self.env = env
        self.input_dim = env.observation_space.shape[0]  # e.g. 9 for a 3x3 puzzle
        self.output_dim = env.action_space.n             # Should be 4
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.target_update_freq = target_update_freq
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy network and target network
        self.policy_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps_done = 0

    def select_action(self, state):
        """
        Epsilon-greedy action selection.
        state is expected to be a flattened NumPy array.
        Returns an action (integer).
        """
        if random.random() < self.exploration_rate:
            return random.randrange(self.output_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(torch.argmax(q_values).item())

    def store_experience(self, state, action, reward, next_state, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self):
        """Sample a minibatch from the replay buffer and update the policy network."""
        if len(self.replay_buffer) < self.batch_size:
            return  # Not enough samples to train
        minibatch = random.sample(self.replay_buffer, self.batch_size)
        
        # Convert lists of experiences to numpy arrays first.
        states_np = np.array([s for s, a, r, ns, d in minibatch], dtype=np.float32)
        actions_np = np.array([a for s, a, r, ns, d in minibatch], dtype=np.int64)
        # rewards_np = np.array([float(r) for s, a, r, ns, d in minibatch], dtype=np.float32)
        rewards_np = np.array(
            [r.item() if hasattr(r, 'item') else float(r) for s, a, r, ns, d in minibatch],
            dtype=np.float32
        )
        next_states_np = np.array([ns for s, a, r, ns, d in minibatch], dtype=np.float32)
        dones_np = np.array([float(d) for s, a, r, ns, d in minibatch], dtype=np.float32)
        
        states = torch.from_numpy(states_np).to(self.device)
        actions = torch.from_numpy(actions_np).unsqueeze(1).to(self.device)
        rewards = torch.from_numpy(rewards_np).to(self.device)
        next_states = torch.from_numpy(next_states_np).to(self.device)
        dones = torch.from_numpy(dones_np).to(self.device)
        
        # Current Q-values from the policy network for the actions taken.
        q_values = self.policy_net(states).gather(1, actions).squeeze(1)
        
        # Next Q-values from the target network (max over actions).
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
        
        expected_q_values = rewards + self.discount_factor * next_q_values * (1 - dones)
        
        loss = nn.MSELoss()(q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, episodes):
        """
        Train the DQN agent over a number of episodes.
        Logs progress every 100 episodes.
        """
        for episode in range(episodes):
            state = self.env.reset()  # state is a flattened NumPy array
            done = False
            total_reward = 0
            step_count = 0
            
            while not done and step_count < 1000:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store_experience(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                step_count += 1
                self.train_step()
                self.steps_done += 1

            # Decay epsilon after each episode
            self.exploration_rate = max(self.min_epsilon, self.exploration_rate * self.epsilon_decay)
            # Update target network periodically
            if (episode + 1) % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {self.exploration_rate}")
        print("Training complete!")

    def solve(self, puzzle_to_solve):
        """
        Solve the puzzle using the trained policy.
        'puzzle_to_solve' is expected to be a flat array.
        Returns a list of states (each as a flat list) representing the solution path.
        """
        # Set the environment state to the puzzle to solve.
        self.env.state = np.array(puzzle_to_solve).reshape(self.env.size, self.env.size)
        solution_states = [self.env.state.flatten().tolist()]  # record initial state
        done = np.array_equal(self.env.state, self.env.goal_state)
        steps = 0
        state = self.env.state.flatten()
        while not done:
        # while not done and steps < 10000:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            solution_states.append(next_state.tolist())
            state = next_state
            steps += 1
            if done:
                print(f"Solved in {steps} moves!")
                break
        return solution_states


if __name__ == "__main__":
    from puzzle_env import SlidingPuzzleEnv  # Ensure you have your environment module.
        
    env = SlidingPuzzleEnv(size=4)
    print("Goal state is:")
    print(env.goal_state)
    
    print("Generating a new puzzle...")
    initial = env.generate_puzzle()  # This sets env.puzzle_to_solve.
    print("Initial puzzle (flattened):", initial)
    
    # Create a DQN agent.
    dqn_agent = DQNAgent(
        env=env,
        learning_rate=1e-3,
        discount_factor=0.99,
        exploration_rate=1.0,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        batch_size=64,
        replay_buffer_size=10000,
        target_update_freq=10
    )
    
    print("Training the DQN agent...")
    dqn_agent.train(episodes=300)
    
    print("Solving the puzzle...")
    solution = dqn_agent.solve(env.puzzle_to_solve)
    # for state in solution:
    #     print(state)
