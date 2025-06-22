import random
import logging
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gym import spaces
from puzzle_env import SlidingPuzzleEnv  


OPPOSITE = {0:1, 1:0, 2:3, 3:2}  # Up<->Down, Left<->Right

# 1) --- Logging setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)


# 2) --- Replay Buffer ---
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(
            lambda x: np.vstack(x), zip(*batch)
        )
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# 3) --- Q-Network ---
class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)


# 4) --- Agent ---
class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=1e-3,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=1000,
        device=None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # main & target networks
        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.learn_steps = 0
        self.target_update_freq = target_update_freq
        
        

    def select_action(self, state, epsilon, last_action=None):
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.policy_net(state_v).cpu().detach().numpy().flatten()

        if last_action is not None:
            q_values[OPPOSITE[last_action]] = -np.inf

        return int(np.argmax(q_values))

    # def select_action(self, state, epsilon):
    #     if random.random() < epsilon:
    #         return random.randrange(self.action_size)
    #     state_v = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    #     q_values = self.policy_net(state_v)
    #     return q_values.argmax(dim=1).item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_v = torch.FloatTensor(states).to(self.device)
        actions_v = torch.LongTensor(actions).to(self.device)
        rewards_v = torch.FloatTensor(rewards).to(self.device)
        next_states_v = torch.FloatTensor(next_states).to(self.device)
        dones_v = torch.BoolTensor(dones).to(self.device)

        # current Q-values
        q_vals = self.policy_net(states_v).gather(1, actions_v)

        # target Q-values
        with torch.no_grad():
            next_q_vals = self.target_net(next_states_v).max(dim=1, keepdim=True)[0]
            target_q = rewards_v + self.gamma * next_q_vals * (~dones_v)

        loss = nn.MSELoss()(q_vals, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target network
        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()


# 5) --- Training Loop & Usage ---
def train(
    env: SlidingPuzzleEnv,
    agent: DQNAgent,
    num_episodes=500,
    max_steps=200,
    eps_start=1.0,
    eps_end=0.01,
    eps_decay=0.995
):
    epsilon = eps_start
    for ep in range(1, num_episodes + 1):
        start_flat = env.generate_puzzle()                         # sets puzzle_to_solve
        state = start_flat.astype(np.float32) / (env.size**2 - 1)
        total_reward = 0.0
        total_loss = 0.0

        for t in range(max_steps):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            next_state = next_state.astype(np.float32) / (env.size**2 - 1)

            agent.buffer.add(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            total_reward += reward
            total_loss += loss

            if done:
                break

        epsilon = max(eps_end, epsilon * eps_decay)

        logging.info(
            f"Episode {ep} | "
            f"Reward: {float(total_reward):7.1f} | "
            f"Steps: {t+1:3d} | "
            f"Epsilon: {float(epsilon):.3f} | "
            f"AvgLoss: {float(total_loss / (t + 1)) if t else float(total_loss):.4f}"
        )

    # save the final model
    torch.save(agent.policy_net.state_dict(), "dqn_puzzle_final_2.pth")
    logging.info("Training complete, model saved as dqn_puzzle_final.pth")


if __name__ == "__main__":
    # create env & agent
    env = SlidingPuzzleEnv(size=3)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(
        state_size=state_dim,
        action_size=action_dim,
        lr=1e-3,
        gamma=0.99,
        buffer_size=50000,
        batch_size=64,
        target_update_freq=500,
    )

    train(env, agent, num_episodes=1000, max_steps=150, eps_decay=0.950)
    # train(env, agent, num_episodes=500, max_steps=150, eps_decay=0.950)
