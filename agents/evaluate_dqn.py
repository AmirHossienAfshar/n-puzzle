# evaluate_dqn.py

import time
import torch
import numpy as np

from search import SlidingPuzzleEnv
from agents.dqn_agent import DQN

def print_board(state_flat, size):
    board = state_flat.reshape(size, size)
    for row in board:
        print(" ".join(f"{int(x):2d}" for x in row))
    print()

def run_episode(env, policy_net, max_steps=100, delay=0.5):
    # --- Use your generate_puzzle() API ---
    start_flat = env.generate_puzzle()                     # 1) shuffle & ensure solvable
    initial = start_flat.astype(np.float32) / (env.size**2 - 1)
    print("Puzzle to solve (env.puzzle_to_solve):")
    print_board(env.puzzle_to_solve, env.size)             # show exactly what was generated
    print("Start solving:")
    print_board(initial, env.size)

    state = initial
    for step in range(1, max_steps+1):
        state_v = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_vals = policy_net(state_v)
        action = q_vals.argmax(dim=1).item()

        next_flat, reward, done, _ = env.step(action)
        next_state = next_flat.astype(np.float32) / (env.size**2 - 1)

        # cast reward to float for formatting
        print(f"Step {step:3d}, action={['↑','↓','←','→'][action]}, reward={float(reward):5.1f}")
        print_board(next_flat, env.size)
        time.sleep(delay)

        state = next_state
        if done:
            print(f"Solved in {step} steps!\n")
            return

    print("Did not solve within step limit.\n")


if __name__ == "__main__":
    env = SlidingPuzzleEnv(size=3)

    # load your trained network
    policy_net = DQN(state_size=env.size**2, action_size=env.action_space.n)
    policy_net.load_state_dict(torch.load("dqn_puzzle_final.pth", map_location="cpu"))
    policy_net.eval()

    # watch a few puzzles
    for epi in range(1, 6):
        print(f"=== Episode {epi} ===")
        run_episode(env, policy_net, max_steps=100, delay=0.3)
