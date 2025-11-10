%%writefile train.py
# train.py
import os
import numpy as np
import matplotlib.pyplot as plt
from env import RuralEnv
from rl_agent import QLearningAgent

def train(episodes=5000, report_every=500):
    env = RuralEnv(grid=(4,4), n_hospitals=6, n_ambulances=3)
    n_states = env.n_states
    n_actions = env.action_space

    agent = QLearningAgent(n_states=n_states, n_actions=n_actions,
                           alpha=0.1, gamma=0.95, epsilon=0.5, decay=0.9992, min_epsilon=0.01)

    rewards = []
    avg_rewards = []
    for ep in range(1, episodes+1):
        state = env.reset()
        # single-step episode (one patient)
        action = agent.select_action(state)
        next_s, reward, done, info = env.step(action)
        agent.learn(state, action, reward, next_s)
        rewards.append(reward)

        if ep % report_every == 0:
            avg = np.mean(rewards[-report_every:])
            print(f"Episode {ep}: avg_reward_last_{report_every} = {avg:.2f}, epsilon={agent.epsilon:.3f}")
            avg_rewards.append((ep, avg))

    # save Q-table and rewards
    os.makedirs("results", exist_ok=True)
    np.save("results/q_table.npy", agent.Q)
    np.save("results/rewards.npy", np.array(rewards))

    # plot reward curve (smoothed)
    plt.figure(figsize=(8,4))
    window = max(1, episodes//100)
    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(smoothed)
    plt.title("Smoothed reward per episode")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.tight_layout()
    plt.savefig("results/reward_curve.png")
    print("Training complete. Q-table and reward curve saved in results/")

if __name__ == "__main__":
    train(episodes=5000, report_every=500)