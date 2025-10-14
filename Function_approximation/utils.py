import gymnasium as gym
import numpy as np
import random
from matplotlib import pyplot as plt
import time

env = gym.make("MountainCar-v0")
n_actions = env.action_space.n

def feature(state):
    position, velocity = state
    position = (position + 1.2) / 1.8
    velocity = (velocity + 0.07) / 0.14
    return np.array([1.0, position, velocity, position**2, velocity**2, position * velocity])

n_features = len(feature(env.reset()[0]))

def Q_value(state,W):
    phi = feature(state)
    return np.dot(W, phi)

def Q(state, action,W):
    return np.dot(W[action], feature(state))

def epsilon_greedy_action(state,W,epsilon = 0.1):
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q_value(state,W))
    
def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def plot_training_rewards(rewards, window=100):
    plt.figure(figsize=(8, 5))
    plt.title("Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")

    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        plt.plot(range(window, len(rewards)+1), moving_avg, label=f"Moving Avg ({window})", color='orange')
    plt.plot(rewards, color='steelblue', alpha=0.6, label="Raw Reward")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def test_agent(W, render=True, episodes=5, sleep_time=0.02):
    env_test = gym.make("MountainCar-v0", render_mode="human" if render else None)
    total_rewards = []

    for ep in range(episodes):
        state, _ = env_test.reset()
        done = False
        ep_reward = 0

        while not done:
            # 选择最优动作（不再探索）
            action = np.argmax(Q_value(state,W))
            next_state, reward, terminated, truncated, _ = env_test.step(action)
            done = terminated or truncated
            ep_reward += reward
            state = next_state

            if render:
                time.sleep(sleep_time)

        print(f"Test Episode {ep + 1}: total reward = {ep_reward}")
        total_rewards.append(ep_reward)

    env_test.close()
    print(f"\nAverage reward over {episodes} test episodes: {np.mean(total_rewards):.2f}")

