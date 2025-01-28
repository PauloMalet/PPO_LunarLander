import gymnasium as gym
import torch
import matplotlib.pyplot as plt
import numpy as np
from models import Memory, PPO


class Test:
    def __init__(
        self,
        model_path="trained_model.pth",
        env_name="LunarLander-v3",
        max_timesteps=350,
        action_std=0.01,
        K_epochs=40,
        eps_clip=0.2,
        gamma=0.99,
        lr=0.0003,
        betas=(0.9, 0.999),
        n_simulations=100,
        env_kwargs=None,
    ):
        self.model_path = model_path
        self.n_simulations = n_simulations
        self.env_name = env_name
        self.max_timesteps = max_timesteps
        self.action_std = action_std
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lr = lr
        self.betas = betas
        self.env_kwargs = env_kwargs or {"continuous": True}

    def run_simulation(self, env, ppo, memory):
        state, _ = env.reset()
        total_reward = 0
        timesteps = 0

        for t in range(self.max_timesteps):
            action = ppo.select_action(state, memory)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            timesteps += 1
            if done:
                break

        return total_reward, timesteps

    def plot_results(self, rewards, timesteps):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        ax1.hist(rewards, bins=20, edgecolor="black")
        ax1.set_title("Distribution of Rewards")
        ax1.set_xlabel("Reward")
        ax1.set_ylabel("Frequency")
        ax1.set_xlim(-50, max(rewards))
        ax1.axvline(
            np.mean(rewards), color="r", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(rewards):.1f}"
        )
        ax1.legend()

        ax2.hist(timesteps, bins=20, edgecolor="black")
        ax2.set_title("Distribution of Episode Lengths")
        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Frequency")
        ax2.axvline(
            np.mean(timesteps), color="r", linestyle="dashed", linewidth=1, label=f"Mean: {np.mean(timesteps):.1f}"
        )
        ax2.legend()

        plt.tight_layout()
        plt.savefig("test_results.png")
        plt.close()

    def load_and_test_model(self):
        env_no_render = gym.make(self.env_name, **self.env_kwargs)
        render_kwargs = {**self.env_kwargs, "render_mode": "human"}
        env_render = gym.make(self.env_name, **render_kwargs)

        state_dim = env_no_render.observation_space.shape[0]
        action_dim = env_no_render.action_space.shape[0]

        ppo = PPO(
            state_dim, action_dim, self.action_std, self.lr, self.betas, self.gamma, self.K_epochs, self.eps_clip
        )
        state_dict = torch.load(self.model_path, weights_only=True)
        ppo.policy_old.load_state_dict(state_dict)
        ppo.policy_old.eval()

        memory = Memory()
        rewards = []
        timesteps = []
        print(f"Running {self.n_simulations} simulations...")

        for i in range(self.n_simulations):
            if i % 10 == 0:
                print(f"Simulation {i}/{self.n_simulations}")

            if i < self.n_simulations - 1:
                reward, t = self.run_simulation(env_no_render, ppo, memory)
            else:
                reward, t = self.run_simulation(env_render, ppo, memory)

            rewards.append(reward)
            timesteps.append(t)
            memory.clear_memory()

        env_no_render.close()
        env_render.close()

        avg_reward = np.mean(rewards)
        std_reward = np.std(rewards)
        avg_timesteps = np.mean(timesteps)
        std_timesteps = np.std(timesteps)

        # Plot results
        self.plot_results(rewards, timesteps)

        # Print summary statistics
        print("\nTest Results Summary:")
        print(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"Average Episode Length: {avg_timesteps:.2f} ± {std_timesteps:.2f}")
        print("Results visualization saved as 'test_results.png'")

        return rewards, timesteps
