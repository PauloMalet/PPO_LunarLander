import gymnasium as gym
import torch
from models import Memory


class Train:
    def __init__(
        self,
        env_name="LunarLander-v3",
        render=True,
        render_interval=500,
        log_interval=20,
        max_episodes=1500,
        max_timesteps=350,
        update_timestep=2000,
        action_std=0.5,
        K_epochs=40,
        eps_clip=0.2,
        gamma=0.99,
        lr=0.0003,
        betas=(0.9, 0.999),
        random_seed=None,
        decay_threshold=0,
        decay_speed=0.999,
        env_kwargs=None,
    ):
        self.env_name = env_name
        self.render = render
        self.render_interval = render_interval
        self.log_interval = log_interval
        self.time_step = 0
        self.env_kwargs = env_kwargs or {"continuous": True}

        # Hyperparameters
        self.max_episodes = max_episodes
        self.max_timesteps = max_timesteps
        self.update_timestep = update_timestep
        self.action_std = action_std
        self.K_epochs = K_epochs
        self.eps_clip = eps_clip
        self.gamma = gamma
        self.lr = lr
        self.betas = betas
        self.decay_threshold = decay_threshold
        self.decay_speed = decay_speed
        self.random_seed = random_seed

        # Environment setup
        self.env = gym.make(env_name, **self.env_kwargs)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        if random_seed:
            print("Random Seed: {}".format(random_seed))
            torch.manual_seed(random_seed)
            self.env.seed(random_seed)

        self.memory = Memory()
        from models import PPO

        self.ppo = PPO(state_dim, action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip)

    def play_episode(self, env):
        curr_reward = 0
        state, _ = env.reset()
        for t in range(self.max_timesteps):
            self.time_step += 1
            action = self.ppo.select_action(state, self.memory)
            state, reward, done, _, _ = env.step(action)

            self.memory.rewards.append(reward)
            self.memory.is_terminals.append(done)

            if self.time_step % self.update_timestep == 0:
                self.ppo.update(self.memory)
                self.memory.clear_memory()
                self.time_step = 0

            curr_reward += reward
            if done:
                break
        return curr_reward, t

    def train(self):
        running_reward = 0
        avg_length = 0
        history_avg_reward = []

        for i_episode in range(1, self.max_episodes + 1):
            if self.render and i_episode % self.render_interval == 0:
                render_env = gym.make(self.env_name, render_mode="human", **self.env_kwargs)
                r, length = self.play_episode(render_env)
                running_reward += r
                avg_length += length
                print(f"Render human : length: {length} \t reward: {r}")
            else:
                r, length = self.play_episode(self.env)
                running_reward += r
                avg_length += length

            if i_episode % 500 == 0:
                torch.save(self.ppo.policy.state_dict(), f"./PPO_continuous_{self.env_name}.pth")
                print(f"Saved at episode {i_episode}")

            if i_episode % self.log_interval == 0:
                avg_length = int(avg_length / self.log_interval)
                running_reward = int((running_reward / self.log_interval))

                print(f"Episode {i_episode} \t Avg length: {avg_length} \t Avg reward: {running_reward}")
                history_avg_reward.append((running_reward, avg_length))

                if running_reward > self.decay_threshold and self.action_std > 0.1:
                    self.action_std = self.action_std * self.decay_speed
                    self.ppo.set_action_std(self.action_std)
                    print("action_std : ", self.action_std)

                running_reward = 0
                avg_length = 0

        with open("history_avg_reward.txt", "w") as file:
            file.write("\n".join(str(reward) for reward in history_avg_reward))

        return history_avg_reward
