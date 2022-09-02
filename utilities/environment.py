import numpy as np
import torch
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import gym

def to_numpy(var):
       return var.detach().numpy()

class GymEnv:
    def __init__(self, env, max_episode_length):
        self._env = gym.make(env)
        self.max_episode_length = max_episode_length

    def reset(self):
        self.t = 0  # Reset internal timer
        state, s, i, o = self._env.reset()
        return torch.tensor(state, dtype=torch.float32) #.unsqueeze(dim=0)

    def step(self, action):
        if torch.is_tensor(action) == True:
            action = action.detach().numpy()
        state, reward, done, info = self._env.step(action) 
        reward = np.mean(reward)

        pat = info["patients"]
        pat_noA = info["patients_noA"]
        reward_noA = info["rew_noA"]
        state_noA = info["risk_noA"] 
        self.t += 1  # Increment internal timer
        observation = torch.tensor(state, dtype=torch.float32) #.unsqueeze(dim=0)
        return observation, reward, pat, pat_noA, np.mean(reward_noA), state_noA, done # 

    def close(self):
        self._env.close()

    @property
    def observation_size(self):
        return self._env.observation_space.shape[0] 

    @property
    def action_size(self):
        return self._env.action_space.shape[0]

    @property
    def bounds(self):
        return self._env.action_space.low[0], self._env.action_space.high[0]    

    # Sample an action randomly from a uniform distribution over all valid actions
    def sample_random_action(self):
        return torch.from_numpy(self._env.action_space.sample())
