import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import os


class Diagnostics(object):

    def __init__(self, num_episodes, interval, save_path='', max_episode_length=None):
        self.num_episodes = num_episodes # validate_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results_r = np.array([]).reshape(num_episodes,0)
        self.results_rNA = np.array([]).reshape(num_episodes,0)

    def __call__(self, env, policy):

        self.is_training = False
        observation = None
        result_r = []
        result_rNA = []

        for episode in range(self.num_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward = 0.
            episode_reward_nA = 0.
                
            assert observation is not None

            # start episode
            done = False
            while not done:
                # basic operation, action ,reward, blablabla ...
                action = policy(observation)
                
                observation, reward, pat, pat_noA, reward_noA, state_noA, done = env.step(action)
                if episode_steps >= self.max_episode_length -1:
                    done = True
                # update
                episode_reward += np.mean(reward)
                episode_reward_nA += np.mean(reward_noA)
                episode_steps += 1

            #print('[Evaluate] #Episode{}: reward:{} reward nA:{}'.format(episode, episode_reward/episode_steps, episode_reward_nA/episode_steps))
            result_r.append(episode_reward)
            result_rNA.append(episode_reward_nA)

        result_r = np.array(result_r).reshape(-1,1)
        self.results_r = np.hstack([self.results_r, result_r])

        result_rNA = np.array(result_rNA).reshape(-1,1)
        self.results_rNA = np.hstack([self.results_rNA, result_rNA])        
        
        '''
        if save:
            self.save_results(self.results_r, '{}/validate_reward'.format(self.save_path))
            self.save_results(self.results_rNA, '{}/validate_reward'.format(self.save_path))
        '''
        
        return np.mean(result_r), np.mean(result_rNA)

    def save_results(self, res, fn):

        y = np.mean(res, axis=0)
        error=np.std(res, axis=0)
                    
        x = range(0, res.shape[1]*self.interval,self.interval)
        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        plt.xlabel('Timestep')
        plt.ylabel('Average Reward')
        ax.errorbar(x, y, yerr=error, fmt='-o')
        plt.savefig(fn+'.png')
        savemat(fn+'.mat', {'reward':res})


def get_output_folder(parent_dir, env_name):
    """Return save folder.
    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.
    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.
    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    os.makedirs(parent_dir, exist_ok=True)
    return parent_dir


