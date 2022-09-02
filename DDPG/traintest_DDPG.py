from copy import deepcopy
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import os


import os
import torch
from torch.autograd import Variable


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

'''
def get_output_folder(parent_dir):
    # set up directories to save logs
    if not os.path.exists('./parent_dir/'):
        os.mkdir('./parent_dir/')
    return    
'''

def train(episodes, iterations, agent, env, output, warmup, writer, max_episode_length):

    

    results_train = {"policy_loss": [], 
               "value_loss": [],
               "reward": [],
               "reward_noA": []}

    #episode = 0
    for episode in tqdm(range(episodes)):
        
        observation = None

        # mean losses in 1 episode, over n iterations
        train_policy_l, train_value_l, reward, reward_noA, observation, action, done, step = train_step(observation, iterations, agent, env, output, warmup, max_episode_length)

        if step > warmup: # end of episode
            
            print('start updating')
            
            print('[EPISODE] {} rew:{} rew_noA:{} lossL:{} lossV:{} a:{} done:{} steps:{}'.format(episode, 
                                                                                            round(reward, 3), 
                                                                                            round(reward_noA, 3), 
                                                                                            round(train_policy_l, 3), 
                                                                                            round(train_value_l, 3),
                                                                                            action,
                                                                                            done,
                                                                                            step))
                                                                                           
            # append in buffer the last state of the trajectory of each episode and get the action (which you wouldn't otherwise have taken cause you stopped)
            agent.buffer.append(
                observation,
                agent.select_action(observation),
                np.repeat(0., 2000), False
            )

            # reset
            observation = None
            #episode += 1  # we go to the next episode only when we're done (we pretend we're done if we have had 500 steps)

            # Update results dictionary
            results_train["policy_loss"].append(train_policy_l)
            results_train["value_loss"].append(train_value_l)
            results_train["reward"].append(reward)
            results_train["reward_noA"].append(reward_noA)
            action = np.array(action)
            # Add loss results to SummaryWriter
            writer.add_scalars(main_tag="Policy Loss", 
                                tag_scalar_dict={"policy_loss": train_policy_l},
                                global_step=episode)
            writer.add_scalars(main_tag="Value Loss", 
                                tag_scalar_dict={"value_loss": train_value_l},
                                global_step=episode)
            writer.add_scalars(main_tag="Mean episode reward", 
                                tag_scalar_dict={"reward": reward},
                                global_step=episode)
            writer.add_scalars(main_tag="Mean episode reward without action", 
                                tag_scalar_dict={"reward_noA": reward_noA},
                                global_step=episode)
            writer.add_scalars(f'Actions taken', {
                                'action[0]': action[0],
                                'action[1]': action[1],
                                'action[2]': action[2]}, episode)                                                                                                                                                                                                              
                                                                                
        else: 
            print('[EPISODE]{} info:{} steps:{}'.format(episode, "done in warmup phase", step))                    
        
        
    
    # Close the writer
    writer.close()



    
def train_step(observation, iterations, agent, env, output, warmup, max_episode_length):

    '''
    Trains a PyTorch model for a single episode.
    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).
    '''
    train_policy_l, train_value_l = 0.0, 0.0
    step = 1
    episode_reward, episode_reward_nA = 0.0, 0.0
    actions_trajectory = []

    while step < (iterations+1): # 1000 steps 
        # reset if it is the start of episode
        if observation is None:
            observation = deepcopy(env.reset())
            agent.reset(observation)
        
        # agent pick action 
        if step <= warmup: 
            action = agent.random_action()
        else:
            action = agent.select_action(observation)
        
        observation2, reward, pat, pat_noA, reward_noA, state_noA, done = env.step(action)
        observation2 = deepcopy(observation2)
        
        # after some time (150) set to true
        if step >= max_episode_length + 1:
            done = True        
        
        # agent observe (incrementally appends r_t, s_t, done)
        agent.observe(reward, observation2, done)
        
        # after a number of step we update the policy and keep track of loss values
        if step > warmup :
            p_loss_item, v_loss_item = agent.update_policy()
            # sum the loss items across all steps (in excess of warmup) in a single episode
            train_policy_l += p_loss_item
            train_value_l += v_loss_item

        if step > warmup and step % 20 == 0: # every 20
            agent.save_model(output, step)

        '''
        # print something during the episode (every 10 steps)
        if (step+10) % int(iterations/20) == 0: # every 50
            print('[ITER SO FAR] Step {} reward:{} reward noA:{} a:{} done:{}'.format(step, round(np.mean(reward),3), 
                                                                    round(np.mean(reward_noA), 3), action, done))   
        '''
        '''
        if evaluate is not None and validate_steps > 0 and step % validate_steps == 0:

            policy = lambda x: agent.select_action(x, decay_epsilon=False)
            validate_reward, validate_reward_nA = evaluate(env, policy)
            #print('[Evaluate] Step {} mean_reward:{} mean_reward:{}'.format(step, validate_reward, validate_reward_nA))
        '''
            
        
        actions_trajectory.append(action)
        episode_reward += np.mean(reward)
        episode_reward_nA += np.mean(reward_noA)
        observation = deepcopy(observation2) 
        
        step += 1 

    train_policy_l = train_policy_l/step # mean loss over one episode
    train_value_l = train_value_l/step
    episode_reward = episode_reward/step
    episode_reward_nA = episode_reward_nA/step
    a = actions_trajectory[-1]

    return train_policy_l, train_value_l, episode_reward, episode_reward_nA, observation, a, done, step-1

class Testing(object):

    def __init__(self, test_episodes, save_path='', max_episode_length=None):
        self.test_episodes = test_episodes # validate_episodes
        self.max_episode_length = max_episode_length
        self.save_path = save_path
        

    def __call__(self, env, policy, save=False):

        self.is_training = False
        self.results_test = {"reward": [],
               "reward_noA": []}
        observation = None        

        for episode in range(self.test_episodes):

            # reset at the start of episode
            observation = env.reset()
            episode_steps = 0
            episode_reward, episode_reward_nA = 0.0, 0.0
                
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

            # Adjust metrics to get average loss and accuracy per batch 
            episode_reward = episode_reward/episode_steps
            episode_reward_nA = episode_reward_nA/episode_steps
        
        self.results_test["reward"].append(episode_reward)
        self.results_test["reward_noA"].append(episode_reward_nA)
        
        return self.results_test, done, episode_steps

def test(episodes, agent, env, evaluate, model_path, writer):
    agent.load_weights(model_path)
    agent.is_training = False
    agent.eval()
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    

    for episode in tqdm(range(episodes)):
        # reset at the start of episode
        
        
        results, done, step = evaluate(env, policy)
        print('[EPISODE] {} rew:{} rew_noA:{} done:{} steps:{}'.format(episode, np.array(results["reward"]), np.array(results["reward_noA"]), done, step))

        if writer:
            # Add loss results to SummaryWriter
            writer.add_scalars(main_tag="Mean episode reward/Train", 
                                tag_scalar_dict={"reward": np.array(results["reward"])},
                                global_step=episode)
            writer.add_scalars(main_tag="Mean episode reward without action/Train", 
                                tag_scalar_dict={"reward_noA": np.array(results["reward_noA"])},
                                global_step=episode)
        
            # Close the writer
            writer.close()
        else:
            pass    

def test_step(observation, agent, model_path, env, test_step, max_episode_length):
    """
    Test for a single epoch.
    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.
    """
    agent.is_training = False
    agent.load_weights(model_path)
    
    agent.eval()

    test_step = 1
    episode_reward, episode_reward_nA = 0.0, 0.0
    policy = lambda x: agent.select_action(x, decay_epsilon=False)
    done = False
    observation = None

    while not done:
        observation = env.reset()
        assert observation is not None
        # basic operation, action ,reward, blablabla ...
        action = policy(observation)
        
        observation, reward, pat, pat_noA, reward_noA, state_noA, done = env.step(action)
        if test_step >= max_episode_length -1:
            done = True
        # update

        
        episode_reward += np.mean(reward)
        episode_reward_nA += np.mean(reward_noA)
        test_step += 1

    # Adjust metrics to get average loss and accuracy per batch 
    episode_reward = episode_reward/test_step
    episode_reward_nA = episode_reward_nA/test_step

    return episode_reward, episode_reward_nA, observation, done, test_step-1
