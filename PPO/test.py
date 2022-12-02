import os
import glob
import time
from datetime import datetime

import torch
import numpy as np
import sys
from PPO import PPO
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from environment import Env

#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "Model_update"

    has_continuous_action_space = True  # continuous action space; else discrete

    max_ep_len = 100 #1000                   # max timesteps in one episode
    max_training_timesteps = 200 #int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = max_ep_len * 2   #10        # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2           # log avg reward in the interval (in num timesteps)
    save_model_freq = max_ep_len * 2 #int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    
    ################ PPO hyperparameters ################
    total_test_episodes = 10    # total num of testing episodes
    update_timestep = max_ep_len * 2 #4      # update policy every n timesteps
    K_epochs = 8 #80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################


    

    #####################################################

    env = Env()
    state_dim = env.observation_size
    action_dim = env.action_size

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes+1):
        ep_reward = 0
        s_alt, state = env.reset()
        time = 1
        for t in range(1, max_ep_len+1):
            time += 1
            action = ppo_agent.select_action(state)
            state, reward, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.step(action, state.detach().numpy())
            ep_reward += reward

            if time % 25 == 0:
                print('Intermediate:{}/{} Action:{} Reward: {}'.format(time, max_ep_len+1, action, round(ep_reward/time, 2)))


            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()

        test_running_reward +=  ep_reward / max_ep_len
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward/ (max_ep_len+1) , 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':

    test()
