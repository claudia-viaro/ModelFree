import os
from datetime import datetime
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from arguments import get_config
import seaborn as sns
import sys
from logger import Logger
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from environment import Env
from plot_data import plot_datashift, plot_states, get_count, plot1

from PPO_agent import PPO


def plot3(rewards, results_dir, n_episodes):
        x_axis = range(1, n_episodes)
        
        plot3 = sns.lineplot(x=x_axis, y=rewards)
        plt_rew = plot3.get_figure()
        #Returns the :class:~matplotlib.figure.Figure instance the artist belongs to
        sample_file_name = "mean_reward_" 
        plt_rew.savefig(results_dir + sample_file_name, dpi=300, bbox_inches='tight')
        plt_rew.show()


################################### Training ###################################
def train():
    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    env_name = "Model_update"

    has_continuous_action_space = True  # continuous action space; else discrete
    
    multiple = 2
    max_ep_len = 100 #1000                   # max timesteps in one episode
    max_training_timesteps = 5 # max_ep_len * 10 * multiple  #int(3e6)   # break training loop if timeteps > max_training_timesteps

    print_freq = 200    ## max_ep_len * multiple      # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * multiple           # log avg reward in the interval (in num timesteps)
    save_model_freq = max_ep_len * multiple #int(1e5)          # save model frequency (in num timesteps)

    action_std = 0.6                    # starting std for action distribution (Multivariate Normal)
    action_std_decay_rate = 0.05        # linearly decay action_std (action_std = action_std - action_std_decay_rate)
    min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
    action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
    #####################################################

    ## Note : print/log frequencies should be > than max_ep_len

    ################ PPO hyperparameters ################
    update_timestep = 100 #  #max_ep_len * multiple   # update policy every n timesteps
    K_epochs = 80               # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)
    #####################################################

    print("training environment name : " + env_name)

    env = Env()
    state_dim = env.observation_size
    action_dim = env.action_size

    ###################### logging ######################
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--config_name", type=str, default="model_update")
    parser.add_argument("--strategy", type=str, default="information")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = get_config(args)
    logger = Logger(args.logdir, args.seed)
    logger.log("\n=== Loading experiment [device: {}] ===\n".format(DEVICE))
    logger.log(args)
    #### log files for multiple runs are NOT overwritten
    log_dir = "PPO_logs"
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    log_dir = log_dir + '/' + env_name + '/'
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)

    #### get number of log files in log directory
    run_num = 0
    current_num_files = next(os.walk(log_dir))[2]
    run_num = len(current_num_files)

    #### create new log file for each run
    log_f_name = log_dir + '/PPO_' + env_name + "_log_" + str(run_num) + ".csv"

    print("current logging run number for " + env_name + " : ", run_num)
    print("logging at : " + log_f_name)
    #####################################################

    ################### checkpointing ###################
    run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

    directory = "PPO_preTrained"
    if not os.path.exists(directory):
          os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
          os.makedirs(directory)


    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)
    #####################################################


    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", action_std)
        print("decay rate of std of action distribution : ", action_std_decay_rate)
        print("minimum std of action distribution : ", min_action_std)
        print("decay frequency of std of action distribution : " + str(action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    
    
    
    
    
    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')
    

    # directory to save plots
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, 'plots_ppo/')
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    list_reward = []

    # training loop
    while time_step <= max_training_timesteps: # 5000 #int(3e6)
        
        patients, state = env.reset()
        start_state = state
        print_state = np.float64(start_state.detach().cpu().numpy())
        
        Xa_initial = patients[:, 1]
        Xs_initial = patients[:, 0]
        print("{} New pop draw - initial risk: {:.3f}, min: {:.3f}, max: {:.3f}".format(time_step, np.mean(print_state), np.min(print_state), np.max(print_state)))
        print("-----Count risks levels: {}".format(get_count(start_state)))
        
        

        
        current_ep_reward = 0

        for t in range(1, max_ep_len+1): #10

            # select action with policy
            action, log_prob = ppo_agent.select_action(state)
            
            next_state, reward, pat, s_LogReg, r_LogReg, Xa_pre, Xa_post, outcome, done = env.step(action, state.detach().numpy())
            print_nextstate = np.float64(next_state.detach().cpu().numpy())
            if t == (max_ep_len):
                done = True
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            
            current_ep_reward += reward

            # update PPO agent
            if t % update_timestep == 0: #every 1000
                loss_actor_list, loss_actor = ppo_agent.update()
                # print average reward till last episode
                rew = current_ep_reward / t
                
                ppo_agent.save(checkpoint_path)
                
                print("-----Step: {} State: {} Action: {} Train loss: {} Done: {}".format(t, np.round(np.mean(print_nextstate), 3), action, torch.sum(loss_actor), done))
                

                print("-----Avg Reward : {} ".format(np.round(rew, 3)))  # print_avg_reward, rew are the same
                print("-----Count risks levels: {}".format(get_count(print_nextstate)))
                
                print_running_reward = 0
                print_running_episodes = 0
            
            if t == max_ep_len:
                print_nextstate = np.float64(next_state.detach().cpu().numpy())
                plot1(ax, fig, Xa_initial, Xa_post, print_state, print_nextstate, results_dir, sample_file_name)
            # if continuous action space; then decay action std of ouput action distribution
            if has_continuous_action_space and time_step % action_std_decay_freq == 0:
                ppo_agent.decay_action_std(action_std_decay_rate, min_action_std)

            '''
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
            '''
            
            '''
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0
            
            
            # save model weights
            if time_step % save_model_freq == 0:
                
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                
                print("model saved")
                
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")
            ''' 
            state = next_state 
            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward/max_ep_len
        list_reward.append(print_running_reward)
        print("-----E: {} Reward: {} ".format(time_step, np.round(print_running_reward, 3)))


        #print("print_running_reward", print_running_reward/i_episode, i_episode)
        print_running_episodes += 1

        log_running_reward += current_ep_reward

        


        log_running_episodes += 1
        time_step +=1
        i_episode += 1
    
    

    plot3(list_reward, results_dir, max_training_timesteps)
    log_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':

    train()
