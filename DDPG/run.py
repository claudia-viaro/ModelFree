import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/ddpg_values')
import numpy as np
import matplotlib.pyplot as plt
import os
from copy import deepcopy
from torchvision.transforms.functional import to_tensor
import gym
import gym_update
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from environment import GymEnv
from agentDDPG import DDPG
from traintest_DDPG import train, test, Testing, get_output_folder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model-free RL')
    parser.add_argument('--mode', default='train', type=str, help='train/test')

    # about environment
    parser.add_argument('--env', default='update-v0', type=str, help='open-ai gym environment')
    parser.add_argument('--seed', default=-1, type=int, help='')

    # for the NN and optim
    parser.add_argument('--hidden1', default=400, type=int, help='hidden num of first fully connect layer')
    parser.add_argument('--hidden2', default=300, type=int, help='hidden num of second fully connect layer')
    parser.add_argument('--rate', default=0.002, type=float, help='critic learning rate')
    parser.add_argument('--prate', default=0.001, type=float, help='actor learning rate')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--init_w', default=0.003, type=float, help='') 
    parser.add_argument('--resume', default='output', type=str, help='Resuming model path for testing')

    # for the buffer
    parser.add_argument('--bsize', default=64, type=int, help='minibatch size') #64
    parser.add_argument('--bcapacity', default=60000, type=int, help='buffer capacity') # 6000000

    parser.add_argument('--window_length', default=1, type=int, help='')
    parser.add_argument('--tau', default=0.005, type=float, help='to update target networks')

    # for action noise process
    parser.add_argument('--noise_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--noise_sigma', default=0.2, type=float, help='noise sigma') 
    parser.add_argument('--noise_mu', default=0.0, type=float, help='noise mu') 

    # episodes/epochs/steps etc
    parser.add_argument('--episodes', default=20, type=int, help='how many episodes to perform during training') 
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory') #20
    parser.add_argument('--train_iter', default=200, type=int, help='how many steps in a single episode') #200000
    parser.add_argument('--max_episode_length', default=500, type=int, help='') #500
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--validate_episodes', default=50, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=5, type=int, help='how many steps to perform a validate experiment') # was 2000
    parser.add_argument('--output', default='output', type=str, help='')

    parser.add_argument('-f')
    args = parser.parse_args()
    
    args.output = get_output_folder(args.output, args.env)
    if args.resume == 'default':
        args.resume = 'output/{}-run0'.format(args.env)

    # init environment
    environ = GymEnv(args.env, args.max_episode_length)
    agent = DDPG(environ.observation_size, environ.action_size, args, environ)
    evaluate = Testing(args.validate_episodes, args.output, max_episode_length=args.max_episode_length)

    if args.mode == 'train':
        #train_version(args.episodes, args.train_iter, agent, environ, args.warmup, writer, args.max_episode_length)
        train(args.episodes, args.train_iter, agent, environ, args.output, args.warmup, writer, args.max_episode_length)

    elif args.mode == 'test':
        test(args.validate_episodes, agent, environ, evaluate, args.output, writer)

    else:
        raise RuntimeError('undefined mode {}'.format(args.mode))

'''
def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():

    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

# Create an example writer
example_writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb0",
                               extra="5_epochs")    
'''                               