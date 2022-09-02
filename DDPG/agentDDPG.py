import numpy as np
import torch
import torch.nn as nn
from torch import optim
from pathlib import Path
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from model import (Actor, Critic)
from buffer import SequentialMemory
from noise import OUActionNoise

def to_tensor(ndarray):
    return torch.from_numpy(np.array(ndarray).astype(np.float))

def to_numpy(var):
       return var.data.numpy()

criterion = nn.MSELoss()

class DDPG(object):
    def __init__(self, num_states, num_actions, args, env):
        
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = num_states
        self.nb_actions= num_actions
        
        # Create Actor and Critic Network
        nn_config = {
            'hidden1':args.hidden1, 
            'hidden2':args.hidden2, 
            'init_w':args.init_w
        }
        self.actor = Actor(num_states, num_actions, **nn_config)
        self.actor_target = Actor(num_states, num_actions, **nn_config)
        self.actor_optim  = optim.Adam(self.actor.parameters(), args.prate)
        
        self.critic = Critic(num_states, num_actions, **nn_config)
        self.critic_target = Critic(num_states, num_actions, **nn_config)
        self.critic_optim  = optim.Adam(self.critic.parameters(), args.rate)

        self.env = env
        
        #Create replay buffer
        self.ou_noise = OUActionNoise(args.noise_mu, args.noise_sigma, args.noise_theta)
        self.buffer = SequentialMemory(limit=args.bcapacity, window_length=args.window_length)
 

        # Hyper-parameters
        self.batch_size = args.bsize
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon

        # 
        self.epsilon = 1.0
        self.s_t = None # Most recent state
        self.a_t = None # Most recent action
        self.is_training = True

    def update(self, target, source):
      for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - self.discount) + param.data * self.discount)


    def update_policy(self):
        # Sample batch # self.buffer contains appended objects. now we separate them. 
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.buffer.sample_and_split(self.batch_size)       
        
        reward_batch = reward_batch.unsqueeze(1)   
        target_actions = self.actor_target(next_state_batch) # (batch size x 3) next state is # batch size x 2000
        next_q_values = self.critic_target([next_state_batch, target_actions]) # batch size x 1
        reward_batch1 = self.discount*torch.tensor(terminal_batch.astype(float)) *next_q_values
        target_q_batch = torch.add(reward_batch, reward_batch1)

        # Critic update
        self.critic.zero_grad()
        q_batch = self.critic([state_batch, action_batch])
        value_loss = criterion(q_batch, target_q_batch.float())
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()
        policy_actions = self.actor(state_batch)
        policy_loss = -self.critic([state_batch, policy_actions])

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        self.update(self.actor_target, self.actor)
        self.update(self.critic_target, self.critic)

        return policy_loss.item(), value_loss.item()

    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()


    def observe(self, r_t, s_t1, done):
        if self.is_training:
            self.buffer.append(self.s_t, self.a_t, r_t, done) 
            self.s_t = s_t1
    
    def random_action(self):
        action = self.env.sample_random_action()
        self.a_t = action
        return action
    
    def select_action(self, s_t, decay_epsilon=True):
        state = torch.tensor(np.array(s_t), dtype=torch.float32)
        action = self.actor_target(state) 
        #print("unclipped", action)
        noise = self.ou_noise()
        # Adding noise to action
        action = action.detach().numpy()
        
        action += self.is_training*max(self.epsilon, 0)* + noise
        
        lower_bound, upper_bound = self.env.bounds
        
        # We make sure action is within bounds
        action = np.clip(action, lower_bound, upper_bound)
        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        self.a_t = action
        return action

    
    def reset(self, obs):
        self.s_t = obs
        #self.random_process.reset_states()
     

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )


    def save_model(self,output, step):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
        
        #print(f"[INFO] Saving model @{'{}'.format(step)}") 
