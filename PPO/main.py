import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import gym_update
import argparse
from normalization import Normalization, RewardScaling
from buffer import ReplayBuffer
from agent import PPO_agent
import sys
sys.path.append('C:/Users/cvcla/my_py_projects/ModelFree/utilities')
from environment import GymEnv

'''SOME COMMENTS
- 1000 training steps 
- each training step starts a trajectory (with env.reset)
- each training step contains #episode_steps numb of transitions
- a trajectory ends when you reach done
- !! length of trajectory is very short (done is reached after 1/2 transitions)
- update happens after every 80 training steps
'''

def evaluate_policy(args, env, agent, state_norm):
    times = 3
    evaluate_reward = 0
    for _ in range(times):
        s = env.reset()
        if args.use_state_norm:
            s = state_norm(s, update=False)  # During the evaluating,update=False
        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(s)  # We use the deterministic policy during the evaluating
            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a

            s_, r, pat, pat_noA, reward_noA, state_noA, done = env.step(action)
            if args.use_state_norm:
                s_ = state_norm(s_, update=False)
            episode_reward += r
            s = s_
        evaluate_reward += episode_reward

    return evaluate_reward / times


def main(args, env_name, number, seed):
    env = GymEnv(args.env, args.max_episode_length)
    env_evaluate = GymEnv(args.env, args.max_episode_length)  # When evaluating the policy, we need to rebuild an environment
    # Set random seed
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env_evaluate.seed(seed)
    # env_evaluate.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    args.state_dim = env.observation_size
    args.action_dim = env.action_size
    lower_bound, upper_bound = env.bounds
    args.max_action = float(upper_bound)
    args.max_episode_steps = args.max_episode_length  # Maximum number of steps per episode
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_action={}".format(args.max_action))
    print("max_episode_steps={}".format(args.max_episode_steps))

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO_agent(args)
    writer = SummaryWriter(log_dir='runs/PPO_continuous/env_{}_{}_number_{}_seed_{}'.format(env_name, args.policy_dist, number, seed))

    state_norm = Normalization(shape=args.state_dim)  # Trick 2:state normalization
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
    

    while total_steps < args.max_train_steps:
        # start one episode, reset state, scale
        s = env.reset() # tensor
        if args.use_state_norm:
            s = state_norm(s)
        if args.use_reward_scaling:
            reward_scaling.reset() # 
        episode_steps = 0
        done = False


        while not done:
            # start one step in an episode
            episode_steps += 1
            a, a_logprob = agent.choose_action(s)  # Action and the corresponding log probability

            if args.policy_dist == "Beta":
                action = 2 * (a - 0.5) * args.max_action  # [0,1]->[-max,max]
            else:
                action = a
            s_, r, pat, pat_noA, r_noA, state_noA, done = env.step(action) # s is tensor
            
            if args.use_state_norm:
                s_ = state_norm(s_)
                
            if args.use_reward_norm:
                r = reward_norm(r)
                r_noA = reward_norm(r_noA)
            elif args.use_reward_scaling:
                r = reward_scaling(r)
                r_noA = reward_scaling(r_noA)
            #print("r", r.shape, total_steps, done)
            # When dead or win or reaching the max_episode_steps, done will be Ture, we need to distinguish them;
            # dw means dead or win,there is no next state s';
            # but when reaching the max_episode_steps,there is a next state s' actually.
            
            # set dw (dead/win) to T when done or num steps reaches max (150)
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            # collect objects in buffer
            replay_buffer.store(s, a, a_logprob, r, r_noA, s_, dw, done)
            s = s_
            total_steps += 1
            #print("tot steps", total_steps, "episode seps", episode_steps)
            # When the number of transitions in buffer reaches batch_size (80), then update
            # you have a buffer of size batch_size
            if replay_buffer.count == args.batch_size:
                loss_dict = agent.update(replay_buffer, total_steps, writer)

                
                print('[EPISODE] {} rew:{} rew_noA:{} a loss:{} c loss:{} a:{} done:{}'.format(total_steps, 
                                                                                            round(np.mean(replay_buffer.r), 3), 
                                                                                            round(np.mean(replay_buffer.r_noA), 3), 
                                                                                            loss_dict['actor'], 
                                                                                            loss_dict['critic'],
                                                                                            replay_buffer.a[-1],
                                                                                            replay_buffer.done[-1]
                                                                                            ))
                replay_buffer.count = 0

            # Evaluate the policy every 'evaluate_freq' steps
            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward = evaluate_policy(args, env_evaluate, agent, state_norm)
                evaluate_rewards.append(evaluate_reward)
                #print("evaluate_num:{} \t evaluate_reward:{} \t".format(evaluate_num, evaluate_reward))
                writer.add_scalar('step_rewards_{}'.format(env_name), evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                #if evaluate_num % args.save_freq == 0:
                #    np.save('./data_train/PPO_{}_env_{}_number_{}_seed_{}.npy'.format(args.policy_dist, env_name, number, seed), np.array(evaluate_rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=1000, help=" Maximum number of training steps") # int(3e6)
    parser.add_argument('--max_episode_length', default=150, type=int, help='') #500
    parser.add_argument('--env', default='update-v0', type=str, help='open-ai gym environment')
    parser.add_argument("--evaluate_freq", type=float, default=20, help="Evaluate the policy every 'evaluate_freq' steps") # 5e3
    parser.add_argument("--save_freq", type=int, default=20, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian")
    parser.add_argument("--batch_size", type=int, default=80, help="Batch size")#2048
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size") #64
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    env_name = ['update-v0']

    main(args, env_name, number=1, seed=10)