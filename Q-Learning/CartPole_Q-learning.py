###################################################################
# This code is one of many codes which aim to provide a simple tool
# which can be helpful to test and visualize RL algorithms
###################################################################
# This specific code is for testing n-steps TD Q learing algorithm 
# in the gym CartPole-v0 environment.
# It is possible to change and tune # multiple parameters as training
# parametars (epsilon, learning rate, decay) and the initial Q-values 
# for each state_action pair. by doing so it is possible to understand 
# how each parameter can affect the training process

import gym
import numpy as np
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='Python Q-learning example')

parser.add_argument('--q_table_file', type=str, default='q_table_mountain.csv', metavar='The Q table file path',
                    help='The name of csv file that contains the q_table (default: q_table_mountain.csv)')
parser.add_argument('-u','--use_existing_q_table', default=False, action='store_true',
                    help='True for using existing Q-table, Flase for starting from the Beginning')
parser.add_argument('-s','--save_q_table', default=False, action='store_true',
                    help='True for saving the trained Q-table (default: False) (This will replace the file that have the name used as argument for --q_table_file)')
parser.add_argument('-g','--gamma', type=float, default=0.99, metavar='discount',
                    help='discount factor (default: 0.99)')
parser.add_argument('-a', '--alpha', type=float, default=0.2, metavar='learning rate',
                    help='learning rate value (default: 0.2)')
parser.add_argument('-d', '--decay', type=float, default=0.95, metavar='Decay',
                    help='Decay factor (default: 0.95)')
parser.add_argument('-e','--epsilon', type=float, default=0.99, metavar='Epsilon',
                    help='Starting epsilon value (default: 0.99)')
parser.add_argument('--episodes', type=int, default=50000,
                    help='Number of training episodes (default:5000)')
parser.add_argument('--max_t', type=int, default=10000,
                    help='Max steps in each episod (default:10000)')
parser.add_argument('--render_each', type=int, default=100,
                    help='Number of episodes before rendering (default:100 "after each 100 episodes a full episod will be shown if the argument --render_while_training is set to True")')
parser.add_argument('--render_while_training', default=False, action='store_true',
                    help='True for rendering first episod of each 100 episod while training (default: False)')
parser.add_argument('-i', '--initial_q_value', type=float, default=0.0, metavar='Initial Q-values',
                    help='The initial Q values for each state_action pair (default: 0.0)')
parser.add_argument('-n', '--n_steps', type=int, default=10, metavar='n steps TD',
                    help='The number of steps to use to calculat q_value (defaule: 10, use -1 for MC learning)')
parser.add_argument('--done_reward', type=int, default=0, metavar='Final Reward',
                    help='The reward that the agent gets when finishing an episod (defaule: 0)')

args = parser.parse_args()

# Loading chosen arguments
data_file               = args.q_table_file
use_existing_q_table    = args.use_existing_q_table
gamma                   = args.gamma
alpha                   = args.alpha
decay                   = args.decay
epsilon                 = args.epsilon
episodes                 = args.episodes
render_while_training   = args.render_while_training
max_t                   = args.max_t
render_each             = args.render_each
save_q_table            = args.save_q_table
initial_q_value         = args.initial_q_value
n_steps                 = args.n_steps
done_reward             = args.done_reward


#################################################################
# The following part is used to visualize the state distributaion
# which can help to define the digitizing for each state
#################################################################
# def Visualizing_observation_distribution(observation):
#     obs = np.array(observation)
#
#     f, axis = plt.subplots(2, 2, figsize=(8, 4), sharey=True)
#     for i, title in enumerate(['Position', 'Velocity', 'Pole Angle', 'Pole Velocity']):
#         ax = axis[i//2, i%2]
#         l = list(zip(*obs))[i]
#         ax.hist(l, bins=100)
#         ax.set_title(title)
#         xmin, xmax = ax.get_xlim()
#         ax.set_xlim(min(xmin, -xmax), max(xmax, -xmin))
#         ax.grid()
#     f.show()
#
# seen_obs = []
# for _ in range(10000):
#     seen_obs.append(list(env.reset()))
#     done = False
#     while not done:
#         s, r, done, _ = env.step(env.action_space.sample())
#         seen_obs.append(list(s))
#        
# Visualizing_observation_distribution(seen_obs)
##################################################################

from gym.core import ObservationWrapper

class Observation_Wrapper (ObservationWrapper):
    def observation(self, state):
        # the number of digits was decided accourding to the distribution of tje 
        # values for each state 
        state_ = [round(state[0], 2), round(state[1], 1), round(state[2], 2), round(state[3], 1)]
        state = state_
        return tuple(state)

# Function for defining the environment
def make_env(name="CartPole-v0", wrapper=Observation_Wrapper):
    env=wrapper(gym.make(name).env)
    return env

# env = Observation_Wrapper(gym.make("CartPole-v0").env)
env = make_env()

s = env.reset()
n_actions = env.action_space.n
print("Action space: {}".format(env.action_space))
print("Number of actions: {}".format(n_actions))
print("State space: {}".format(env.observation_space))
print("State: {}".format(s))

writer = SummaryWriter(comment=f"CartPole_Q_L")

# Define The agent
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):
        """
        Q-Learning Agent
        based on practice RL course on coursera
        Class variables
        - self.epsilon
        - self.alpha
        - self.discount

        Class functions
        - self.get_legal_actions(state)
            { Returns a list of all actions possible in this state}
        - self.get_qvalue(state, action)
            { Returns the Q value of the pair (state, action)}
        - self.set_qvalue(state, action, value)
            { Set the value of the pair (state, action)}
        - self.get_value(state)
            { Returns the value of the (state) which is the max Q_value of
            all possible actions in this state}
        - self.update(state, action, reward, next_state):
            { Calculate and update the Q value of the pair (state, action)}
        - self.get_best_action(state)
            { Returns the best action possible in this state according to
            the current Q values}
        - self.get_action(state)
            { Returns an action which is the best action with possibility of
             (1-epsilon) and returns a random action with a posibility of
             epsilon}
        """
        self.get_legal_actions = get_legal_actions
        self._qvalues = defaultdict(lambda: defaultdict(lambda: initial_q_value))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        """
        Returns the Q value  of the pair (state, action)}
        """
        return self._qvalues[state][action]

    def set_qvalue(self, state, action, value):
        """
        Set the Q value  of the pair (state, action)}
        """
        self._qvalues[state][action]=value

    def get_value(self, state):
        """
        Returns the value of the (state) which is the max Q_value of
        all possible actions in this state
        """
        possible_actions = self.get_legal_actions(state)

        # if there is no legal actions we return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value = np.NINF
        for action in possible_actions:
            _v = self.get_qvalue(state, action)
            if _v > value:
                value = _v
        
        return value

    def update(self, state, action, reward, next_state):
        """
        Calculate and update the Q value of the pair (state, action)
        """
        g = self.discount
        a = self.alpha
        
        _q = (1-a)*self.get_qvalue(state, action) + \
            a*(reward + g*self.get_value(next_state))
             
        self.set_qvalue(state, action, _q)

    def update_multiple(self, state_action, rewards, next_state):
        """
        Calculate and update the Q value of the pair (state, action) which
        happened n steps before the current state
        """
        g = self.discount
        a = self.alpha
        l = len(state_action)
        reward = rewards[-1]
        if next_state == None:
            _q = reward
        else:
            _q = reward + g*self.get_value(next_state)

        # print(f"rewards: {rewards}")
        # print(f"next_state value: {self.get_value(next_state)}")
        # print(f"_q: {_q}")

        state , action = state_action[0]
        # calculate changes in value for n steps back
        for i in range(l-2, -1, -1):
            _q = rewards[i] + g*_q
        
        new_q = (1-a)*self.get_qvalue(state, action)+a*_q

        # print(f"new_q: {new_q}")
        # print(f"q_value before updating :{self.get_qvalue(state, action)}")
        self.set_qvalue(state, action, new_q)

        # print(f"q_value after updating :{self.get_qvalue(state, action)}")

    def get_best_action(self, state):
        """
        Returns the best action possible in this state according to
        the current Q values
        """
        possible_actions = self.get_legal_actions(state)

        # if there is no legal actions we return None
        if len(possible_actions) == 0:
            return None

        best_value = np.NINF
        best_action = None
        for action in possible_actions:
            _v = self.get_qvalue(state, action)
            if _v > best_value:
                best_value = _v 
                best_action = action

        return best_action

    def get_action(self, state):
        """
        Returns an action which is the best action with possibility of
        (1-epsilon) and returns a random action with a posibility of
        epsilon
        """
        possible_actions = self.get_legal_actions(state)
        
        # if there is no legal actions we return None
        if len(possible_actions) == 0:
            return None

        ep = self.epsilon
        r = np.random.uniform()
        if r < ep:
            action = np.random.choice(possible_actions)
        else:
            action = self.get_best_action(state)

        return action

    def save_qvalue(self, filename='mountain_q'):
        """
        Saving the qvalue dict
        """
        d_ = dict(self._qvalues)
        np.save(filename, np.array(d_))

    def load_qvalue(self, filename='mountain_q'):
        """
        Loading the qvalue dict
        """
        n_ = f"{filename}.npy"
        P = np.load(n_)
        Q_ = defaultdict(lambda: defaultdict(lambda: 0))
        Q_.update(P.item())
        return Q_

# Function for Playing a full episod and train the agent
def play_and_train(env, agent, max_t = 10**4, n=10):
    """
    Function for playing an episode and train the agent
    """
    total_reward = 0.0
    state = env.reset()
    rewards = []
    state_action = []
    for t in range(max_t):
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        state_action.append((state, action))
        rewards.append(reward)

        agent.update(state, action, reward, next_state)
        
        if t >= n:
            agent.update_multiple(state_action[-1*n:], rewards[-1*n:], next_state)
                
        state = next_state
        total_reward += reward

        if done:
            l_state, l_action = state_action[-1]
            l_value = agent.get_qvalue(l_state, l_action)
            
            for i in range(min(n-1, len(state_action)-1)):
                s_, a_ = state_action[-1*i-1]
                v_ = agent.get_qvalue(s_, a_)
                
                agent.update_multiple(state_action[-1*i-1:], rewards[-1*i-1:], next_state= None)
                v_ = agent.get_qvalue(s_, a_)
                
            break
    
    return total_reward


# Defining the learning rates and n_steps we want to test
learning_rates = [0.1, 0.2, 0.5]
steps = [1, 10, 50]

# Defining dict of an agent and env for each parameters compination 
# want to test (learning_rate, n_steps)
agent_env_dict = {}
for lr in learning_rates:
    for n_s in steps:
        agent_env_dict[(lr, n_s)] = {"agent": QLearningAgent(lr, epsilon, gamma, get_legal_actions=lambda s: range(n_actions)),
                                     "env": make_env()}


agent = QLearningAgent(alpha=alpha, epsilon=epsilon,discount=gamma,
                       get_legal_actions=lambda s: range(n_actions))

# Defining the reward dict for each agent
all_rewards = {}
for k in agent_env_dict.keys():
    all_rewards[k] = []

for i in range(episodes):
    for k in agent_env_dict.keys():
        all_rewards[k].append(play_and_train(agent_env_dict[k]['env'], agent_env_dict[k]['agent'], max_t=max_t, n=k[1]))

    dict_tb={f"r_lr{k[0]}_n{k[1]}":all_rewards[k][-1] for k in agent_env_dict.keys()}
    writer.add_scalars("CartPole Rewards", dict_tb, i)

    if i % 100 == 0 and i > 0:
        for k in agent_env_dict.keys():
            agent_env_dict[k]["agent"].epsilon *= decay

writer.close()