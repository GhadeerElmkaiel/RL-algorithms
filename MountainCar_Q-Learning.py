###################################################################
# This code is one of many codes which aim to provide a simple tool
# which can be helpful to test and visualize RL algorithms
###################################################################
# This specific code is for testing TD Q learing algorithm in  the 
# Gym MountainCar-v0 environment.
# It is possible to change and tune # multiple parameters as training
# parametars (epsilon, learning rate, decay) and the initial Q-values 
# for each state_action pair. by doing so it is possible to understand 
# how each parameter can affect the training process


import gym
import time
import numpy as np
import argparse
from matplotlib import pyplot as plt
from utils import plot_data, create_plot
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from gym.core import ObservationWrapper



parser = argparse.ArgumentParser(description='Python Q-learning example')
# parser.add_argument('--test', default=False, action='store_true',
#                     help='for testing existing Q-table. When not mentioned will train the q-table')
parser.add_argument('--q_table_file', type=str, default='q_table_mountain.csv', metavar='The Q table file path',
                    help='The name of csv file that contains the q_table (default: q_table_mountain.csv)')
parser.add_argument('-u','--use_existing_q_table', default=False, action='store_true',
                    help='True for using existing Q-table, Flase for starting from the Beginning')
parser.add_argument('-s','--save_q_table', default=False, action='store_true',
                    help='True for saving the trained Q-table (default: False) (This will replace the file that have the name used as argument for --q_table_file)')
parser.add_argument('-g','--gamma', type=float, default=0.9, metavar='discount',
                    help='discount factor (default: 0.9)')
parser.add_argument('-a', '--alpha', type=float, default=0.1, metavar='learning rate',
                    help='learning rate value (default: 0.1)')
parser.add_argument('-d', '--decay', type=float, default=0.95, metavar='Decay',
                    help='Decay factor (default: 0.95)')
parser.add_argument('-e','--epsilon', type=float, default=0.99, metavar='Epsilon',
                    help='Starting epsilon value (default: 0.99)')
parser.add_argument('--episods', type=int, default=5000,
                    help='Number of training episods (default:5000)')
parser.add_argument('--max_t', type=int, default=10000,
                    help='Max steps in each episod (default:10000)')
parser.add_argument('--render_each', type=int, default=100,
                    help='Number of episods before rendering (default:100 "after each 100 episods a full episod will be shown if the argument --render_while_training is set to True")')
parser.add_argument('--render_while_training', default=False, action='store_true',
                    help='True for rendering first episod of each 100 episod while training (default: False)')
parser.add_argument('-i', '--initial_q_value', type=float, default=-10.0, metavar='Initial Q-values',
                    help='The initial Q values for each state_action pair (default: -10.0)')
#TODO
# Add the possiblity of using MC
parser.add_argument('-n', '--n_steps', type=int, default=10, metavar='n steps TD',
                    help='The number of steps to use to calculat q_value (defaule: 10)')
parser.add_argument('--done_reward', type=int, default=0, metavar='Final Reward',
                    help='The reward that the agent gets when finishing an episod (defaule: 0)')


args = parser.parse_args()

def write_dict(dif_dict, filename = "dict.txt"):
    f = open(filename, "w")
    keys = dif_dict.keys()
    f.write(f"number of states: {len(keys)}\n\n")
    for i in keys:
        keys2 = dif_dict[i].keys()
        for j in keys2:
            f.write(f"{{{i}\t:{j}\t}} : {dif_dict[i][j]}\n")
        f.write("\n")


# Wrapper class to wrap the observation and do digitization 
class Observation_Wrapper (ObservationWrapper):
    def observation(self, state):
        # the number of digits was decided accourding to the distribution of tje 
        # values for each state 
        state_ = [round(state[0], 2), round(state[1], 3)]
        state = state_
        return tuple(state)

# Defining the environment (Gym Taxi version 2 environment) with our wrapper
# The wrapper round the states to only two digites so we can represent it
# with a finit Q-table
env = Observation_Wrapper(gym.make("MountainCar-v0").env)
s = env.reset()
# plt.imshow(env.render('rgb_array'))
n_actions = env.action_space.n
print("Action space: {}".format(env.action_space))
print("Number of actions: {}".format(n_actions))
print("State space: {}".format(env.observation_space))
print("State: {}".format(s))

# # loading chosen arguments
data_file               = args.q_table_file
# train                 = not args.test
use_existing_q_table    = args.use_existing_q_table
# test_after_training   = args.test_after_training
gamma                   = args.gamma
alpha                   = args.alpha
decay                   = args.decay
epsilon                 = args.epsilon
episods                 = args.episods
render_while_training   = args.render_while_training
max_t                   = args.max_t
render_each             = args.render_each
save_q_table            = args.save_q_table
initial_q_value         = args.initial_q_value
n_steps                 = args.n_steps
done_reward             = args.done_reward

# Define the writer for tensorboard
writer = SummaryWriter(comment=f"M-Car_Q_L.{n_steps}_steps.ep{epsilon}.lr{alpha}.d{gamma}")

# Define The agent
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions, done_reward=done_reward):
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
        self.done_reward = 

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
        # in the case of Terminal State
        if next_state == None:
            _q = (1-a)*self.get_qvalue(state, action) + \
                a*(reward+ self.done_reward)
        else:
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
            _q = reward + self.done_reward
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

        return action

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


# env_1_1 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_2_1 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_5_1 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_1_10 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_2_10 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_5_10 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_1_50 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_2_50 = Observation_Wrapper(gym.make("MountainCar-v0").env)
# env_5_50 = Observation_Wrapper(gym.make("MountainCar-v0").env)

agent = QLearningAgent(alpha, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_2_1 = QLearningAgent(alpha=0.2, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_5_1 = QLearningAgent(alpha=0.5, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_1_10 = QLearningAgent(alpha=0.1, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_2_10 = QLearningAgent(alpha=0.2, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_5_10 = QLearningAgent(alpha=0.5, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_1_50 = QLearningAgent(alpha=0.1, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_2_50 = QLearningAgent(alpha=0.2, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))
# agent_5_50 = QLearningAgent(alpha=0.5, epsilon, gamma, get_legal_actions=lambda s: range(n_actions))

def play_and_train(env, agent, t_max=10**4, n=10):
    """
    This function run a full game, it get action from the agent using 
    e-greedy policy, train the agent using update function when it is
    possible, and it returns the total reward of the episod
    """
    total_reward = 0.0
    rewards = []
    state = env.reset()
    state_action = []
    for t in range(t_max):
        # get the action from the agent
        action = agent.get_action(state)
        next_state, reward, done, _ = env.step(action)

        state_action.append((state, action))
        rewards.append(reward)
        # print(f"step {{{t}}}: state {{{state}}}, action: {{{action}}}")

        # train the agent and updating the value for current state
        # agent.update(state, action, reward, next_state)
        if t > n:
            agent.update_multiple(state_action[-1*n:], rewards[-1*n:], next_state)
                
        state = next_state
        total_reward += reward
        if done:
            # print("Done it in this episod!")
            # print(f"Last state_action pair was: {state_action[-1]}")
            l_state, l_action = state_action[-1]
            l_value = agent.get_qvalue(l_state, l_action)
            # print(f"Last state_action value before updating was: {l_value}")
            # print("__________________________________")
            for i in range(n-1):
                # print(f"Updating q_value for state_action pair: {state_action[-1*i-1]}")
                s_, a_ = state_action[-1*i-1]
                v_ = agent.get_qvalue(s_, a_)
                # print(f"Value before updating: {v_}")
                agent.update_multiple(state_action[-1*i-1:], rewards[-1*i-1:], next_state= None)
                v_ = agent.get_qvalue(s_, a_)
                # print(f"Value after updating: {v_}")
                # print("###########")
            break

    return total_reward

print("Starting the training")

all_rewards = []
avarage_rewards = []

fig, ax = create_plot()

# Training
for i in range(episods):
    all_rewards.append(play_and_train(env, agent, t_max=max_t, n=n_steps))

    if i ==0:
        avarage_reward = all_rewards[0]
    elif(i < 100):
        avarage_reward = sum(all_rewards)*1.0/(i+1)
    else:
        avarage_reward = sum(all_rewards[-100:])/100.0

    writer.add_scalars('Rewards', {'reward': all_rewards[-1],
                                   'avg_r': avarage_reward}, i)

    # plot_dict = {'Rewards': all_rewards, 'Avarage Rewards': avarage_rewards}

    if i % 100 == 0 and i > 0:
        agent.epsilon = agent.epsilon*decay
        # plot_data(ax, plot_dict)
        name = f"mountain_{int(i/100)}"
        # agent.save_qvalue(name)
        if i % 1000 == 0 and i > 0 and save_q_table:
            name = f"dict_{int(i/1000.0)}_n_{n_steps}.txt"
            #write_dict(agent._qvalues, filename=name)

writer.close()
# write_dict(agent._qvalues)
