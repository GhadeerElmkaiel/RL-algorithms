###################################################################
# This code is one of many codes which aim to provide a simple tool
# which can be helpful to test and visualize RL algorithms
###################################################################
# This specific code is for testing Deep Cross-entropy algorithm 
# in the gym CartPole-v0 environment.
# It is possible to change and tune # multiple parameters as training
# parametars (learning rate).
# for each state_action pair, by doing so it is possible to understand 
# how each parameter can affect the training process

import gym
import numpy as np
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter

from torch import nn
from torch.nn import functional as f


parser = argparse.ArgumentParser(description='CartPole Deep Cross-entropy example')

parser.add_argument('--nn_parameters', type=str, default='parameters.csv', metavar='The parameters file path',
                    help='The name of file that contains the neural network parameters (default: )')
parser.add_argument('-u','--use_existing_parameters', default=False, action='store_true',
                    help='True for using existing parameters, Flase for starting from the Beginning')
parser.add_argument('-s','--save_parameters', default=False, action='store_true',
                    help='True for saving the trained Q-table (default: False) (This will replace the file that have the name used as argument for --q_table_file)')
parser.add_argument('-g','--gamma', type=float, default=0.99, metavar='discount',
                    help='discount factor (default: 0.99)')
parser.add_argument('-a', '--alpha', type=float, default=0.2, metavar='learning rate',
                    help='learning rate value (default: 0.2)')
parser.add_argument('-h', '--hidden_neurons', type=int, default=20, metavar='hidden neurons',
                    help='Number of hidden neurons in the hidden layer (default: 20)')
# parser.add_argument('-d', '--decay', type=float, default=0.95, metavar='Decay',
#                     help='Decay factor (default: 0.95)')
# parser.add_argument('-e','--epsilon', type=float, default=0.99, metavar='Epsilon',
#                     help='Starting epsilon value (default: 0.99)')
parser.add_argument('--episodes', type=int, default=50000,
                    help='Number of training episodes (default:5000)')
parser.add_argument('--max_t', type=int, default=10000,
                    help='Max steps in each episod (default:10000)')
parser.add_argument('--render_each', type=int, default=100,
                    help='Number of episodes before rendering (default:100 "after each 100 episodes a full episod will be shown if the argument --render_while_training is set to True")')
parser.add_argument('--render_while_training', default=False, action='store_true',
                    help='True for rendering first episod of each 100 episod while training (default: False)')
# parser.add_argument('-i', '--initial_q_value', type=float, default=0.0, metavar='Initial Q-values',
#                     help='The initial Q values for each state_action pair (default: 0.0)')
parser.add_argument('-n', '--n_steps', type=int, default=10, metavar='n steps TD',
                    help='The number of steps to use to calculat q_value (defaule: 10, use -1 for MC learning)')
parser.add_argument('--done_reward', type=int, default=0, metavar='Final Reward',
                    help='The reward that the agent gets when finishing an episod (defaule: 0)')

args = parser.parse_args()

# Loading chosen arguments
data_file               = args.nn_parameters
use_existing_parameters = args.use_existing_parameters
gamma                   = args.gamma
alpha                   = args.alpha
hidden_neurons          = args.hidden_neurons
# decay                   = args.decay
# epsilon                 = args.epsilon
episodes                = args.episodes
render_while_training   = args.render_while_training
max_t                   = args.max_t
render_each             = args.render_each
save_parameters         = args.save_parameters
# initial_q_value         = args.initial_q_value
n_steps                 = args.n_steps
done_reward             = args.done_reward


def make_env(name = "CartPole-v0"):
    env = gym.make(name)
    return env

env = make_env()
s = env.reset()
n_actions           = env.action_space.n
n_observations      = env.observation_space.n
print("Action space: {}".format(env.action_space))
print("Number of actions: {}".format(n_actions))
print("State space: {}".format(env.observation_space))
print("State: {}".format(s))


# Defining the Policy Neural Network
Class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()

        self.fc1    = nn.Linear(n_observations, hidden_neurons)
        self.b1     = nn.BatchNorm1d(hidden_neurons)
        self.fc2    = nn.Linear(hidden_neurons, hidden_neurons)
        self.b2     = nn.BatchNorm1d(hidden_neurons)
        self.fc3    = nn.Linear(hidden_neurons, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.b1(x)
        x = F.relu(self.fc2(x))
        x = self.b2(x)
        x = Fsoftmax(self.fc3(x))

        return x






writer = SummaryWriter(comment=f"CartPole_D_C_E")

# Define The agent
class DeepCrossEntropyAgent:
    def __init__(self, alpha, discount):
        """
        Deep Cross-Entropy Agent
        Class variables
        - self.policy
        - self.alpha
        - self.discount

        Class functions
        - self.train(elite state_action pairs, elite rewards):
            { Update the neural network so it predict the elite actions in the 
              corresponding states}
        - self.get_probabitlities(state)
            { Returns probabilities of taking each action in this state}
        - self.get_best_action(state)
            { Returns the best action possible in this state according to
            the current neural network}
        - self.get_action(state)
            { Returns an action which is the best action with possibility of
             (1-epsilon) and returns a random action with a posibility of
             epsilon}
        """
        self.alpha = alpha
        self.discount = discount
        self.policy = Policy()

    def train(self, state, action, reward, next_state):
        """
        Update the neural network so it predict the elite actions in the 
        corresponding states
        """
        g = self.discount
        a = self.alpha
        
        #TODO
        pass

    def get_probabitlities(self, state):
        """
        Returns probabilities of taking each action in this state
        """
        #TODO
        pass

    def get_best_action(self, state):
        """
        Returns the best action possible in this state according to
        the current neural network
        """
        #TODO
        pass

    def get_action(self, state):
        """
        Returns an action using the neural output as a probability distribution
        for the actions
        """
        #TODO
        pass

    def save_parameters(self, filename='CartPole'):
        """
        Saving the qvalue dict
        """
        #TODO
        pass

    def load_parameters(self, filename='CartPole'):
        """
        Loading the qvalue dict
        """
        #TODO
        pass



# Function for Playing a full episod and train the agent
def play_and_train(env, agent, max_t = 10**4):
    """
    Function for playing an episode and train the agent
    """
    pass

# Defining dict of an agent and env for each learning rate we
# want to test 
learning_rates = [0.1, 0.2]
agent_env_dict = {}
for lr in learning_rates:
    agent_env_dict[lr] = {"agent": DeepCrossEntropyAgent(lr, gamma),
                          "env": make_env()}

agent = DeepCrossEntropyAgent(alpha=alpha,discount=gamma)

# Defining the reward dict for each agent
all_rewards = {}
for k in agent_env_dict.keys():
    all_rewards[k] = []

for i in range(episodes):
    for k in agent_env_dict.keys():
        all_rewards[k].append(play_and_train(agent_env_dict[k]['env'], agent_env_dict[k]['agent'], max_t=max_t, n=k[1]))

    dict_tb={f"r_lr{k}":all_rewards[k][-1] for k in agent_env_dict.keys()}
    writer.add_scalars("CartPole Cross-entropy", dict_tb, i)


writer.close()
