###################################################################
# This code is one of many codes which aim to provide a simple tool
# which can be helpful to test and visualize RL algorithms
###################################################################
# This specific code is for testing Q learing algorithm
# This code by default uses the q_table that has the name q_table.csv
# this cane be changed while running the code using the argument
# --q_table_file. Multible other parameters can be changed using 
# arguments like training arguments as discount value, epsilon, and 
# learning rate. It is also possible to chosse to render while 
# while training

import gym
import time
import numpy as np
import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt
from utils import plot_data, create_plot

# plt.ion()

parser = argparse.ArgumentParser(description='Python Q-learning example')
parser.add_argument('--test', default=False, action='store_true',
                    help='for testing existing Q-table. When not mentioned will train the q-table')
parser.add_argument('--q_table_file', type=str, default='q_table.csv', metavar='The Q table file path',
                    help='The name of csv file that contains the q_table (default: q_table.csv)')
parser.add_argument('-u','--use_existing_q_table', default=False, action='store_true',
                    help='True for using existing Q-table, Flase for starting from the Beginning')
parser.add_argument('-s','--save_q_table', default=False, action='store_true',
                    help='True for saving the trained Q-table (This will replace the file that have the name used as argument for --q_table_file)')
parser.add_argument('-t','--test_after_training', default=False, action='store_true',
                    help='True for testing 20 episods after finishing the training the trained')
parser.add_argument('-g','--gamma', type=float, default=0.6, metavar='discount',
                    help='discount factor (default: 0.6)')
parser.add_argument('-a', '--alpha', type=float, default=0.1, metavar='learning rate',
                    help='learning rate value (default: 0.1)')
parser.add_argument('-d', '--decay', type=float, default=0.95, metavar='Decay',
                    help='Decay factor (default: 0.95)')
parser.add_argument('-e','--epsilon', type=float, default=0.99, metavar='Epsilon',
                    help='Starting epsilon value(default: 0.99)')
parser.add_argument('--episods', type=int, default=5000,
                    help='Number of training episods')
parser.add_argument('--render_each', type=int, default=100,
                    help='Number of episods before rendering (default:100 "after each 100 episods a full episod will be shown if the argument --render_while_training is set to True")')
parser.add_argument('-r' ,'--render_while_training', default=False, action='store_true',
                    help='True for rendering first episod of each 100 episod while training')


args = parser.parse_args()

# Defining the environment (Gym Taxi version 2 environment)
env = gym.make("Taxi-v2").env
env.render()
print("Action space: {}".format(env.action_space))
print("State space: {}".format(env.observation_space))

# loading chosen arguments
data_file=args.q_table_file
train = not args.test
use_existing_q_table = args.use_existing_q_table
test_after_training = args.test_after_training
gamma = args.gamma
alpha = args.alpha
decay = args.decay
epsilon = args.epsilon
episods = args.episods
render_while_training = args.render_while_training
render_each = args.render_each
save_q_table = args.save_q_table

if train:
    print(f"Training will start now for {episods} episods with the following parameters:")
    print(f"Discount factor (Gamma)= {gamma}, Learning rate (Alpha)= {alpha}, decay factor= {decay}, Exploration probability (epsilon)={epsilon}")
    print(f"render while training= {render_while_training}, render_each= {render_each}, test the results after training= {test_after_training}")
    print(f"q_table file name= {data_file}, save Q-table after training= {save_q_table}")

variable = input('Press any key to continue!: ')
	

# Loading the existing Q-table of Creating a new Q-table
# for Q-values for each (state, action) pair
if os.path.isfile(data_file) and use_existing_q_table:
    q_table = pd.read_csv(data_file,header=None).to_numpy()
    # Check for matching shape
    q_shape = (env.observation_space.n, env.action_space.n)
    meassage = """\nThe q_table you chose has shape of {} which does not match the needed shape of {}Please make sure that the q-table you use has the right dimenstions!""".format(q_table.shape, q_shape)
    assert q_table.shape == q_shape, meassage
else:
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

import random
from IPython.display import clear_output

if train:
    fig, ax = create_plot()
    all_rewards = []
    all_penallties = []
    avarage_rewards = []

    # start of training
    for i in range(episods):
        state = env.reset()

        epoches, penalties, rewards = 0, 0, 0
        done = False

        while not done:
            if i % render_each ==0:
                if render_while_training:
                    env.render()
                    print(q_table[state])
                    time.sleep(0.05)

            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            # Updating the Q value for the pair (state, action) as follows:
            # New_Q_value(state, action) = (1-a)* Old_Q_Value + a*(R+ G*Value(next_state)))
            new_value = (1 - alpha) * old_value + alpha*(reward + gamma *next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1
            
            state = next_state
            epoches += 1
            rewards += reward

        all_rewards.append(rewards)
        if i ==0:
            avarage_rewards.append(rewards)
        elif(i < 100):
            avarage_rewards.append(sum(all_rewards)*1.0/(i+1))
        else:
            avarage_rewards.append(sum(all_rewards[-100:])/100.0)

        if i % 100 == 0 and i>0:
            epsilon = epsilon*decay
            print("___________________________________________")
            print(f"Episode: {i}\nCurrent epsilon: {epsilon}")
            print(f"Avarage reward over past 100 episod: {sum(all_rewards[-100:])/100.0}")
            
            # Provid the ploter with the data to plot
            data_dict = {"all rewards": all_rewards, "avg rewards": avarage_rewards}
            plot_data(ax, data_dict)


    if save_q_table:
        df = pd.DataFrame(q_table)
        df.to_csv(data_file, index= False, header=False)
if (not train) or test_after_training:
    if not train:
        q_table = pd.read_csv(data_file,header=None).to_numpy()
    
    # Sleeping for 3 seconds to get ready for testing
    if test_after_training:
        print("Testing the trained Q-table will start in 3 seconds:")
        time.sleep(1)
        print(2)
        time.sleep(1)
        print(1)
        time.sleep(1)

    epsilon = 0.00
    #print(q_table)
    for i in range(20):
        state = env.reset()
        done = False
        steps = 0
        while not done:
            clear_output(wait=True)
            env.render()
            #print(q_table[state])
            time.sleep(0.05)
            
            action = np.argmax(q_table[state])
            state, re, done, info = env.step(action)
            steps +=1

        # Sleeping after each episod for 0.5 sec
        print("done in {} steps".format(steps))
        time.sleep(0.5)

