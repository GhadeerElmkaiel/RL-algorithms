###################################################################
# This code is used to visualize the observations' distribution of 
# an environment in order to digitize the observation values by 
# rounding each value # to a specifice precision so it is possible 
# to use table Q-learning
###################################################################

import gym
import time
import numpy as np
import argparse
from matplotlib import pyplot as plt

def Visualizing_observation_distribution(observation):
    obs = np.array(observation)

    f, axis = plt.subplots(2, 2, figsize=(8, 4), sharey=True)
    for i, title in enumerate(['Position', 'Velocity', 'Pole Angle', 'Pole Velocity']):
        ax = axis[i//2, i%2]
        l = list(zip(*obs))[i]
        ax.hist(l, bins=100)
        ax.set_title(title)
        xmin, xmax = ax.get_xlim()
        ax.set_xlim(min(xmin, -xmax), max(xmax, -xmin))
        ax.grid()
    f.show()

env = gym.make(CartPole-v0).env

seen_obs = []
for _ in range(10000):
    seen_obs.append(list(env.reset()))
    done = False
    while not done:
        s, r, done, _ = env.step(env.action_space.sample())
        seen_obs.append(list(s))
       
Visualizing_observation_distribution(seen_obs)