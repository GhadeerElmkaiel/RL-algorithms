# Q-Learning

Testing table Q-Learning on multiple different envs (with Discrete and Continuous observation (after Wrapping the environments with observation wrapper that achieve rounding over the observation to a specific precision)

There are multiple environments for testing table Q-Learning like (Taxi-v2, MountainCar-v0, and CartPole-v0) environments.
### Taxi-v2:
Using table to represent the Q value for each pair (state, action) "The number of pairs (state, action) is limited (500 states, 6 actions)" 
### MountainCar-v0:
Using a Wrapper for the environment it is possible to change the observations from continuous to discrete (limited number) by rounding them to a specific precision. Then it is possible to use Q-learning to teach an agent to climp the mountain.

In this code it is possible to change multiple training parameters like learning rat, epsilon, decay, number of TD steps and multiple other parameters.
### CartPole-v0:
As in MountainCar it is possible to get discrete observations and Use standart Q-learning to teach an agent to balance the pole.

In this code it is possible to change multiple training parameters like learning rat, epsilon, decay, number of TD steps and multiple other parameters.

### Visualize_observation:
Code for visualizing the observations' distribution so it is possible to digitize the observation and use a table Q-learining. 


