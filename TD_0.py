# Author: Grace Calkins
# Date: 2/15/2023
# For AE 598RL, UIUC Spring 2023
import numpy as np


def TD_0(pi, env, alpha, num_episodes, gamma=0.95):
    """
    Performs TD(0) based on Sutton 6.1
    :param gamma: discount factor
    :param num_episodes: number of episodes to learn on
    :param env: environment object
    :param pi: policy to learn the value function of
    :param alpha: step size
    :return V: learned state value function
    """

    if not(0 < alpha <= 1):
        print("alpha value input to TD(0) invalid")
        return

    V = np.zeros(env.num_states)

    # For each episode loop
    for ii in range(num_episodes):
        # Initialize simulation
        s = env.reset()

        # Simulate until episode is done
        done = False
        while not done:
            # choose action according to pi
            a = pi[s]
            # take action, observe sprime and r
            (s1, r, done) = env.step(a)
            V[s] = V[s] + alpha*(r + gamma*V[s1] - V[s])
            s = s1

    return V

