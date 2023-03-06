# Author: Grace Calkins
# Date: 2/15/2023
# For AE 598RL, UIUC Spring 2023
import numpy as np
from epsilon_greedy import epsilon_greedy, epsilon_greedy_pi_star

def sarsa(env, alpha, epsilon, num_episodes, gamma=0.95):
    """
    Performs SARSA per Sutton 6.4
    :param env: environment object
    :param alpha: step size
    :param epsilon: policy epsilon greedy parameter
    :param num_episodes: number of episodes to learn on
    :param gamma: discount factor
    :return Q: optimal action value function
    :return pi: optimal policy
    :return log: log of data from episodes for plotting
    """
    Q = np.zeros((env.num_states, env.num_actions))

    log = {
        'n': [], # number of episodes
        'return': [],
        'alpha': alpha,
        'epsilon': epsilon
    }

    # Repeat for each episode
    for ii in range(num_episodes):
        # Initialize simulation
        s = env.reset()
        # Choose a according to an e-greedy policy
        a = epsilon_greedy(epsilon, env.num_actions, s, Q)
        # Simulate until episode is done
        done = False
        episode_return = 0
        counter = 0
        while not done:
            # take action a, observe r and sprime
            (s1, r, done) = env.step(a)
            # Choose aprime from sprime using policy derived from Q
            a1 = epsilon_greedy(epsilon, env.num_actions, s1, Q)
            # Reassign Q
            Q[s, a] += alpha*(r + gamma*Q[s1, a1] - Q[s, a])
            # Get return for logging
            episode_return += gamma**counter * r
            counter = counter + 1
            s = s1
            a = a1

        log['n'].append(ii)
        log['return'].append(episode_return)

        # Print if ii is a factor of 500
        if (ii % 500) == 0:
            print(f'SARSA Episode {ii} / {num_episodes}')

    # Get pi according to epsilon-greedy for all states
    pi = epsilon_greedy_pi_star(env.num_states, Q)

    return Q, pi, log






