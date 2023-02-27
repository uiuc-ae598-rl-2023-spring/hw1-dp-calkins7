import random
import numpy as np

def epsilon_greedy(epsilon, num_actions, s, Q_pi):
    """
    Samples an action for a given state following e-greedy policy based on the given Q(s,a)
    :param epsilon: e small
    :param num_actions: number of actions in environment
    :param s: state
    :param Q_pi: [num_states x num_actions] sized matrix of the action value function
    :return a_greedy[0]: the greedy action
    """
    # Identify greedy action
    greedy = np.argmax(Q_pi[s][:])

    # Compute epsilon greedy probabilities
    weights = np.ones(num_actions)*epsilon/num_actions
    weights[greedy] = 1-epsilon+epsilon/num_actions

    # Choose action with epsilon greedy weights
    a_greedy = random.choices(range(num_actions), weights=weights, k=1)

    return a_greedy[0]

def epsilon_greedy_pi_star(num_states, Q_pi):
    """
    Computes the e-greedy policy for a given Q(s,a) with no samplings
    :param num_states: number of states in env
    :param Q_pi: [num_states x num_actions] sized matrix of the action value function
    :return pi_star: [num_actions] array of optimal policy for each state
    """
    pi_star = np.zeros(num_states)
    # Loop over all states
    for s in range(num_states):
        # Identify greedy action
        greedy = np.argmax(Q_pi[s][:])

        pi_star[s] = greedy

    return pi_star