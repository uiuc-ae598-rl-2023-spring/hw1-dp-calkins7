# Author: Grace Calkins
# Date: 2/15/2023
# For AE 598RL, UIUC Spring 2023
import numpy as np

def valu_iter(env, tol, gamma = 0.95):
    """
    Performs value iteration following Sutton 4.4
    :param env: environment object
    :param tol: tolerance for convergence
    :param gamma: discount factor
    :return V: optimal state value function
    :return pi: optimal policy
    :return log: log of data from learning for plotting
    """

    # Settings and Initializing
    Delta = 1000
    max_iter = 1000
    V = np.zeros((1, env.num_states))

    log = {
        'n': [0],
        'V': []
    }

    # Value Iteration
    iter = 0
    while Delta > tol and iter < max_iter:
        Delta = 0
        for idx, s in enumerate(range(env.num_states)):
            v = V[0][idx]
            V[0][idx] = 0
            V_a = []
            for idx2, a in enumerate(range(env.num_actions)):
                V_sum = 0
                for idx1, s1 in enumerate(range(env.num_states)):
                    V_sum += env.p(s1, s, a) * (env.r(s, a) + gamma * V[0][idx1])
                V_a.append(V_sum)
            V[0][idx] = max(V_a)
            # V[0][idx] = V_a
            Delta = max(Delta, np.abs(v - V[0][idx]))

        iter = iter + 1
        print(f'VI: {iter} iter')

        # Save number of policy eval iterations for plotting
        log['n'].append(iter)
        log['V'].append(V.copy())

    print(f'Value Iteration Closed in {iter} iter')

    # Get policy to return
    pi = np.zeros((1, env.num_states))
    for idx, s in enumerate(range(env.num_states)):
        V_a = []
        for idx2, a in enumerate(range(env.num_actions)):
            V_sum = 0
            for idx1, s1 in enumerate(range(env.num_states)):
                V_sum += ((env.p(s1, s, a) * (env.r(s, a) + gamma * V[0][idx1])))
            V_a.append(V_sum)
        pi[0][idx] = np.argmax(np.array(V_a))

    return V, pi, log