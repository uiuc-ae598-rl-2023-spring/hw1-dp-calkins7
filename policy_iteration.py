# Author: Grace Calkins
# Date: 2/15/2023
# For AE 598RL, UIUC Spring 2023

import numpy as np


def poli_iter(env, gamma = 0.95):
    """
    Performs Policy iteration per Sutton Ch 4.3
    :param env: environment object
    :param gamma: discount factor
    :return V: state value function
    :return pi: optimal policy
    :return log: data on number of iterations and value function for each iteration for plotting
    """

    # Initialize
    V = np.zeros((1, env.num_states)) # Value function
    pi = np.ones((1, env.num_states), dtype = int)  # Policy

    policy_stable = False
    max_iter_main = 1000
    m = 0
    max_iter_main = 1000

    # Policy Evaluation Parameters
    max_iter = 1000
    tol = 1e-16

    log = {
        'n': [0],
        'V': []
    }

    while not policy_stable and m <= max_iter_main:
        n = 0
        # Policy Evaluation
        Delta = np.inf
        while Delta >= tol and n <= max_iter:
            Delta = 0
            for idx, s in enumerate(range(env.num_states)):
                v = V[0][idx]
                V[0][idx] = 0
                V_sum = 0
                # Sum over all sprime states
                for idx1, s1 in enumerate(range(env.num_states)):
                    V_sum += (env.p(s1, s, pi[0][idx]) * (env.r(s, pi[0][idx]) + gamma*V[0][idx1]))
                V[0][idx] = V_sum
                # Update Delta
                Delta = max(Delta, np.abs(v - V[0][idx]))
                # print(n, Delta)
            # print(V)
            n = n + 1

        print(f'Iter {m}: Policy Evaluation Closed in {n} iter')
        if n == max_iter:
            print("Max iteration reached in policy evaluation before convergence")

        # Save number of policy eval iterations for plotting
        log['n'].append(n)
        log['V'].append(V.copy())

        # Policy Improvement
        policy_stable = True
        for idx, s in enumerate(range(env.num_states)):
            old_action = pi[0][idx]
            V_sum = np.zeros(env.num_actions)
            # Sum over all possible sprime and actions
            for idx1, s1 in enumerate(range(env.num_states)):
                for idx2, a in enumerate(range(env.num_actions)):
                    V_sum[idx2] = V_sum[idx2] + (env.p(s1, s, a) * (env.r(s, a) + gamma * V[0][idx1]))

            # Get the argmax of all future possibilities to get optimal policy
            pi[0][idx] = np.argmax(V_sum)

            if old_action != pi[0][idx]:
                policy_stable = False

        if policy_stable:
            print("Policy Iteration Complete with " + str(m) + " Total Iterations")
            return V, pi, log

        m = m + 1

    if m == max_iter_main:
        print("Max iteration reached in policy iteration before convergence")

    return V, pi, log