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
    tol = 1e-10

    log = {
        'n': [0],
        'V': []
    }

    while not policy_stable and m <= max_iter_main:
        # Policy Evaluation
        V, n = poli_eval(env, V, pi, gamma, tol, max_iter)

        print(f'Iter {m}: Policy Evaluation Closed in {n} iter')
        if n == max_iter:
            print("Max iteration reached in policy evaluation before convergence")

        # Save number of policy eval iterations for plotting
        log['n'].append(n)
        log['V'].append(V.copy())

        # Policy Improvement
        policy_stable, pi = poli_improve(env, pi, V, gamma)

        if policy_stable:
            print("Policy Iteration Complete with " + str(m) + " Total Iterations")
            return V, pi, log

        m = m + 1

    if m == max_iter_main:
        print("Max iteration reached in policy iteration before convergence")

    return V, pi, log


def poli_eval(env, V, pi, gamma, tol, max_iter):
    """
    Performs policy evaluation
    :param env: environment object
    :param V: input state value function
    :param pi: policy
    :param gamma: discount factor
    :param tol: convergence tolerance
    :param max_iter: maximum allowable iterations
    :return V: updated state value function
    :return n: number of policy eval iterations
    """
    n = 0
    # Policy Evaluation
    Delta = np.inf
    while Delta >= tol and n <= max_iter:
        Delta = 0
        for idx, s in enumerate(range(env.num_states)):
            v = V[0][idx]
            V[0][idx] = 0
            # Sum over all sprime states
            for idx1, s1 in enumerate(range(env.num_states)):
                V[0][idx] += (env.p(s1, s, pi[0][idx]) * (env.r(s, pi[0][idx]) + gamma * V[0][idx1]))
            # Update Delta
            Delta = max(Delta, np.abs(v - V[0][idx]))
            # print(n, Delta)
        # print(V)
        n = n + 1

    return V, n

def poli_improve(env, pi, V, gamma):
    """
    Performs greedy policy improvement
    :param env: environment object
    :param pi: input policy
    :param V: state value function
    :param gamma: discount factor
    :return policy_stable: whether the policy is stable (ends PI loop)
    :return pi: improved policy
    """
    policy_stable = True
    for idx, s in enumerate(range(env.num_states)):
        old_action = pi[0][idx]
        V_sum = np.zeros(env.num_actions)
        # Sum over all possible sprime and actions
        for idx1, s1 in enumerate(range(env.num_states)):
            for idx2, a in enumerate(range(env.num_actions)):
                V_sum[idx2] += (env.p(s1, s, a) * (env.r(s, a) + gamma * V[0][idx1]))

        # Get the argmax of all future possibilities to get optimal policy
        pi[0][idx] = np.argmax(V_sum)

        if old_action != pi[0][idx]:
            policy_stable = False

    return policy_stable, pi