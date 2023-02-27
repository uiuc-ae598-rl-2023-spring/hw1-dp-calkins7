import matplotlib.pyplot as plt
import numpy as np
import os

def plot_state_val_func(log, name, path, modelFreeTag=False, logPass=0):
    """
    Plots state value function versus the state
    :param log: data from run
    :param name: figure name
    :param path: figure save path
    :param modelFreeTag: True if the simulation is model-free, false if model-based
    :param logPass: contains extra data if the simulation is model-free for labelling
    :return: figure
    """
    # Unpack model-free params if needed
    if logPass != 0:
        eps = logPass['epsilon']
        alpha = logPass['alpha']

    # Extract V depending on type of data passed in
    V = log['V']
    if len(np.shape(V)) > 1:
        V = log['V'][-1]
        V = V[0]
    s = np.size(V)

    # Plot
    ax = plt.subplot()
    ax.plot(np.arange(s), V)
    plt.grid()
    plt.xlabel('State')
    plt.ylabel('V(s)')
    if modelFreeTag:
        plt.title('Learned Value Function, $\epsilon$ = %.3f, α = %.3f' % (eps, alpha))
    else:
        plt.title('Learned Value Function')
    plt.savefig(os.path.join('.', 'figures', path, name))
    plt.close()
    return


def plot_mean_value_func(log, name, path):
    """
    Plots mean value function (learning rate for policy and value iteration)
    :param log: data from episode
    :param name: figure name
    :param path: figure save path
    :return: figure
    """
    V_mean = []
    for ii in range(len(log['V'])):
        V_mean.append(np.mean(log['V'][ii]))

    ax = plt.subplot()
    ax.plot(log['n'][0:-1], V_mean)
    plt.xlabel('Number of Policy Eval Iter')
    plt.ylabel('Mean V(s)')
    plt.savefig(os.path.join('.', 'figures', path, name))
    plt.close()
    return


def plot_example_traj_grid(pi_star, env, name, modelFreeTag=False, logPass=0):
    """
    Plots gridworld example traj
    :param pi_star: optimal policy
    :param env: environment object
    :param name: name of figure to save
    :param modelFreeTag: True if model-free, False if model-based
    :param logPass: extra data if model-free
    :return: figure
    """
    # Unpack
    if logPass != 0:
        eps = logPass['epsilon']
        alpha = logPass['alpha']

    # Initialize simulation
    s = env.reset()
    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
    }

    pi_star = np.squeeze(pi_star)

    # Simulate until episode is done
    done = False
    while not done:
        a = pi_star[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)

    # Plot data and save to png file
    ax1 = plt.subplot()
    ax1.plot(log['t'], log['s'])
    ax1.plot(log['t'][:-1], log['a'])
    ax1.plot(log['t'][:-1], log['r'])
    plt.legend(['s', 'a', 'r'])
    if modelFreeTag:
        plt.title('Trajectory, $\epsilon$ = %.3f, α = %.3f' % (eps, alpha))
    else:
        plt.title('Trajectory')

    plt.savefig(os.path.join('.', 'figures', 'gridworld', name))
    plt.close()
    return


def plot_gridworld_policy(pi_star, name, modelFreeTag=False, logPass=0):
    """
    Plots gridworld policy
    :param pi_star: policy to plot
    :param name: figure name
    :param modelFreeTag: True if model-free, False if model-based
    :param logPass: extra data if model-free
    :return: figure
    """
    # Unpack
    if logPass != 0:
        eps = logPass['epsilon']
        alpha = logPass['alpha']

    row = 10
    pi_star = np.squeeze(pi_star)
    fig, ax = plt.subplots(1,1)

    counter = 0
    for i in range(5):  # row
        col = 0
        for j in range(5):  # col
            pi = pi_star[counter]
            # get coords

            if pi == 0:  # right
                x = col+0.5
                y = row-1
                dx = 1
                dy = 0
            elif pi == 2:  # left
                x = col+1.5
                y = row-1
                dx = -1
                dy = 0
            elif pi == 1:  # up
                x = col+1
                y = row-1.5
                dx = 0
                dy = 1
            elif pi == 3:  # down
                x = col+1
                y = row-0.5
                dx = 0
                dy = -1

            plt.arrow(x, y, dx, dy, width = 0.05, head_width=0.3, head_length=0.2, color='k')
            counter = counter + 1
            col = col + 2
        row = row - 2

    ax.set_xticks(np.arange(0, 10, 2))
    ax.set_yticks(np.arange(0, 10, 2))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.xlim([0, 10])
    plt.ylim([0,10])
    plt.grid()
    if modelFreeTag:
        plt.title('Gridworld Policy, $\epsilon$ = %.3f, α = %.3f' % (eps, alpha))
    else:
        plt.title('Gridworld Policy')
    plt.savefig(os.path.join('.', 'figures', 'gridworld', name))
    plt.close()
    return


def plot_return(log, name, path):
    """
    Plots return over episode number
    :param log: data from episode
    :param name: figure save name
    :param path: figure save path
    :return: figure
    """
    G = log['return']
    n = log['n']

    ax = plt.subplot()
    ax.plot(n, G)
    plt.grid()
    plt.xlabel('Episode Number')
    plt.ylabel('G')
    plt.title("Episode Return")
    plt.savefig(os.path.join('.', 'figures', path, name))
    plt.close()
    return


def plot_learning_curve(logs, name, path, eOrATag):
    """
    Plots learning curve for model-free algorithms for varying eps or alpha
    :param logs: data from varying episodes
    :param name: figure save name
    :param path: figure save path
    :param eOrATag: 'e' is eps is varied, 'a' is alpha is varied
    :return: figure
    """
    logSize = len(logs)

    ax = plt.subplots()

    for ii in range(logSize):
        log = logs[ii]
        G = log['return']
        n = log['n']

        if eOrATag == 'e':
            plt.plot(n, G, label=f'$\epsilon$ = %.3f' % log['epsilon'])
        elif eOrATag == 'a':
            plt.plot(n, G, label=f'α = %.3f' % log['alpha'])

    plt.grid()
    plt.legend(loc="upper left")
    plt.xlabel('Episode Number')
    plt.ylabel('G')
    log = logs[-1]
    if eOrATag == 'e':
        plt.title(f'Episode Return, α = %.3f' % log['alpha'])
    elif eOrATag == 'a':
        plt.title(f'Episode Return, $\epsilon$ = %.3f' % log['epsilon'])
    plt.savefig(os.path.join('.', 'figures', path, name))
    plt.close()
    return


def plot_example_traj_pendulum(pi_star, env, name, path):
    """
    Plots example trajectory of pendulum
    :param pi_star: optimal policy to run for epispode
    :param env: environment object
    :param name: figure save name
    :param path: figure save path
    :return: figure
    """
    # Initialize simulation
    s = env.reset()

    # Create log to store data from simulation
    log = {
        't': [0],
        's': [s],
        'a': [],
        'r': [],
        'theta': [env.x[0]],  # agent does not have access to this, but helpful for display
        'thetadot': [env.x[1]],  # agent does not have access to this, but helpful for display
    }

    # Simulate until episode is done
    done = False
    while not done:
        a = pi_star[s]
        (s, r, done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
        log['theta'].append(env.x[0])
        log['thetadot'].append(env.x[1])

    # Plot data and save to png file
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    ax[0].plot(log['t'], log['s'])
    ax[0].plot(log['t'][:-1], log['a'])
    ax[0].plot(log['t'][:-1], log['r'])
    ax[0].legend(['s', 'a', 'r'])
    ax[1].plot(log['t'], log['theta'])
    ax[1].plot(log['t'], log['thetadot'])
    ax[1].legend(['theta', 'thetadot'])
    plt.savefig(os.path.join('.', 'figures', path, name))
    plt.close()
    return


def plot_pendulum_policy(pi_star, name, path, n_theta, n_thetadot, log):
    """
    Plots pendulum policy as a contour in the theta_dot vs theta space
    :param pi_star: policy
    :param name: figure save name
    :param path: figure save path
    :param n_theta: number of thetas in discrete model
    :param n_thetadot: number of theta_dots for discrete model
    :param log: data on hyperparmeters
    :return: figure
    """
    eps = log['epsilon']
    alpha = log['alpha']

    # reshape policy into theta, theta_dot space
    pi_star = np.reshape(pi_star, (n_theta, n_thetadot))
    X,Y = np.meshgrid( np.arange(0, n_thetadot), np.arange(0, n_theta))

    fig, ax = plt.subplots(1,1)
    plt.grid()
    surf = ax.contourf(X, Y, pi_star)
    plt.ylabel('θ')
    plt.xlabel('$\dot{θ}$')
    plt.title('Pendulum Policy, $\epsilon$ = %.3f, α = %.3f' % (eps, alpha))
    cbar = fig.colorbar(surf)
    cbar.ax.set_ylabel('$\pi(s)$')

    plt.savefig(os.path.join('.', 'figures', path, name))
    return






