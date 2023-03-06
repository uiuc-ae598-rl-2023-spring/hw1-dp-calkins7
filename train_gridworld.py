import gridworld
from policy_iteration import poli_iter
from value_iteration import valu_iter
from SARSA import sarsa
from Q_learning import q_learning
from TD_0 import TD_0
from plotters import *


def main():
    tag_PI = False  # Policy Iteration
    tag_VI = False  # Value Iteration
    tag_SARSA_gridworld = False  # SARSA on gridworld
    tag_Q_gridworld = True  # Q-learning on gridworld
    tag_run_eps_alpha_grid = True  # generate plots for learning rate with varying eps and alpha


    if tag_PI:
        env = gridworld.GridWorld(hard_version=False)
        env.reset()
        V_star, pi_star, log = poli_iter(env)
        print(V_star)
        print(pi_star)

        # Plot mean value function for number of policy evals
        plot_mean_value_func(log, 'PI_mean_value_func.png', 'gridworld')

        # Simulate episode to show example traj
        plot_example_traj_grid(pi_star, env, 'PI_traj.png')

        # Plot policy
        plot_gridworld_policy(pi_star, 'PI_policy.png')

        # Plot state value function
        plot_state_val_func(log, 'PI_state_val.png', 'gridworld')

    if tag_VI:
        env = gridworld.GridWorld(hard_version=False)
        env.reset()
        V_star, pi_star, log = valu_iter(env, tol=1e-16)
        print(V_star)
        print(pi_star)

        # Plot mean value function for number of policy evals
        plot_mean_value_func(log, 'VI_mean_value_func.png', 'gridworld')

        # Simulate episode to show example traj
        plot_example_traj_grid(pi_star, env, 'VI_traj.png')

        # Plot policies
        plot_gridworld_policy(pi_star, 'VI_policy.png')

        # Plot state value function
        plot_state_val_func(log, 'VI_state_val.png', 'gridworld')

    if tag_SARSA_gridworld:
        env = gridworld.GridWorld(hard_version=False)
        env.reset()

        alpha = 0.5
        epsilon = 0.1
        num_episodes = 10000

        Q_star, pi_star, log = sarsa(env, alpha, epsilon, num_episodes)
        print(Q_star)
        print(pi_star)

        # Apply TD0 to learn state value function
        V_star = TD_0(pi_star, env, alpha, num_episodes=1000)
        print(V_star)

        # Plot return per number of episodes
        plot_return(log, 'SARSA_grid_return.png', 'gridworld')

        # Simulate episode to show example traj
        plot_example_traj_grid(pi_star, env, 'SARSA_grid_traj.png', modelFreeTag=True, logPass=log)

        # Plot policies
        plot_gridworld_policy(pi_star, 'SARSA_grid_policy.png', modelFreeTag=True, logPass=log)

        # Plot state value function
        log['V'] = V_star
        plot_state_val_func(log, 'SARSA_grid_state_val.png', 'gridworld', modelFreeTag=True, logPass=log)

        num_episodes = 1000
        if tag_run_eps_alpha_grid:
            # plot learning curve for varying eps
            alpha = 0.5
            epsVec = np.linspace(0.01, 1, 5)
            logs = []
            for eps in epsVec:
                Q_star, pi_star, log = sarsa(env, alpha, eps, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'SARSA_grid_vary_eps.png', 'gridworld', 'e')

            # plot learning curve for varying alpha
            epsilon = 0.1
            alpVec = np.linspace(0.01, 1, 5)
            logs = []
            for alp in alpVec:
                Q_star, pi_star, log = sarsa(env, alp, epsilon, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'SARSA_grid_vary_alp.png', 'gridworld', 'a')


    if tag_Q_gridworld:
        env = gridworld.GridWorld(hard_version=False)
        env.reset()

        alpha = 0.5
        epsilon = 0.2
        num_episodes = 10000

        Q_star, pi_star, log = q_learning(env, alpha, epsilon, num_episodes)
        print(Q_star)
        print(pi_star)

        # Apply TD0 to learn state value function
        V_star = TD_0(pi_star, env, alpha, num_episodes)

        # Plot return per number of episodes
        plot_return(log, 'Q_grid_return.png', 'gridworld')

        # Simulate episode to show example traj
        plot_example_traj_grid(pi_star, env, 'Q_grid_traj.png', modelFreeTag=True, logPass=log)

        # Plot policies
        plot_gridworld_policy(pi_star, 'Q_grid_policy.png', modelFreeTag=True, logPass=log)

        # Plot state value function
        log['V'] = V_star
        plot_state_val_func(log, 'Q_grid_state_val.png', 'gridworld', modelFreeTag=True, logPass=log)

        num_episodes = 1000
        if tag_run_eps_alpha_grid:
            # plot learning curve for varying eps
            epsVec = np.linspace(0.01, 1, 5)
            logs = []
            for eps in epsVec:
                Q_star, pi_star, log = sarsa(env, alpha, eps, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'Q_grid_vary_eps.png', 'gridworld', 'e')

            # plot learning curve for varying alpha
            alpVec = np.linspace(0.01, 1, 5)
            logs = []
            for alp in alpVec:
                Q_star, pi_star, log = sarsa(env, alp, epsilon, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'Q_grid_vary_alp.png', 'gridworld', 'a')


if __name__ == '__main__':
    main()
