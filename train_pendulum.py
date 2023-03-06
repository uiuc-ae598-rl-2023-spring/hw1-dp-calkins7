import discrete_pendulum
from test_discrete_pendulum import test_x_to_s
from SARSA import sarsa
from Q_learning import q_learning
from TD_0 import TD_0
from plotters import *


def main():
    tag_SARSA_pendulum = True  # SARSA on pendulum
    tag_Q_pendulum = False  # Q-learning on pendulum
    tag_run_eps_alpha_grid = False  # generate plots for learning rate with varying eps and alpha

    if tag_SARSA_pendulum:
        n_theta = 15
        n_thetadot = 21
        env = discrete_pendulum.Pendulum(n_theta, n_thetadot) #, n_tau=10)
        env.reset()

        alpha = 0.5
        epsilon = 0.01

        num_episodes = 10000

        # Apply unit test to check state representation
        test_x_to_s(env)

        Q_star, pi_star, log = sarsa(env, alpha, epsilon, num_episodes)
        print(Q_star)
        print(pi_star)

        # Apply TD0 to learn state value function
        V_star = TD_0(pi_star, env, alpha, num_episodes=1000)

        # Plot return per number of episodes
        plot_return(log, 'SARSA_pend_return.png', 'pendulum')

        # Simulate episode to show example traj
        plot_example_traj_pendulum(pi_star, env, 'SARSA_pend_traj.png', 'pendulum')

        # Plot policies
        plot_pendulum_policy(pi_star, 'SARSA_pend_policy.png', 'pendulum', n_theta, n_thetadot, log)

        # Plot state value function
        log['V'] = V_star
        plot_state_val_func(log, 'SARSA_pend_state_val.png', 'pendulum')

        if tag_run_eps_alpha_grid:
            num_episodes = 5000
            # plot learning curve for varying eps
            epsVec = np.linspace(0.01, 1, 5)
            logs = []
            for eps in epsVec:
                Q_star, pi_star, log = sarsa(env, alpha, eps, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'SARSA_pend_vary_eps.png', 'pendulum', 'e')

            # plot learning curve for varying alpha
            alpVec = np.linspace(0.01, 1, 5)
            logs = []
            for alp in alpVec:
                Q_star, pi_star, log = sarsa(env, alp, epsilon, num_episodes)
                logs.append(log)
            plot_learning_curve(logs, 'SARSA_pend_vary_alp.png', 'pendulum', 'a')

    if tag_Q_pendulum:
        n_theta = 20
        n_thetadot = 26
        env = discrete_pendulum.Pendulum(n_theta, n_thetadot)

        # Apply unit test to check state representation
        test_x_to_s(env)

        alpha = 0.5
        epsilon = 0.01
        num_episodes = 5000
        #
        # env.reset()
        # Q_star, pi_star, log = q_learning(env, alpha, epsilon, num_episodes)
        # print(Q_star)
        # print(pi_star)
        #
        # # Apply TD0 to learn state value function
        # V_star = TD_0(pi_star, env, alpha, num_episodes=5000)
        #
        # # Plot return per number of episodes
        # plot_return(log, 'Q_pend_return.png', 'pendulum')
        #
        # # Simulate episode to show example traj
        # plot_example_traj_pendulum(pi_star, env, 'Q_pend_traj.png', 'pendulum')
        #
        # # Plot policies
        # plot_pendulum_policy(pi_star, 'Q_pend_policy.png', 'pendulum', n_theta, n_thetadot, log)
        #
        # # Plot state value function
        # log['V'] = V_star
        # plot_state_val_func(log, 'Q_pend_state_val.png', 'pendulum')

        if tag_run_eps_alpha_grid:
            # plot learning curve for varying eps
            epsVec = np.linspace(0.01, 1, 5)
            logs = []
            for eps in epsVec:
                Q_star, pi_star, log = q_learning(env, alpha, eps, num_episodes)
                logs.append(log)
                print(f'eps = {eps} complete')
            plot_learning_curve(logs, 'Q_pend_vary_eps.png', 'pendulum', 'e')

            # plot learning curve for varying alpha
            alpVec = np.linspace(0.01, 1, 5)
            logs = []
            for alp in alpVec:
                Q_star, pi_star, log = q_learning(env, alp, epsilon, num_episodes)
                logs.append(log)
                print(f'alpha = {alp} complete')
            plot_learning_curve(logs, 'Q_pend_vary_alp.png', 'pendulum', 'a')




if __name__ == '__main__':
    main()


