
import numpy as np
stored_histories = []       # store position = f(t) every so often for an episode
Σ = np.zeros(Ne)            # learning gain per episode

naive = Swimmer()
smart = Swimmer()

def training(stored_histories, Σ, smart, naive):
    Q = L*Ns*np.ones((12, 4))   # 12 states and 4 possible actions. Each column is an action, ka: up, down, right, left
    prev_Q = Q

    for ep in tqdm(range(Ne)):  # for each episode

        # total long-term vertical displacement, one for each realization
        R_tots_naive = np.zeros(N_ensemble)
        R_tots_smart = np.zeros(N_ensemble)

        # initialize Q with its final value from the last episode
        Q = prev_Q

        # iterate over different realizations of noise and initial conditions
        for realization in range(N_ensemble):

            # assign random orientation and position 
            naive.reinitialize()
            smart.reinitialize()

            # iterate over stages within an episode
            for stage in range(Ns): 
                # select an action eps-greedily. Note naive never changes its action/strategy (i.e. trying to swim upward)
                if np.random.uniform(0, 1) < eps:
                    action_index = smart.take_random_action()
                else:
                    action_index = smart.take_greedy_action(Q)

                # record prior state
                old_state_index = state_lookup_table[smart.my_state]

                # given selected action, update the state
                naive.update_kinematics()
                smart.update_kinematics()
                smart.update_state()      # only need to update smart particle since naive has ka = [0, 1]

                # calculate reward based on new state
                naive.calc_reward(stage)
                smart.calc_reward(stage)

                # update Q matrix 
                state_index = state_lookup_table[smart.my_state]
                Q[old_state_index, action_index] += alpha*(smart.r[stage] + \
                        gamma*(np.max(Q[state_index,:])-np.max(Q[old_state_index,:])))

            # calculate Rtot for this realization
            R_tot_naive = np.mean(naive.r)
            R_tot_smart = np.mean(smart.r)

            # collect Rtot across realizations
            R_tots_naive[realization] = R_tot_naive
            R_tots_smart[realization] = R_tot_smart

        # TODO: add warning based on average initial position and orientation vanishing 
    #     if abs(signaltonoise(R_tots_naive))<1 or abs(signaltonoise(R_tots_smart))<1:
    #       raise Exception(("Signal to nosie ratios % 5.2f and % 5.2f are too small. " + \
    #             " Consider increasing ensemble size.") %(signaltonoise(R_tots_naive), signaltonoise(R_tots_smart)))

        # store final Q for this episode
        prev_Q = Q

        # calculate ensemble average of total gain for this episode
        avg_R_tot_naive = np.mean(R_tots_naive)
        avg_R_tot_smart = np.mean(R_tots_smart)

        # calculate learning gain for this episode
        Σ[ep] = avg_R_tot_smart/avg_R_tot_naive - 1

        # plot trajectory every so often 
        if stage%n_updates==0:
            history_X_total = np.array(smart.history_X_total)
            stored_histories.append((ep,history_X_total))
    
    return Q

Q = training(stored_histories, Σ, smart, naive)
