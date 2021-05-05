def qLearning(env, num_episodes, discount_factor = 0.9,
                                alpha = 0.1, epsilon = 0.1):
      
        
        
        
       
    def createPolicy(Q, epsilon, num_actions):     
        """
        Creates an epsilon-greedy policy based
        on a given Q-function and epsilon.

        Returns a function that takes the state
        as an input and returns the probabilities
        for each action in the form of a numpy array 
        of length of the action space(set of possible actions).
        """
        # import library
        import numpy as np    
        def policyFunction(state):
            Action_probabilities = np.ones(num_actions,
                    dtype = float) * epsilon / num_actions
            best_action = np.argmax(Q[state])
            Action_probabilities[best_action] += (1.0 - epsilon)
            return Action_probabilities
        return policyFunction




    """
    Q-Learning algorithm: Off-policy TD control.
    Finds the optimal greedy policy while improving
    following an epsilon-greedy policy
    """
    # import libraries
    from collections import defaultdict
    import itertools
    import numpy as np
    from numpy import random
    
    # Action value function
    # A nested dictionary that maps
    # state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_n))

    # Keeps track of useful statistics
    length = []
    rewards = []

    # Create an epsilon greedy policy function
    # appropriately for environment action space
    policy = createPolicy(Q, epsilon, env.action_n)

    # For every episode
    for ith_episode in range(num_episodes):

        # Reset the environment and pick the first action
        state = env.reset(int(ith_episode))
        rew = 0   
        for t in itertools.count():
            
            # random or not
            if env.random_learning:
                np.random.seed(t)

            # get probabilities of all actions from current state
            action_probabilities = policy(state)

            # choose action according to 
            # the probability distribution
            action = np.random.choice(np.arange(len(action_probabilities)),
                       p = action_probabilities)

            # take action and get reward, transit to next state
            next_state, reward, done = env.step(action)

            # Update reward
            rew += reward

            # TD Update
            best_next_action = np.argmax(Q[next_state])    
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            td_delta = td_target - Q[state][action]
            Q[state][action] += alpha * td_delta

            # done is True if episode terminated   
            if done:
                length.append(t)
                rewards.append(rew)
                break

            state = next_state
    stats = [length,rewards]

    return Q,length,rewards