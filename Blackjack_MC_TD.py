'''
Monte Carlo  and Temporal-Di↵erence in blackjack
Requirements:
1. First visit monte carlo - on-policy
   # reference to Sutton and Barto on-policy first-visit MC control
2. TD(1) - off-policy
'''
import gym
import numpy as np
import matplotlib.pyplot as plt
# hyperperameters
original_epsilon = 1 # epsilon-greedy for both methods it decays
k = 1  # the counter for episodes in order to decay epsilon
num_episodes = 10000
step_size = 0.1
gamma = 0.9
num_tests = 5000 # required to calculate win rate
env = gym.make('Blackjack-v1').unwrapped




def generate_episode(Qtable, epsilon):
    '''
    use policy under epsilon-policy of current Qtable 
    to generate episode
    observation or state is a tuple of 
    (player_sum, dealer_sum, useable_ace) 
    '''
    state = env.reset()
    episode = []
    not_end = True
    # to follow the policy of epsilon greedy
    q0 = 0
    q1 = 0
    while not_end:
        if Qtable.get((state, 0)) != None:
            q0 = Qtable[(state,0)]
        if Qtable.get((state, 1)) != None:
            q1 = Qtable[(state,1)]
        
        if q0 == q1:
            action = np.random.choice(np.arange(2))
        elif q0 < q1:
            probability = np.array([epsilon/2, 1-epsilon/2])
            action = np.random.choice(np.arange(2), p=probability)
        else:
            probability = np.array([1-epsilon/2, epsilon/2])
            action = np.random.choice(np.arange(2), p=probability)
        next_state, reward, done, _ = env.step(action) 
        episode.append((state, action, reward))
        state = next_state
        if done:
            not_end = False

    return episode


def MC():
    # initialize all tabels
    Qtable = {}
    N = {}
    Sum_all_episodes = {}
    for k in range(num_episodes):
        # epsilon = 1/(k+1) # epsilon in lecture 5
        epsilon = ((num_episodes-k)/num_episodes) # linear decreasing epsilon
        episode = generate_episode(Qtable, epsilon)
        explored_sa_pair = []
        for i,state_action_reward in enumerate(episode):
            state, action, _ = state_action_reward
            last_reward = episode[-1][-1]
            horizon_to_end = len(episode) - (i+1)
            return_of_state = last_reward*(gamma**horizon_to_end)
            
            # count first visit of every episode
            if (state, action) in explored_sa_pair:
                continue
            else:
                explored_sa_pair.append((state, action))

                # update tables
                # Select state,action to avoid sending Error message
                if Sum_all_episodes.get((state,action)) == None:
                    Sum_all_episodes[(state,action)] = return_of_state
                else:
                    Sum_all_episodes[(state,action)] += return_of_state
                
                if N.get((state,action)) == None:
                    N[(state,action)] = 1
                else:
                    N[(state,action)] += 1

                
                Qtable[(state,action)] = Sum_all_episodes[(state,action)]/N[(state,action)]
    return Qtable


def TD():
      # initialize all tabels
    Qtable = {}
    for k in range(num_episodes):
        state = env.reset()
        not_end = True
        epsilon = 1/(k+1)
        
        while not_end:
            # epsilon greedy
            q0 = 0
            q1 = 0
            if Qtable.get((state, 0)) != None:
                q0 = Qtable[(state,0)]
            if Qtable.get((state, 1)) != None:
                q1 = Qtable[(state,1)]
            
            if q0 == q1:
                action = np.random.choice(np.arange(2))
            elif q0 < q1:
                probability = np.array([epsilon/2, 1-epsilon/2])
                action = np.random.choice(np.arange(2), p=probability)
            else:
                probability = np.array([1-epsilon/2, epsilon/2])
                action = np.random.choice(np.arange(2), p=probability)
            
            next_state, reward, done, _ = env.step(action)

            # next actio n
            q0 = 0
            q1 = 0
            if Qtable.get((next_state, 0)) != None:
                q0 = Qtable[(next_state,0)]
            if Qtable.get((next_state, 1)) != None:
                q1 = Qtable[(next_state,1)]
            
            if q0 == q1:
                next_action = np.random.choice(np.arange(2))
            elif q0 < q1:
                next_action = 1
            else:
                next_action = 0



            if Qtable.get((next_state, next_action)) == None:
                Qtable[(next_state, next_action)] = 0

            if Qtable.get((state, action)) == None:
                Qtable[(state, action)] = 0
            
            Qtable[(state, action)] += step_size*(reward + gamma*Qtable[(next_state, next_action)]-Qtable[(state, action)]) 
            
            state = next_state
            if done:
                not_end = False
            
            
        if k%100 ==0:
            print('TD:[{}]/[{}]'.format(k, num_episodes))
    return Qtable




def test(Qtable):
    win_counter = 0
    for i in range(num_tests):
        state = env.reset()
        q0 = 0
        q1 = 0
        not_end = True
        while not_end:
            if Qtable.get((state, 0)) != None:
                q0 = Qtable[(state,0)]
            if Qtable.get((state, 1)) != None:
                q1 = Qtable[(state,1)]
            
            if q0 == q1:
                action = np.random.choice(np.arange(2))
            elif q0 < q1:
                action = 1
            else:
                action = 0
            next_state, reward, done, _ = env.step(action) 
            
            if reward == 1:
                win_counter+=1
            
            state = next_state
            if done:
                not_end = False

    print(win_counter/num_tests)
    return win_counter/num_tests








if __name__=="__main__":
    xlist = []
    ylist = []
    for i in range(100):
        num_episodes = i * 100
        # policy = TD()
        policy = MC()
        winrate = test(policy)
        xlist.append(i*100)
        ylist.append(winrate)



    x_axis = np.array(xlist)
    y_axis = np.array(ylist)
    plt.plot(x_axis, ylist)
    plt.xlabel('num of episode')
    plt.ylabel('win_rate')
    # plt.savefig('TD.png')
    plt.savefig('MC.png')
    plt.show()