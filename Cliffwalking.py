'''
Q-Learning and Sarsa in Cliffwalking
'''
from typing import OrderedDict
import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
# hyperparametres
step_size = 0.1
num_episodes = 5000
gamma = 0.9
env = gym.make('CliffWalking-v0').unwrapped

def max_q(Qtable,state):
    '''
    compare all actions if is in Qtable
    '''
    qlist = []
    if Qtable.get((state,0)) != None:
        q0 = Qtable[(state, 0)]
        qlist.append((q0, 0))

    if Qtable.get((state,1)) != None:
        q1 = Qtable[(state, 1)]
        qlist.append((q1, 1))

    if Qtable.get((state,2)) != None:
        q2 = Qtable[(state, 2)]
        qlist.append((q2, 2))

    if Qtable.get((state,3)) != None:
        q3 = Qtable[(state, 3)]
        qlist.append((q3, 3))

    return qlist

def Qlearning():
    Qtable = {}
    cumulative_reward_list = []
    for k in range(num_episodes):
        state = env.reset()
        not_end = True
        epsilon = 1/(k+1)
        cumulative_reward = 0
        while not_end:
            # epsilon-greedily choosing action
            qlist = max_q(Qtable, state)
            if not qlist:
                # qlist is empty
                # the state and action not in Qtable use random policy
                action = env.action_space.sample()
            else:   
                # explored state action
                qlist = sorted(qlist)
                # return the action of max value
                action_chosen = qlist[-1][1]
                probability = (epsilon/env.action_space.n)*np.ones(4)
                probability[action_chosen] += 1-epsilon
                action = np.random.choice(np.arange(4), p=probability)

            next_state, reward, done, _ = env.step(action)
            next_qlist = max_q(Qtable, next_state)
            cumulative_reward += reward
            # update Qtable
            if not next_qlist:
                # next_qlist is empty
                if Qtable.get((state,action)) == None:
                    Qtable[(state,action)] = step_size * reward
                else:
                    Qtable[(state, action)] += step_size*(reward-Qtable[(state, action)])
            else:
                next_qvalue = next_qlist[-1][0]
                if Qtable.get((state,action)) == None:
                    Qtable[(state,action)] = step_size *  (reward+ gamma*next_qvalue)
                else:
                    Qtable[(state, action)] += step_size*( reward+ gamma*next_qvalue - Qtable[(state, action)])            
            
            state = next_state
            if done:
                not_end = False
        if k%100 ==0:
            print('Q-Learning:[{}]/[{}]'.format(k, num_episodes))
        cumulative_reward_list.append(cumulative_reward)


    x_axis = np.arange(num_episodes)
    plt.plot(x_axis, np.array(cumulative_reward_list))
    plt.xlabel('num of episode')
    plt.ylabel('cumulative reward')
    plt.savefig('Q-Learning.png')
    plt.show()


    return Qtable

def Sarsa():
    Qtable = {}
    cumulative_reward_list = []
    for k in range(num_episodes):
        state = env.reset()
        not_end = True
        epsilon = 1/(k+1)
        cumulative_reward = 0
        step_counter = 0 # to avoid perambulating
        while not_end:
            step_counter += 1
            # epsilon-greedily choosing action
            qlist = max_q(Qtable, state)
            if not qlist:
                # qlist is empty
                # the state and action not in Qtable use random policy
                action = env.action_space.sample()
            else:   
                # explored state action
                qlist = sorted(qlist)
                # return the action of max value
                action_chosen = qlist[-1][1]
                probability = (epsilon/env.action_space.n)*np.ones(4)
                probability[action_chosen] += 1-epsilon
                action = np.random.choice(np.arange(4), p=probability)
            
            next_state, reward, done, _ = env.step(action)
            if done:
                not_end = False
            
            else:
                # epsilon-greedily choosing next_action in sarsa
                next_qlist = max_q(Qtable, next_state)
                if not next_qlist:
                    next_action = env.action_space.sample()
                    if Qtable.get((state, action)) == None:
                        Qtable[(state, action)] = step_size*(reward)
                    else:
                        Qtable[(state, action)] += step_size*(reward-Qtable[(state, action)]) 
                else:
                    next_qlist = sorted(next_qlist)
                    next_action_chosen = next_qlist[-1][1]
                    next_probability = (epsilon/env.action_space.n)*np.ones(4)
                    next_probability[next_action_chosen] += 1-epsilon
                    next_action = np.random.choice(np.arange(4), p=next_probability)
                
                
            
            # update Qtable
            if Qtable.get((next_state, next_action)) == None:
                Qtable[(next_state, next_action)] = 0
            
            if Qtable.get((state, action)) == None:
                Qtable[(state, action)] = 0
            

            
            Qtable[(state, action)] += step_size*(reward + gamma*Qtable[(next_state,next_action)]-Qtable[(state,action)])
                    
            
            cumulative_reward += reward
            
            # if step_counter>200:
            #     Qtable[(state, action)] += -50
            #     # print(step_counter)
            
            
            state = next_state
            
        if k%100 ==0:
            print('Sarsa:[{}]/[{}]'.format(k, num_episodes))
        cumulative_reward_list.append(cumulative_reward)

    print(cumulative_reward_list)
    x_axis = np.arange(num_episodes)
    plt.plot(x_axis, np.array(cumulative_reward_list))
    plt.xlabel('num of episode')
    plt.ylabel('cumulative reward')
    # plt.savefig('Sarsa.png')
    plt.show()
    return Qtable


def safer_Qlearning():
    Qtable = {}
    cumulative_reward_list = []
    for k in range(num_episodes):
        state = env.reset()
        not_end = True
        epsilon = 1/(k+1)
        cumulative_reward = 0
        while not_end:
            # epsilon-greedily choosing action
            qlist = max_q(Qtable, state)
            if not qlist:
                # qlist is empty
                # the state and action not in Qtable use random policy
                action = env.action_space.sample()
            else:   
                # explored state action
                qlist = sorted(qlist)
                # return the action of max value
                action_chosen = qlist[-1][1]
                probability = (epsilon/env.action_space.n)*np.ones(4)
                probability[action_chosen] += 1-epsilon
                action = np.random.choice(np.arange(4), p=probability)

            next_state, reward, done, _ = env.step(action)
            # change reward
            if action == 0 or action == 1:
                reward = 0
            next_qlist = max_q(Qtable, next_state)
            cumulative_reward += reward
            # update Qtable
            if not next_qlist:
                # next_qlist is empty
                if Qtable.get((state,action)) == None:
                    Qtable[(state,action)] = step_size * reward
                else:
                    Qtable[(state, action)] += step_size*(reward-Qtable[(state, action)])
            else:
                next_qvalue = next_qlist[-1][0]
                if Qtable.get((state,action)) == None:
                    Qtable[(state,action)] = step_size *  (reward+ gamma*next_qvalue)
                else:
                    Qtable[(state, action)] += step_size*( reward+ gamma*next_qvalue - Qtable[(state, action)])            
            
            state = next_state
            if done:
                not_end = False
        if k%100 ==0:
            print('safer Q-Learning:[{}]/[{}]'.format(k, num_episodes))
        cumulative_reward_list.append(cumulative_reward)


    x_axis = np.arange(num_episodes)
    plt.plot(x_axis, np.array(cumulative_reward_list))
    plt.xlabel('num of episode')
    plt.ylabel('cumulative reward')
    # plt.savefig('Q-Learning.png')
    plt.show()


    return Qtable

def show_path(Qtable):
    path = []
    epsilon =0
    state = env.reset()
    not_end = True
    while not_end:
        # epsilon-greedily choosing action
        path.append(state)
        qlist = max_q(Qtable, state)
        if not qlist:
            # qlist is empty
            # the state and action not in Qtable use random policy
            action = env.action_space.sample()
        else:   
            # explored state action
            qlist = sorted(qlist)
            # return the action of max value
            action_chosen = qlist[-1][1]
            probability = (epsilon/env.action_space.n)*np.ones(4)
            probability[action_chosen] += 1-epsilon
            action = np.random.choice(np.arange(4), p=probability)

        next_state, _, done, _ = env.step(action)
        state = next_state
        if done:
            not_end = False
    return path


if __name__ == "__main__":
    Q1table = Qlearning()
    # Qtable = Sarsa()
    # Q2table = safer_Qlearning()
    path1 = show_path(Q1table)
    print(path1)
    

