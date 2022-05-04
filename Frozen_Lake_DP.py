'''
Project 1 is aimed to use policy iteration 
to solve the problem of FrozenLake
env.P[state][action] knowledge needed
'''
import gym
import numpy as np

env = gym.make('FrozenLake-v1')
gamma = 1
epsilon = 1e-6

# for state_probability, next_state, reward, terminated in env.P[state][action]:
#     print(state_probability, next_state, reward, terminated)



def init():
    value_mat = np.zeros(16)
    policy_mat = (1/4)*np.ones((16, 4))
    return value_mat, policy_mat

def policy_evaluation(value_mat, policy_mat):
    '''
    use Bellman equation to update value_mat
    '''
    new_value_list = []
    for state in range(16):
        state_policy = policy_mat[state]
        new_value = 0
        for action in range(4):
            # calculate action value
            action_value = 0
            for state_probability, next_state, reward, terminated in env.P[state][action]:
                action_value += state_probability * (reward + gamma * value_mat[next_state])
            
            new_value += state_policy[action]*action_value    
        
        new_value_list.append(new_value) 

        new_value_mat = np.array(new_value_list)        
    return new_value_mat

def policy_improvement(value_mat, policy_mat):
    '''
    to update policy_mat greedily
    '''
    new_policy_mat = np.zeros((16,4))

    for state in range(16):
        # calculate action value for every action
        action_value_list = []
        for action in range(4):
            action_value = 0
            for state_probability, next_state, reward, terminated in env.P[state][action]:
                action_value += state_probability * (reward + gamma * value_mat[next_state])

            action_value_list.append(action_value)

        # return the best action
        action_value_mat = np.array(action_value_list)
        index = np.argmax(action_value_mat)
        new_policy_mat[state][index] = 1

    return new_policy_mat



def get_policy():
    value_mat, policy_mat = init()

    policy_not_stable = True
    # main loop
    while policy_not_stable:

        # policy evaluatin loop
        max_difference = 1
        while max_difference > epsilon:
            new_value_mat = policy_evaluation(value_mat, policy_mat)
            max_difference = np.max(np.abs(new_value_mat - value_mat))
            value_mat = new_value_mat
            print(value_mat)
        
        # policy improvement
        new_policy_mat = policy_improvement(value_mat, policy_mat)
        if (new_policy_mat == policy_mat).all():
            policy_not_stable = False
        else:
            policy_mat = new_policy_mat
            print(new_policy_mat)

    return policy_mat
    



if __name__ == "__main__":
    policy_mat = get_policy()
    print('final policy mat',policy_mat)
    observation = 0
    env.reset()
    for _ in range(1000):
        env.render() # visualization
        observation, reward, done, info = env.step(np.argmax(policy_mat[observation])) 
        
    
    env.close()
     
        