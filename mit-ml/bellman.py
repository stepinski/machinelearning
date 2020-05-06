import numpy as np
V = np.array([0,0,0,0,0])
reward = np.array([0,0,0,0,1])
gamma = 1/2
left = np.array([[0.5,0.5,0,0,0],[1/3,2/3,0,0,0],[0,1/3,2/3,0,0],[0,0,1/3,2/3,0],[0,0,0,1/3,2/3]])
right = np.array([[2/3,1/3,0,0,0],[0,2/3,1/3,0,0],[0,0,2/3,1/3,0],[0,0,0,2/3,1/3],[0,0,0,0.5,0.5]])
stay = np.array([[0.5,0.5,0,0,0],[0.25,0.5,0.25,0,0],[0,0.25,0.5,0.25,0],[0,0,0.25,0.5,0.25],[0,0,0,0.5,0.5]])
for i in range(0,30):
    left_action = left@(gamma*V+reward)
    right_action = right@(gamma*V+reward)
    stay_action = stay@(gamma*V+reward)
    transition_choices = np.stack([left_action,right_action,stay_action])
    V = np.max(transition_choices, axis=0)
print(V)