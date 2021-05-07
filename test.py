import gym
import d4rl
import numpy as np
from numpy.core.numeric import roll



env = gym.make("halfcheetah-expert-v1")

offline_data = d4rl.qlearning_dataset(env)

print(len(offline_data['observations']))
print(offline_data.keys())

rollouts = []
start_idx = 0
print(np.any(offline_data['terminals']))
exit()
for i in range(len(offline_data['observations'])):


    if offline_data['terminals'][i] == True:
        rollout = np.array(offline_data['observations'][start_idx:i+1])

        start_idx = i + 1
        rollouts.append(rollout)
        print(len(rollout))