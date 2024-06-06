import gymnasium as gym
import numpy as np
import random
from gymnasium import spaces
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO

from sb3_callbacks import SaveOnBestTrainingRewardCallback


class GPUScheduler(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(19)
        self.observation_space = spaces.Dict(
            {
                'mig_status': spaces.Box(low=0, high=6, shape=(80, 5)),
                'required_gpu': spaces.Discrete(5)
            },
            seed=0
        )
        self.available_gpus = []

    def step(self, action):
        mig_config = (
            [0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 1, 0],
            [3, 0, 0, 1, 0],
            [0, 0, 2, 0, 0],
            [1, 1, 1, 0, 0],
            [3, 0, 1, 0, 0],
            [0, 2, 1, 0, 0],
            [2, 1, 1, 0, 0],
            [2, 1, 1, 0, 0],
            [4, 0, 1, 0, 0],
            [1, 3, 0, 0, 0],
            [3, 2, 0, 0, 0],
            [3, 2, 0, 0, 0],
            [5, 1, 0, 0, 0],
            [5, 1, 0, 0, 0],
            [5, 1, 0, 0, 0],
            [5, 1, 0, 0, 0],
            [6, 0, 0, 0, 0],
        )

        observation = self.available_gpus
        user_group = []
        mig_change = False
        while not mig_change:
            req_type = random.sample(['acquire', 'release'], 1)[0]
            if user_group and req_type == 'release':
                user_req = user_group.pop(random.randint(0, len(user_group) - 1))
                observation[user_req['gpu_id']][user_req['gpu_type']] += 1

            else:
                user_got_gpus = [x['user_id'] for x in user_group]
                user_id = random.sample([x for x in range(100) if x not in user_got_gpus], 1)[0]
                gpu_type = random.randint(0, 4)

                mig_change = True
                for gpu_id, available_gpu in enumerate(observation):
                    if available_gpu[gpu_type] != 0:
                        observation[gpu_id][gpu_type] -= 1
                        user_group.append({'user_id': user_id, 'gpu_id': gpu_id, 'gpu_type': gpu_type})
                        mig_change = False
                        break

        unuse_gpus = []
        if user_group:
            using_gpus = list(set([x['gpu_id'] for x in user_group]))
            unuse_gpus = [x for x in range(len(observation)) if x not in using_gpus]
        if unuse_gpus:
            gpu_id = unuse_gpus[0]
            observation[gpu_id] = mig_config[action]
            rewards = 10
        else:
            gpu_id = len(observation)
            observation.append(mig_config[action])
            rewards = -40 if mig_config[action][gpu_type] == 0 else -10
        rewards += len(set(user_got_gpus))
        observation[gpu_id][gpu_type] -= 1

        self.available_gpus = observation

        terminated = False
        if len(observation) > 50:
            terminated = True

        observation = np.concatenate((np.array(observation), np.zeros((80 - len(observation), 5))), axis=0)
        observation = {
            'mig_status': observation,
            'required_gpu': gpu_type
        }
        return observation, rewards, terminated, None, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.available_gpus = []
        init_ag = np.zeros((80, 5))
        gpu_type = random.randint(0, 4)

        observation = {
            'mig_status': np.array(init_ag),
            'required_gpu': gpu_type
        }
        return observation, []


log_dir = 'logs'

env = GPUScheduler()

env = Monitor(env, log_dir)
model = PPO("MultiInputPolicy", env, verbose=1)

callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir='logs')
model.learn(total_timesteps=30_000, callback=callback)

env = model.get_env()
obs = env.reset()
for i in range(1000):
    act, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(act)
    if done:
        obs = env.reset()
