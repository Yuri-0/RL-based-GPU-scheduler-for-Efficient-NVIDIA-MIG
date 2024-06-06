from stable_baselines3 import PPO
from scipy.spatial.distance import euclidean
from typing import Union
import numpy as np

available_gpus, user_group = [], []
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
req_gpus_config = {
    '1g': 0,
    '2g': 1,
    '3g': 2,
    '4g': 3,
    '7g': 4
}

model = PPO.load('logs/best_model/best_model.zip')


def predictor(obs: list, req_gpus: int) -> np.ndarray:
    obs = {
        'mig_status': np.concatenate((np.array(obs), np.zeros((80 - len(obs), 5))), axis=0),
        'required_gpu': req_gpus
    }

    act, _state = model.predict(obs, deterministic=True)

    return act


def allocator(req_type: str, user_id: str, req_gpus: Union[str | None] = None) -> Union[[int, int] | None]:
    user_id = int(user_id.split('-')[-1])
    if req_gpus is not None:
        req_gpus = req_gpus_config[req_gpus]
    if req_type == 'acquire':
        for gpu_id, available_gpu in enumerate(available_gpus):
            if available_gpu[req_gpus] != 0:
                available_gpus[gpu_id][req_gpus] -= 1
                user_group.append({'user_id': user_id, 'gpu_id': gpu_id, 'gpu_type': req_gpus})

                return gpu_id, mig_config.index(available_gpu) + 1

        mig_no = predictor(available_gpus if available_gpus else [[0 for _ in range(5)]], req_gpus)

        unuse_gpus = []
        if mig_config[mig_no][req_gpus] != 0:
            if user_group:
                using_gpus = list(set([x['gpu_id'] for x in user_group]))
                unuse_gpus = [x for x in range(len(available_gpus)) if x not in using_gpus]
        else:
            distances = [[x, euclidean(mig_config[mig_no], xx)] for x, xx in enumerate(mig_config) if xx[req_gpus] != 0]
            mig_no = sorted(distances, key=lambda x: x[1])[0][0]
        if unuse_gpus:
            gpu_id = unuse_gpus[0]
            available_gpus[gpu_id] = mig_config[mig_no]
        else:
            gpu_id = len(available_gpus)
            available_gpus.append(mig_config[mig_no])
        available_gpus[gpu_id][req_gpus] -= 1

        return gpu_id, mig_no + 1

    else:
        for idx, user in enumerate(user_group):
            if user['user_id'] == user_id:
                available_gpus[user['gpu_id']][user['gpu_type']] += 1
                user_group.remove(idx)
