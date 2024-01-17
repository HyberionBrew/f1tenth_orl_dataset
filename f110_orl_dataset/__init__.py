from gymnasium.envs.registration import register
from .dataset_env import F1tenthDatasetEnv

# This will override the existing registration#
register(
    id='f110-real-v0',
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/dataset_real_v0.zip"}
)

register(
    id='f110-sim-v0',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/dataset_sim_v0.zip"}
    # kwargs={'param1': value1, 'param2': value2, ...}
)

register(
    id='f110-real-v1',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/f110-real-v1.zip",
              'bad_trajectories': [7028, 9789,13554, 22841,25100,59487,60742,67519,72348,74356,7781, 10040]} # one or the other recording problem 
    # kwargs={'param1': value1, 'param2': value2, ...}
)

register(
    id='f110-sim-v1',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/f110-sim-v1.zip",}
              #'bad_trajectories': [7028, 9789,13554, 22841,25100,59487,60742,67519,72348,74356,7781, 10040]} # one or the other recording problem 
    # kwargs={'param1': value1, 'param2': value2, ...}
)