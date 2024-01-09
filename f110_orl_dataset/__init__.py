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