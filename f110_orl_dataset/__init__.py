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

register(
    id='f110-real-stoch-v1',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/f110-sim-v1.zip",
              'bad_trajectories': [ 72539, 84587, 87097, 109185, 138050, 49196, 49447,
    151604, 112950, 114456, 115962, 116213, 116464],}  # manually filtered bad trajectories
              #'bad_trajectories': [7028, 9789,13554, 22841,25100,59487,60742,67519,72348,74356,7781, 10040]} # one or the other recording problem 
    # kwargs={'param1': value1, 'param2': value2, ...}
)

register(
    id='f110-real-stoch-v2',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/f110-sim-v1.zip",
              'bad_trajectories': [161444,283691, 82322, 
                                   202835, 
                                   152551, 268314, 155685, 161312,
                                    278488], # 166966
              'eval_model_names': ["pure_pursuit2_0.6_1.0_raceline2_0.3_0.5",
                                    "pure_pursuit2_0.7_0.9_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.8_0.95_raceline3_0.3_0.5",
                                    # "pure_pursuit2_0.8_1.25_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.44_0.85_raceline1_0.3_0.5",
                                    "pure_pursuit2_0.52_0.9_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.65_1.2_centerline_0.3_0.5",
                                    "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_centerline_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_raceline4_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.8_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                  #   "StochasticContinousFTGAgent_1.0_0_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_1.0_1_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "pure_pursuit2_0.4_0.3_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.44_0.3_raceline1_0.3_0.5"],
              'bad_agents': ["pure_pursuit2_0.8_1.25_raceline8_0.3_0.5", "StochasticContinousFTGAgent_0.4_5_0.5_0.03_0.1_5.0_0.3_0.5"]
              }  # manually filtered bad trajectories
              #'bad_trajectories': [7028, 9789,13554, 22841,25100,59487,60742,67519,72348,74356,7781, 10040]} # one or the other recording problem 
    # kwargs={'param1': value1, 'param2': value2, ...}
)

register(
    id='f110-sim-stoch-v2',  
    entry_point='f110_orl_dataset.dataset_env:F1tenthDatasetEnv',
    kwargs = {'dataset_url': "https://github.com/HyberionBrew/f110_datasets/raw/main/f110-sim-v1.zip",
              'bad_trajectories': [], # 166966
              'eval_model_names': ["pure_pursuit2_0.6_1.0_raceline2_0.3_0.5",
                                    "pure_pursuit2_0.7_0.9_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.8_0.95_raceline3_0.3_0.5",
                                    # "pure_pursuit2_0.8_1.25_raceline8_0.3_0.5", # only has 49 trajectories :/
                                    "pure_pursuit2_0.44_0.85_raceline1_0.3_0.5",
                                    "pure_pursuit2_0.52_0.9_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.65_1.2_centerline_0.3_0.5",
                                    "pure_pursuit2_0.68_1.1_raceline8_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_centerline_0.3_0.5",
                                    "pure_pursuit2_0.73_0.95_raceline4_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.5_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.6_2_0.8_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_0.8_2_0.7_0.03_0.1_5.0_0.3_0.5",
                                  #   "StochasticContinousFTGAgent_1.0_0_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "StochasticContinousFTGAgent_1.0_1_0.2_0.03_0.1_5.0_0.3_0.5",
                                    "pure_pursuit2_0.4_0.3_raceline4_0.3_0.5",
                                    "pure_pursuit2_0.44_0.3_raceline1_0.3_0.5"]
              }  # manually filtered bad trajectories
              #'bad_trajectories': [7028, 9789,13554, 22841,25100,59487,60742,67519,72348,74356,7781, 10040]} # one or the other recording problem 
    # kwargs={'param1': value1, 'param2': value2, ...}
)