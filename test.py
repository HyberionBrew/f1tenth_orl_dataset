import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

F110Env = gym.make('f110-real-v0',
                   encode_cyclic=True,
                   flatten_obs=True,
                   timesteps_to_include=(0,250),
                   reward_config="reward_raceline.json",
        **dict(name='f110-real-v0',
            config = dict(map="Infsaal2", num_agents=1),
              render_mode="human")
    ) 

#print(F110Env.observation_space_orig)
ds = F110Env.get_dataset(#zarr_path="/home/fabian/msc/f110_dope/ws_release/dataset_real_2312.zarr",
    only_agents=["StochasticContinousFTGAgent_0.15_5_0.2_0.15_2.0"],)


print(ds["infos"]["obs_keys"])
print(ds["observations"])
print(ds["actions"][:5])
#print(ds["rewards"][:250])
#print(np.count_nonzero(ds["rewards"]))
#print(ds["infos"]["lidar_timestamp"][:5])