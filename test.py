import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

F110Env = gym.make('f110-real-v1',
                   encode_cyclic=True,
                   flatten_obs=True,
                   timesteps_to_include=(0,250),
                    use_delta_actions=False, # control if actions are deltas or absolute
                   reward_config="reward_progress.json",
        **dict(name='f110-real-v1',
            config = dict(map="Infsaal2", num_agents=1),
              render_mode="human")
    ) 

#print(F110Env.observation_space_orig)
ds = F110Env.get_dataset(
    #only_agents=["StochasticContinousFTGAgent_0.15_5_0.2_0.15_2.0"],)
)

print(ds["actions"][:40])
print(ds["model_name"][:10])
plt.plot(ds["observations"][:1000,4:7])
plt.plot(ds["timeouts"][:1000])
plt.plot(ds["actions"][:1000])
print(F110Env.keys)
plt.legend(["ang_vels_z", "linear_vels_x", "linear_vels_y", "timeout","raw_action_steering", "raw_action_speed"])
plt.show()