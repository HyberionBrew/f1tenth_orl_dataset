import f110_gym
import f110_orl_dataset
import gymnasium as gym
import numpy as np

# loads both datasets with the settings of the F110 OPE benchmark (downloads them on first use)
for name, reward_config in [("f110-real-stoch-v2", "reward_progress.json"), ("f110-sim-stoch-v2", None)]:
    F110Env = gym.make(name,
                       encode_cyclic=True,
                       flatten_obs=True,
                       timesteps_to_include=(0,250),
                       use_delta_actions=True, # control if actions are deltas or absolute
                       set_terminals=True,
                       reward_config=reward_config, # f110-sim-stoch-v2 has no new_rewards, see relabel_reward.py
                       include_time_obs=True,
                       include_progress=False,
                       use_compute_termination=True,
                       remove_cons_terminals=True,
        **dict(name=name,
            config = dict(map="Infsaal3", num_agents=1, params=dict(vmin=0.0, vmax=2.0)),
              render_mode="human")
    )
    ds = F110Env.get_dataset()
    finished = ds["terminals"] | ds["timeouts"]
    print(name)
    print("  observation keys:", F110Env.keys)
    print("  rows:", len(ds["observations"]), "episodes:", int(finished.sum()))
    print("  agents:", len(np.unique(ds["model_name"])), "of which OPE target policies:", len(F110Env.eval_agents))
    print("  actions", ds["actions"].shape, "log_probs", ds["log_probs"].shape, "rewards", ds["rewards"].shape)
