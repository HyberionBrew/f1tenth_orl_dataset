# F1tenth offline RL / OPE datasets

Datasets of a 1:10 scale F1tenth car driving on the `Infsaal3` track, recorded in the real world and re-run in simulation. These are the datasets used by the [F110 OPE benchmark](https://github.com/HyberionBrew/f110_ope_benchmark).

# How to install
Tested with conda and python=3.8.

First, you will need an adapted version of the f1tenth-gym.

checkout the v1.0.0 branch:

```
git clone https://github.com/HyberionBrew/f1tenth_gym.git
cd f1tenth_gym
git checkout v1.0.0

```

install it (`yamldataclassconfig>=2` breaks the track loading of the gym):

```
cd ..
pip install "yamldataclassconfig==1.5.0"
pip install -e f1tenth_gym
```
Install this package
```
git clone https://github.com/HyberionBrew/f1tenth_orl_dataset.git
pip install -e f1tenth_orl_dataset
```
Run a small test inside the f1tenth_orl_dataset folder, this downloads both datasets to `~/.f110_rl_datasets` and loads them:

```
python test.py
```

# Explanation of the datasets
The datasets are recorded with a control frequency of 20 Hz. For the real world dataset lidar and pose data are also available with roughly this frequency. Episodes are at most 251 timesteps long.

- f110-real-stoch-v2: real-world data of 57 stochastic agents (follow-the-gap and pure pursuit, see [stochastic_ftg_agents](https://github.com/HyberionBrew/stochastic_ftg_agents/tree/ope-benchmark), branch `ope-benchmark`). The 15 agents in `eval_model_names` (50-90 trajectories each) are the OPE target policies, all others form the behavior dataset. Use `use_compute_termination=True` as the benchmark does: `compute_termination` marks every step from the first crash onwards, combining the raw `done` flag with an offline lidar/pose crash check (see `apply_termination_condition.py`); together with `remove_cons_terminals=True` the steps after the first terminal are dropped. Trajectories listed in `bad_trajectories` and the agents in `bad_agents` were excluded manually.
Since the most recent pose and lidar data are used, they might be old; `infos` contains the timestamps of when the action was computed and when pose, lidar and imu were last updated. IMU data was collected at 120 Hz, only the last IMU update is part of the observations, the full IMU stream is stored in the `imus` group of the zarr archive.
- f110-sim-stoch-v2: the 15 OPE target agents re-run in simulation from the same starting points as their real trajectories. It contains no `new_rewards`, so use `reward_config=None` or add them with `python f110_orl_dataset/relabel_reward.py --dataset f110-sim-stoch-v2 --reward_config <reward>.json`.

The legacy datasets f110-real-v0, f110-sim-v0, f110-real-v1 and f110-sim-v1 are no longer supported by this package; their archives remain available at [f110_datasets](https://github.com/HyberionBrew/f110_datasets).

# Observations
`F110Env.keys` returns the list of the observation keys. With the default `encode_cyclic=True` these are:
- poses_x, poses_y
- theta_sin, theta_cos
- ang_vels_z, linear_vels_x, linear_vels_y
- previous_action_steer, previous_action_speed

In the real dataset the velocities are estimates and contain outliers, in the simulated dataset linear_vels_y and ang_vels_z are 0.

# Important Arguments for gym.make
- as a name choose either `f110-real-stoch-v2` or `f110-sim-stoch-v2`
- When using `encode_cyclic=True` poses_theta (and progress) are replaced with sin and cosine encoded counterparts.
- `include_progress` adds the progress along the centerline of the track (between 0 and 1); f110-sim-stoch-v2 has no progress observation.
- `include_time_obs` adds an observation that is between 0 and 1 to denote the timestep with respect to the episode.
- `use_delta_actions` selects whether `actions` are the per-step changes of the steering and speed targets (`True`) or the targets themselves (`False`).
- `reward_config` selects the reward from the `new_rewards` group: reward_progress.json, reward_checkpoint.json, reward_lifetime.json or reward_min_act.json. With `None` the reward recorded during collection is used.

# Important Arguments for get_dataset
- timesteps_to_include controls the timesteps of an episode that are included, the benchmark uses (0, 250).
- only_agents/remove_agents, allows you to only pick certain agents and remove others.
- eval_only/train_only keep only the OPE target policies, respectively only the behavior agents.
- with zarr_path you can pick a zarr dir that is not standard


# Important other data Fields
- infos: timestamps of the recording
- model_name: the agent that was used to compute the actions; it can be loaded with `f110_agents.agent.Agent().load(name=model_name)` from [stochastic_ftg_agents](https://github.com/HyberionBrew/stochastic_ftg_agents/tree/ope-benchmark).
- actions: the first position gives the steering angle, the second the velocity (targets, not necessarily the achieved values).
- log_probs: the log probability of the action under the agent, per dimension as [speed, steering]; their sum is the log probability of the action.
- scans: The lidar scans
- terminals: 1 if termination
- timeouts: 1 if timeout, also 1 if termination is 1.
