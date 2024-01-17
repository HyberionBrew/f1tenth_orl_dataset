# Trajectories

![trajectories_with_skulls](https://github.com/HyberionBrew/f1tenth_orl_dataset/assets/31421881/738391f5-ca3f-4712-8a40-fcdcef8ba9ef)


The f110-real-v1 dataset contains a bit over 300 trajectories, collected with 30 different agents. Each trajectory contains 251 timesteps. In the above picture crashes (terminals) are marked with black dots, normal truncations are black x's and the colored x's are the starting points along the track.

# How to install
Tested with conda and python=3.8.

First, you will need an adapted version of the f1tenth-gym.

checkout the v1.0.0 branch:

```
git clone https://github.com/HyberionBrew/f1tenth_gym.git
cd f1tenth_gym
git checkout v1.0.0

```

install it:

```
cd ..
pip install -e f1tenth_gym
```
Install this package
```
git clone https://github.com/HyberionBrew/f1tenth_orl_dataset.git
pip install -e f1tenth_orl_dataset
```
Run a small test inside the f1tenth_orl_dataset folder:

```
python test.py
```
The test file shows most of the functionality.

# Explanation of the datatsets
The datasets are recorded with a control frequency of 20 HZ. For the real world dataset lidar and pose data are also available with roughly this frequency.

Currently, there are four datasets available:
- f110-real-v0: contains data collected in the real world. This dataset contains 25 trajectories per Agent, there are a total of 7 different agents. The dataset lacks imu data, which will be added with the next update.
Since the most recent pose and available lidar data are used, they might be old, you can see the timestamp of when the agent is invoked in the infos agent_action field. The lidar and pose timestamp field in infos inform on when the pose and lidar data where last udpated. 

- f110-sim-v0: contains data that uses the same starting points as the real dataset but runs in simulation. The timestamps in infos are always 0 and irrelevant, since we have perfect scans/pose.

- f110-real-v1: 10 trajectories per agent, for 30 Agents (sometimes a bit more sometimes a bit less, due to some trajectories being invalid due to recording issues) There is imu data, which was collected at 120Hz, only the last IMU update is made available in the observations. Other IMU data would need to be extracted from the zarr archive, where it is stored as imu with its respective timestamps.
- f110-sim-v1: Same starting points as real, just in sim.



# Observations
`dataset["infos"]["obs_keys"]`

returns a list of the observation keys.

The following are the default observations:
- poses_x
- poses_y
- poses_theta
- ang_vels_z
- linear_vels_x
- linear_vels_y
- progress

The vels are currently 0. Progress is defined over the middle line of the track and is between 0 and 1.
# Important Arguments for gym.make
- as a name choose either the real or sim dataset
- When using `encode_cyclic=True` progress and poses_theta are replaced with sin and cosine encoded counterparts.

- `include_time_obs` adds an observation that is between 0 and 1 to denote the timestep with respect to the episode.
- `reward_config` assigns the reward that is present in the rewards datafield:
  * reward_raceline.json (distance to the raceline)
  * reward_progress.json ( standard time differen progress, with respect to middle line)
  * reward_lidar.json (the shortest lidar ray - essentially safety)

# Important Arguments for get_dataset
- timesteps_to_include controls the timesteps of an epsiode that are included. For sim each episode has a maximum of 500, in real 250.
- only_agents/remove_agents, allows you to only pick certain agents and remove others.
- with zarr_path you can pick a zarr dir that is not standard


# Important other data Fields
- infos: for the real data contains the timestamps when the data was recorded
- model_names: Yields the model that was used to compute the actions, if you are interested in running the models, please reach out to me.
- action: the first position gives the target steering angle, the second the target velocity (this is not necessarily the achieved speed and velocity).
- log_probs: The log probability of taking this action
- scans: The lidar scans
- terminals: 1 if termination
- timeouts: 1 if timeout (In real dataset after 250 timesteps, in sim after 500 per default), also 1 if termination is 1.


