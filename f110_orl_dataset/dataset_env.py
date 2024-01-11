import gymnasium as gym
# import so PAth
from pathlib import Path
from f110_gym.envs.f110_env import F110Env
# import spaces
from gymnasium.spaces import Box, Discrete
import numpy as np

import os 
import sys
import pickle
import zarr
from typing import Union, Tuple, Dict, Optional, List, Any
from f110_gym.envs import F110Env
from tqdm import tqdm 
from .config import *
import requests
import zipfile
import os
import numpy as np

# import ordered dict
from collections import OrderedDict
obs_dictionary_keys_circular = [
    "poses_x",
    "poses_y",
    "theta_sin",
    "theta_cos",
    "ang_vels_z",
    "linear_vels_x",
    "linear_vels_y",
    # "previous_action",
    "previous_action_steer",
    "previous_action_speed",
    "progress_sin",
    "progress_cos",
]

obs_dictionary_keys_og = [
    "poses_x",
    "poses_y",
    "poses_theta",
    "ang_vels_z",
    "linear_vels_x",
    "linear_vels_y",
    "previous_action_steer",
    "previous_action_speed",
    "progress",
]

if __name__ == "__main__":
    F110Env = gym.make('f110_with_dataset-v0',
    # only terminals are available as of tight now 
        **dict(name='f110_with_dataset-v0',
            config = dict(map="Infsaal", num_agents=1),
              render_mode="human")
    ) 

class F1tenthDatasetEnv(F110Env):
    """F110 environment which can load an offline RL dataset from a file.

    Similar to D4RL's OfflineEnv but with different data loading and
    options for customization of observation space. The dataloader Enviroment 
    is heavily inspired by the Trifinder RL Dataset 
    (https://webdav.tuebingen.mpg.de/trifinger-rl/docs/index.html)"""
    def __init__(
        self,
        name,
        dataset_url,
        reward_config = None,
        ref_max_score=None,
        ref_min_score=None,
        real_robot=False,
        flatten_obs=True,
        scale_obs=False,
        set_terminals=False,
        data_dir=None,
        original_reward = False,
        deprecated_action_scale = True,
        use_delta_actions = True,
        include_time_obs = True,
        agent_config_dir = None,
        redownload = False, # debugging, if ds changes
        encode_cyclic = True,
        timesteps_to_include = None,
        delta_factor = 0.3,
        **f1tenth_kwargs
        ):
        """
        Args:
            name (str): Name of the dataset.
            dataset_url (str): URL pointing to the dataset.
            ref_max_score (float): Maximum score (for score normalization)
            ref_min_score (float): Minimum score (for score normalization)
            real_robot (bool): Whether the data was collected on real
                robots.
            image_obs (bool): Whether observations contain camera
                images.
            visualization (bool): Enables rendering for simulated
                environment.
            flatten_obs (bool): Whether to flatten the observation. Can
                be combined with obs_to_keep.
            scale_obs (bool): Whether to scale all components of the
                observation to interval [-1, 1]. Only implemented
                for flattend observations.
            set_terminals (bool): Whether to set the terminals instead
                of the timeouts.
            data_dir (str or Path): Directory where the dataset is
                stored.  If None, the default data directory
                (~/.f110_rl_datasets) is used.
            recomputed_reward (bool): Whether to use the recomputed reward
            deprecated_action_scale (bool): Whether to use the deprecated action scale, 
                all dataset < v1 use this @ true
            use_delta_actions (bool): Whether to return delta actions as actions
        """
        super(F1tenthDatasetEnv, self).__init__(**f1tenth_kwargs)

        self.name = name
        self.dataset_url = dataset_url
        self.name = name
        self.ref_max_score = ref_max_score
        self.ref_min_score = ref_min_score
        self.real_robot = real_robot
        self.flatten_obs = flatten_obs
        self.scale_obs = scale_obs
        self.set_terminals = set_terminals
        self.deprecated_action_scale = deprecated_action_scale
        self.encode_cyclic = encode_cyclic
        self.timesteps_to_include = timesteps_to_include
        self.reward_config = reward_config
        self.include_time_obs = include_time_obs
        self.use_delta_actions = use_delta_actions
        self.delta_factor = delta_factor
        self._local_dataset_path = None
        if agent_config_dir is None:
            agent_config_dir = Path(__file__).parent / "agent_configs"
        
        print("Agent configs taken from:", agent_config_dir)

        if data_dir is None:
            data_dir = Path.home() / ".f110_rl_datasets" 
            data_dir.mkdir(parents=True, exist_ok=True)
    
            data_dir = data_dir / self.name
            data_dir = Path(data_dir)
        self.data_dir = data_dir

        # check if dataset exists
        if not(Path(f"{self.data_dir}.zip").exists()) or redownload:
            print(f"Attempting download - sometimes this does not work for me - you can always download the dataset from {self.dataset_url}")
            print(f"Place it in {self.data_dir} and rename it to {self.data_dir}.zip")
            self._download_file(self.dataset_url, f"{self.data_dir}.zip")#self.data_dir)

        if not(self.data_dir.exists()):
            self._unzip_file(f"{self.data_dir}.zip")        
       
        rays = int(1080/SUBSAMPLE)

        state_dict = OrderedDict()
        if encode_cyclic:
            self.keys = obs_dictionary_keys_circular
        else:
            self.keys = obs_dictionary_keys_og

        for obs in self.keys:
            # if contained in the original observation space
            if obs in self.observation_space.spaces:
                state_dict[obs] = self.observation_space.spaces[obs]
            else:
                # assumes that additional obs are in [0,1]
                state_dict[obs] = Box(0, 1, (1,), np.float32)
                # this is not actually true for prev_action!       

        if include_time_obs:
            state_dict["timestep"] = Box(0, 1, (1,), np.float32)
            self.keys.append("timestep")
        
        self.state_space = gym.spaces.Dict(state_dict)
        # print(self.action_space)
        self.observation_space = self.state_space 

        self.observation_space_orig = self.observation_space
        self.laser_obs_space = gym.spaces.Box(0, 1, (rays,), np.float32)
        self._orig_flat_obs_space = gym.spaces.flatten_space(self.observation_space)
        
        if self.flatten_obs:
            self.observation_space = self._orig_flat_obs_space



        self.dataset = dict(
            actions=[],
            observations=[],
            rewards=[],
            terminals=[],
            timeouts=[],
            infos=[],)
    
    def _download_file(self,url, destination):
        response = requests.get(url, stream=True)
        response.raise_for_status()
        print(destination)
        # destination = Path(destination)
        with open(destination, "wb") as file, tqdm(
            desc=str(destination),
            total=int(response.headers.get('content-length', 0)),
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                bar.update(file.write(data))

    def _unzip_file(self, zip_path):
        # Extract the directory from the zip path
        extract_path = os.path.splitext(zip_path)[0]

        # Create the directory if it doesn't exist
        os.makedirs(extract_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract all the contents into the specified directory
            for file in tqdm(iterable=zip_ref.namelist(), total=len(zip_ref.namelist()), desc='Extracting '):
                zip_ref.extract(member=file, path=extract_path)
    

    def get_dataset(self,
                    zarr_path: Union[str, os.PathLike] = None,
                    remove_agents: Optional[np.ndarray] = [],
                    only_agents: Optional[np.ndarray] = [],
                    debug=False, # load only part
                    ):
        """
        Loads and processes a dataset from a specified Zarr file.

        This function opens a Zarr file located at `zarr_path` and processes its contents 
        to form a structured dataset. It allows filtering of the data based on specific 
        agent models through `remove_agents` and `only_agents` parameters.

        Parameters:
        zarr_path (Union[str, os.PathLike], optional): The file path to the Zarr file 
            containing the dataset. If None, the path is taken from the instance's 
            `data_dir` attribute.
        remove_agents (Optional[np.ndarray], optional): An array of agent model names 
            to be excluded from the dataset. If specified, data related to these agents 
            will be removed.
        only_agents (Optional[np.ndarray], optional): An array of agent model names 
            to be exclusively included in the dataset. If specified, only data related 
            to these agents will be retained.
        debug (bool, optional): If set to True, only a portion of the dataset is loaded 
            for debugging purposes.

        Returns:
        dict: A dictionary containing the processed dataset. The dictionary structure 
            includes keys like 'actions', 'observations', and 'rewards', among others, 
            each containing corresponding data from the Zarr file.

        Raises:
        AssertionError: If both `remove_agents` and `only_agents` are specified, as they 
            are mutually exclusive.

        Example:
        >>> dataset = get_dataset(zarr_path="path/to/dataset.zarr", remove_agents=np.array(["agent1", "agent2"]))
        >>> print(dataset.keys())
        dict_keys(['actions', 'observations', 'rewards', ...])
        """

        assert len(remove_agents)==0 or len(only_agents)==0, "Cannot specify both only_agents and without_agents"
        if zarr_path is None:
            zarr_path = self.data_dir
        root = zarr.open(zarr_path, mode='r')
        
        print(f"The following agents are contained in the dataset: {[i for i in np.unique(root['model_name'][:])]}")
        # load all from the zarr file
        temp_dataset = dict()
        if self.use_delta_actions:
            temp_dataset["actions"] = root['actions'][:] * self.delta_factor
        else:
            temp_dataset["actions"] = root['raw_actions'][:]

        og_dataset_size = len(temp_dataset["actions"])
        # now we need to extract the observations, some additional processing is required
        raw_observations = {}
        for key in root["observations"].keys():
            raw_observations[key] = root["observations"][key][:]
        # from the raw observations, populate the observation dataset
        temp_dataset["observations"] = dict()
        temp_dataset["observations"]["poses_x"] = raw_observations["poses_x"][:]
        temp_dataset["observations"]["poses_y"] = raw_observations["poses_y"][:]
        # calculate theta from sin and cos

        temp_dataset["observations"]["ang_vels_z"] = raw_observations["ang_vels_z"][:]
        temp_dataset["observations"]["linear_vels_x"] = raw_observations["linear_vels_x"][:]
        temp_dataset["observations"]["linear_vels_y"] = raw_observations["linear_vels_y"][:]
        temp_dataset["observations"]["previous_action_steer"] = raw_observations["previous_action"][:,0]
        temp_dataset["observations"]["previous_action_speed"] = raw_observations["previous_action"][:,1]
        print("previous_action", temp_dataset["observations"]["previous_action_steer"][:5])

        #obs['progress_sin'] = np.array(np.sin(new_progress*2 * np.pi),dtype=np.float32)
        # obs['progress_cos'] = np.array(np.cos(new_progress*2 * np.pi),dtype=np.float32)
        #progress = 
        if self.encode_cyclic:
            temp_dataset["observations"]["progress_sin"] = raw_observations["progress_sin"][:]
            temp_dataset["observations"]["progress_cos"] = raw_observations["progress_cos"][:]
            temp_dataset["observations"]["theta_sin"] = raw_observations["theta_sin"][:]
            temp_dataset["observations"]["theta_cos"] = raw_observations["theta_cos"][:]
        else:
            temp_dataset["observations"]["progress"] = raw_observations["progress"][:]
            theta_sin = raw_observations["theta_sin"][:]
            theta_cos = raw_observations["theta_cos"][:]
            theta = np.arctan2(theta_sin, theta_cos)
            temp_dataset["observations"]["poses_theta"] = theta
        

        temp_dataset["terminals"] = root['done'][:]
        temp_dataset["timeouts"] = root['truncated'][:]

        if self.reward_config is not None:
            # check if the key is available
            if not self.reward_config in root["new_rewards"]:
                # print(f"Warning: reward config {self.reward_config} not available")
                raise ValueError(f"Reward config {self.reward_config} not available")
            temp_dataset["rewards"] = root["new_rewards"][self.reward_config][:]
        else:
            temp_dataset["rewards"] = root['rewards'][:]
        temp_dataset["infos"] = dict()
        for key in root["infos"].keys():
            if len(root["infos"][key][:]) == 0:
                print("Warning: empty info key: ", key)
                temp_dataset["infos"][key] = np.zeros_like(root["truncated"][:])
            else:
                temp_dataset["infos"][key] = root["infos"][key][:]
        temp_dataset["model_name"] = root['model_name'][:]
        temp_dataset["log_probs"] = root['log_prob'][:]
        temp_dataset["scans"] = root["observations"]['lidar_occupancy'][:]

        indices_to_remove = np.array([])
        
        if len(remove_agents) > 0:
            # remove agents
            # Find all indices where the model_name matches any name in remove_agents
            indices_to_remove = np.isin(root['model_name'][:], remove_agents)
            # Get the indices where the condition is true
            indices_to_remove = np.where(indices_to_remove)[0]
        
        if len(only_agents) > 0:
            indices_to_keep = np.isin(root['model_name'][:], only_agents)
            indices_to_keep = np.where(indices_to_keep)[0]
            # Get all indices that are not in indices_to_keep
            all_indices = np.arange(root['model_name'].shape[0])
            indices_to_remove_only_agents = np.setdiff1d(all_indices, indices_to_keep)
            indices_to_remove = np.union1d(indices_to_remove, indices_to_remove_only_agents)

        if self.timesteps_to_include is not None:
            clipped_indices, new_truncates = self.clip_trajectories(temp_dataset, 
                                                                    min_len=self.timesteps_to_include[0], 
                                                                    max_len=self.timesteps_to_include[1])
            
            temp_dataset["timeouts"] = temp_dataset["timeouts"] | new_truncates
            indices_to_remove = np.union1d(indices_to_remove, clipped_indices)
        # cast to int indices_to_remove
        indices_to_remove = indices_to_remove.astype(np.int64)
        


        if self.set_terminals:
            temp_dataset["terminals"] = root['done'][:] | root['truncated'][:]
        
        if self.include_time_obs:
            temp_dataset["observations"]["timestep"] = self.add_timesteps(temp_dataset)
            # below is for legacy reasons
            temp_dataset["timesteps"] = temp_dataset["observations"]["timestep"][:].copy()
            # print("timesteps", temp_dataset["observations"]["timestep"][:252])

        dataset_removed = self._remove_indices_from_dataset(temp_dataset, indices_to_remove)

        print("dataset original size: ", og_dataset_size)
        print("dataset after removing indices size: ", len(dataset_removed["observations"]["poses_x"]))
        print("remaining agents:", np.unique(dataset_removed["model_name"]))
        if self.flatten_obs:
            arrays_to_concat = [dataset_removed['observations'][key].reshape([dataset_removed['observations'][key].shape[0], -1]) for key in self.keys]
            concatenated_obs = np.concatenate(arrays_to_concat, axis=-1)
            dataset_removed['observations'] = concatenated_obs.reshape([concatenated_obs.shape[0], -1])
        dataset_removed["infos"]["obs_keys"] = self.keys

        return dataset_removed
    

    def _remove_indices_from_dataset(self, dataset, indices_to_remove):
        """
        Recursively removes entries at specified indices from all arrays in the dataset.

        Args:
        dataset (dict): The dataset from which indices need to be removed.
        indices_to_remove (np.ndarray): Indices that need to be removed.

        Returns:
        dict: The dataset with specified indices removed from all arrays.
        """
        for key, value in dataset.items():
            # If the value is a dictionary, recursively call this function
            if isinstance(value, dict):
                self._remove_indices_from_dataset(value, indices_to_remove)
            # If the value is a numpy array, remove the specified indices
            elif isinstance(value, np.ndarray):
                #print(key)
                #print(value.shape)
                dataset[key] = np.delete(value, indices_to_remove, axis=0)

        return dataset

    def get_observation_keys(self):
        return list(self.dataset["observations"].keys())
    
    def clip_trajectories(self, data_dict, min_len=0, max_len=100):
        terminals = np.logical_or(data_dict['terminals'], data_dict['timeouts'])
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))

        indices_to_remove = []
        truncates = np.zeros_like(data_dict['terminals'])
        for i in range(len(start_indices) - 1):
            start, end = start_indices[i], start_indices[i + 1]
            
            # Determine the actual start and end indices after clipping
            clipped_start = max(start + min_len, start)
            clipped_end = min(start + max_len, end)

            # Identify the indices that need to be removed
            if clipped_start > start:
                indices_to_remove.extend(range(start, clipped_start))
            if clipped_end < end:
                indices_to_remove.extend(range(clipped_end, end))
            truncates[clipped_end - 1] = 1

        # Convert indices_to_remove to numpy array
        indices_to_remove = np.array(indices_to_remove, dtype=np.int64)
        
        # set 
        return indices_to_remove, truncates
    
    def add_timesteps(self, dataset):
        """
        Calculates timesteps as normalized values (ranging from 0 to 1) for each trajectory in the dataset.

        Args:
            dataset (dict): The dataset containing 'terminals' and 'timeouts' keys.

        Returns:
            np.ndarray: An array of timesteps for each trajectory, normalized between 0 and 1.
        """
        # Check for necessary keys
        if 'terminals' not in dataset or 'timeouts' not in dataset:
            print("Error: 'terminals' and 'timeouts' keys are required.")
            return None

        # Calculate end indices of trajectories
        terminals = np.logical_or(dataset['terminals'], dataset['timeouts'])
        end_indices = np.where(terminals)[0] + 1

        # Calculate normalized timesteps for each trajectory
        all_timesteps = []
        start_idx = 0
        for end_idx in end_indices:
            # Number of timesteps in the trajectory
            num_timesteps = end_idx - start_idx
            # Normalized timesteps between 0 and 1
            timesteps = np.linspace(0, 1, num_timesteps, endpoint=False)
            all_timesteps.append(timesteps)
            start_idx = end_idx

        # Concatenate all timesteps
        all_timesteps = np.concatenate(all_timesteps)

        return all_timesteps
    
    def get_laser_scan(self, states, subsample_laser):
        # correctly extract x,y from the obs_dictionary
        xy = states[:, :2]

        if self.encode_cyclic:
            theta_sin = states[:, 2]
            theta_cos = states[:, 3]
            #print(states)
            #print(theta_sin)
            #print(theta_cos)
            theta = np.arctan2(theta_sin, theta_cos)
        else:
            theta = states[:, 2]
        
        
        #print("states laser")
        #print(states)

        #print("theta", theta)
        # Expand the dimensions of theta
        theta = np.expand_dims(theta, axis=-1)
        joined = np.concatenate([xy, theta], axis=-1)
        
        all_scans = []
        for pose in joined:
            # print("sampling at pose:", pose)
            # Assuming F110Env.sim.agents[0].scan_simulator.scan(pose, None) returns numpy array
            scan = self.sim.agents[0].scan_simulator.scan(pose, None)[::subsample_laser]
            scan = scan.astype(np.float32)
            all_scans.append(scan)
        # normalize the laser scan
        all_scans = np.array(all_scans)
        return all_scans
    
    def normalize_laser_scan(self, batch_laserscan):
        batch_laserscan = np.asarray(batch_laserscan)
        assert len(batch_laserscan.shape) == 2, "Batch should be 2D"
        batch_laserscan = np.clip(batch_laserscan, 0, 10)
        batch_laserscan = batch_laserscan / 10
        return batch_laserscan
    

    def unflatten_batch(self, batch):
        batch = np.asarray(batch)

        assert len(batch.shape) == 2, "Batch should be 2D"

        batch_dict = {}
        
        start_idx = 0
        for key, space in self.state_space.spaces.items():
            # Calculate how many columns this part of the observation takes up
            space_shape = np.prod(space.shape)
            
            # Slice the appropriate columns from the batch
            batch_slice = batch[:,start_idx:start_idx+space_shape]
            #print(key)
            #print(batch_slice.shape)
            # If the space has multi-dimensions, reshape it accordingly
            if len(space.shape) > 1:
                batch_slice = batch_slice # .reshape(space.shape)
                # print(batch_slice.shape)
            # squeeze all 1 dimensions exepct dim 0
            else:
                batch_slice = np.squeeze(batch_slice, axis=1)
            batch_dict[key] = batch_slice
            start_idx += space_shape

        assert start_idx == batch.shape[1], "Mismatch in the number of columns"
        return batch_dict
    
    def flatten_batch(self, batch_dict):
        batch = np.zeros((len(batch_dict["poses_x"]), 11))
        # print(batch.shape)
        i = 0
        for key, obs in self.state_space.spaces.items():
            # Calculate how many columns this part of the observation takes up
            
            space_shape = np.prod(obs.shape)
            # Slice the appropriate columns from the batch
            batch_slice = batch_dict[key]
            # If the space has multi-dimensions, reshape it accordingly
            if len(obs.shape) > 1:
                batch_slice = batch_slice.reshape((-1, space_shape))
            batch[:, i:i+space_shape] = batch_slice
            i = i + space_shape
        return batch
    
    def simulate(self, agent, render=True, debug=True, starting_state=None, save_data=False, rollouts=1, episode_length=500):
        """
        Simulates the environment using the specified agent.

        Args:
            agent (Agent): The agent to use for simulation.
            render (bool, optional): Whether to render the simulation.
            debug (bool, optional): Whether to print debug information.
            starting_state (np.ndarray, optional): The starting state of the simulation.
                If None, the simulation starts from the random states along the raceline.
        Returns:
            Tuple (np.ndarray, np.ndarray, np.ndarray, np.ndarray): The states, actions, 
                rewards, and terminals from the simulation.
        """
        import f110_orl_dataset.simulation_helpers as sh
        from f110_agents.rewards import Progress 
        if starting_state is None:
            starting_states = sh.get_start_position(self.track, start_positions=rollouts)
        progress = Progress(self.track, lookahead=200)
        for starting_state in starting_states:
            reset_options = dict(poses=np.array([starting_state]))
            self.reset(options=reset_options)
            agent.reset()
            
            current_steer = 0.0
            current_vel = 0.0
            obs, reward, done, truncated, info = self.step(np.array([[current_steer,current_vel]]))
            progress.reset(np.array([[obs["poses_x"][0], obs["poses_y"][0]]]))
            for timestep in range(episode_length):
                obs["previous_action_steer"] = np.array([current_steer])
                obs["previous_action_speed"] = np.array([current_vel])

                del obs["ego_idx"]
                del obs["lap_counts"]
                del obs["lap_times"]
                
                # we need to add the timestep to the obs, transform the laser scan and 
                # add lidar_occupancy
                new_progress = progress.get_progress(np.array([[obs["poses_x"][0], obs["poses_y"][0]]]))
                obs["progress"] = np.array(new_progress)
                scan = obs["scans"]
                scan = self.normalize_laser_scan(scan)
                obs["lidar_occupancy"] = scan[:,::SUBSAMPLE]
                del obs["scans"]

                if self.encode_cyclic:
                    obs['progress_sin'] = np.array(np.sin(new_progress*2 * np.pi),dtype=np.float32)
                    obs['progress_cos'] = np.array(np.cos(new_progress*2 * np.pi),dtype=np.float32)
                    obs['theta_sin'] = np.array(np.sin(obs["poses_theta"][0]),dtype=np.float32)
                    obs['theta_cos'] = np.array(np.cos(obs["poses_theta"][0]),dtype=np.float32)
                    del obs["poses_theta"]
                    del obs["progress"]
                if self.include_time_obs:
                    obs["timestep"] = np.array([timestep/episode_length])
                # add progress
                print(obs)
                # run agent
                _, action, log_prob = agent(obs) #timestep=np.array([timestep/episode_length]))
                action = action[0]
                if self.use_delta_actions:
                    #print("delta actions")
                    #print(action, current_steer, current_vel)
                    current_steer += action[0] 
                    current_vel += action[1]
                else:
                    current_steer = action[0]
                    current_vel = action[1]
                print("current steer", current_steer)
                print("current vel", current_vel)
                print("obs, linear vel", obs["linear_vels_x"])
                # do 5 steps
                for _ in range(5):
                    obs, reward, done, truncated, info = self.step(np.array([[current_steer,current_vel]]))
                    if done or truncated:
                        break
                if render:
                    self.render()
                if done or truncated:
                    break