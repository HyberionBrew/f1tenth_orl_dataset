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

angle_increment = 4.712389 /54

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
   # "progress_sin",
   # "progress_cos",
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
    #"progress",
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
        bad_trajectories = [],
        bad_agents = [],
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
        include_pose_time_diff = False,
        include_action_pose_time_diff = False,
        include_progress = True,
        agent_config_dir = None,
        redownload = False, # debugging, if ds changes
        encode_cyclic = True,
        timesteps_to_include = None,
        remove_cons_terminals = True,
        delta_factor = 1.0,
        skip_download = False,
        include_vesc_fault_trajectories = False,
        set_previous_step_terminals : int = 0,
        use_compute_termination = True,
        eval_model_names = None,
        train_only = False,
        eval_only = False,
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
        self.train_only = train_only
        self.eval_only = eval_only
        self.bad_agents = bad_agents
        self.eval_agents = eval_model_names
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
        self.include_progress = include_progress
        self.use_delta_actions = use_delta_actions
        self.delta_factor = delta_factor
        self._local_dataset_path = None
        self.bad_trajectories = bad_trajectories
        self.include_pose_time_diff = include_pose_time_diff
        self.include_action_pose_time_diff = include_action_pose_time_diff
        self.remove_cons_terminals = remove_cons_terminals
        self.include_vesc_fault_trajectories = include_vesc_fault_trajectories
        self.set_previous_step_terminals = set_previous_step_terminals
        self.use_compute_termination = use_compute_termination
        self.angles =  np.abs(np.arange(-9,10,1) * angle_increment)
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
        if (not(Path(f"{self.data_dir}.zip").exists()) or redownload) and not(skip_download):
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
        if self.include_progress:
            if self.encode_cyclic:
                state_dict["progress_sin"] = Box(-1, 1, (1,), np.float32)
                state_dict["progress_cos"] = Box(-1, 1, (1,), np.float32)
                self.keys.append("progress_sin")
                self.keys.append("progress_cos")
            else:
                state_dict["progress"] = Box(0, 1, (1,), np.float32)
                self.keys.append("progress")

        if include_time_obs:
            state_dict["timestep"] = Box(0, 1, (1,), np.float32)
            self.keys.append("timestep")
        if include_pose_time_diff:
            state_dict["pose_time_diff"] = Box(0, 1, (1,), np.float32)
            self.keys.append("pose_time_diff")
        if include_action_pose_time_diff:
            state_dict["action_pose_time_diff"] = Box(0, 1, (1,), np.float32)
            self.keys.append("action_pose_time_diff")

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
                    train_only = False,
                    eval_only = False,
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
        assert not (self.train_only and self.eval_only), "Cannot specify both train_only and eval_only"
        # add the bad agent
        if isinstance(remove_agents, list):
            remove_agents = np.array(remove_agents)
        if len(only_agents) == 0:
            remove_agents = np.append(remove_agents, self.bad_agents)
        if eval_only:
            only_agents = self.eval_agents
            remove_agents = []
        
        if train_only:
            remove_agents = np.append(remove_agents, self.eval_agents)

        if zarr_path is None:
            print("using default path")
            zarr_path = self.data_dir
        print(f"path: {zarr_path}")
        root = zarr.open(zarr_path, mode='r')
        # print(root["model_name"].shape)
        print(f"The following # of agents is contained in the dataset: {len([i for i in np.unique(root['model_name'][:])])}")
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
        if "previous_action" in raw_observations:
            temp_dataset["observations"]["previous_action_steer"] = raw_observations["previous_action"][:,0]
            temp_dataset["observations"]["previous_action_speed"] = raw_observations["previous_action"][:,1]
        else:
            temp_dataset["observations"]["previous_action_steer"] = raw_observations["previous_action_steer"][:]
            temp_dataset["observations"]["previous_action_speed"] = raw_observations["previous_action_speed"][:]
        


        if self.include_pose_time_diff:
            temp_dataset["observations"]["pose_time_diff"] = root["infos"]["pose_timestamp"] - np.roll(root["infos"]["pose_timestamp"],-1)
            dones = root['truncated'][:] | root['done'][:]
            # where we have dones set the pose_time_diff to 0
            temp_dataset["observations"]["pose_time_diff"][dones] = 0.0
        if self.include_action_pose_time_diff:
            temp_dataset["observations"]["action_pose_time_diff"] = np.array(root["infos"]["pose_timestamp"]) - np.array(root["infos"]["action_timestamp"])
            dones = root['truncated'][:] | root['done'][:]
            # where we have dones set the pose_time_diff to 0
            temp_dataset["observations"]["action_pose_time_diff"][dones] = 0.0
        # print("previous_action", temp_dataset["observations"]["previous_action_steer"][:5])

        #obs['progress_sin'] = np.array(np.sin(new_progress*2 * np.pi),dtype=np.float32)
        # obs['progress_cos'] = np.array(np.cos(new_progress*2 * np.pi),dtype=np.float32)
        #progress = 
        if self.encode_cyclic:
            if self.include_progress:
                temp_dataset["observations"]["progress_sin"] = raw_observations["progress_sin"][:]
                temp_dataset["observations"]["progress_cos"] = raw_observations["progress_cos"][:]
            temp_dataset["observations"]["theta_sin"] = raw_observations["theta_sin"][:]
            temp_dataset["observations"]["theta_cos"] = raw_observations["theta_cos"][:]
        else:
            if self.include_progress:
                temp_dataset["observations"]["progress"] = raw_observations["progress"][:]
            theta_sin = raw_observations["theta_sin"][:]
            theta_cos = raw_observations["theta_cos"][:]
            theta = np.arctan2(theta_sin, theta_cos)
            temp_dataset["observations"]["poses_theta"] = theta

        if self.use_compute_termination:
            temp_dataset["terminals"] = root["compute_termination"][:]
            temp_dataset["timeouts"] = root["truncated"][:] | root['compute_termination'][:]

        else:
            temp_dataset["terminals"] = root['done'][:]
            temp_dataset["timeouts"] = root['truncated'][:] | root['done'][:]
        # print the distance between all timeouts
        #debug = np.where(temp_dataset["timeouts"] == True)[0]
        #print("timeouts", debug)
        #print("timeouts", debug[1:] - debug[:-1])
        #print("debug", np.count_nonzero(debug[1:] - debug[:-1]))
        #print(np.where(debug[1:] - debug[:-1]==311))

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
        # print("hi")
        #print(indices_to_remove)
        #print(temp_dataset["timeouts"])
        
        if len(self.bad_trajectories) > 0:

            _indices_to_remove = np.array([], dtype=np.int64)
            for trajectory_start in self.bad_trajectories:
                end = temp_dataset["timeouts"][trajectory_start:]
                end_index = np.where(end)[0]

                # Check if end_index is not empty and calculate the actual end point
                if len(end_index) > 0:
                    end_index = end_index[0] + trajectory_start
                else:
                    # If end_index is empty, it means the trajectory goes till the end of the dataset
                    end_index = len(temp_dataset["timeouts"]) - 1

                # Use numpy.append to append the new range to the existing array
                # print(len(np.arange(trajectory_start, end_index + 1, dtype=np.int64)))
                indices_to_remove = np.append(indices_to_remove, np.arange(trajectory_start, end_index + 1, dtype=np.int64))
                # change type to int
                indices_to_remove = indices_to_remove.astype(np.int64)
        # remove all consecutive terminals except the first one
        if self.set_previous_step_terminals>0:
            # loop over starts and ends
            starts = np.where(np.roll(temp_dataset["timeouts"],1)==True)[0]
            terms = np.where(temp_dataset["terminals"]==True)[0]
            #print(starts)
            #print(terms)
            for end in terms:
                # find closest start
                relevant_starts = starts[starts < end]
                start = starts[np.argmin(end-relevant_starts)]
                if start+1 == end:
                    continue
                #print(start, end)
                #print(temp_dataset["terminals"][start:end])
                temp_dataset["terminals"][max(start, end- self.set_previous_step_terminals - 1):end] = True
                temp_dataset["timeouts"][max(start, end- self.set_previous_step_terminals - 1):end] = True
                #print(temp_dataset["terminals"][start:end])
            starts = np.where(np.roll(temp_dataset["timeouts"],1)==True)[0]
            terms = np.where(temp_dataset["terminals"]==True)[0]
            #print(starts)
            #print(terms)
        if self.remove_cons_terminals:
            cons_terminal_indices = self._find_consecutive_terminals_indices(temp_dataset["terminals"])
        else:
            cons_terminal_indices = np.array([])

        #print(cons_terminal_indices)
        indices_to_remove = np.union1d(indices_to_remove,
                                       cons_terminal_indices)
        
        #print(len(indices_to_remove))
        i = 0
        import matplotlib.pyplot as plt
        #print("vesc faults", len(root["vesc_faults"]))
        # print("timeouts", len(temp_dataset["timeouts"]))
        #assert len(root["vesc_faults"]) == len(temp_dataset["timeouts"])

        if not(self.include_vesc_fault_trajectories):
            ends = np.where(temp_dataset["timeouts"] == True)[0]
            starts = np.where(np.roll(temp_dataset["timeouts"],1) == True)[0]
            for start, end in zip(starts, ends):
                #print(start, end)
                #print(np.array(root["vesc_faults"][start:end],dtype=bool))
                if abs(start - end) < 2:
                    continue
                try:
                    if np.array(root["vesc_faults"][start:end],dtype=bool).any():
                        #plt.plot(np.array(root["vesc_faults"][start:end]))
                        #plt.show()
                        i += 1
                        indices_to_remove = np.union1d(indices_to_remove, 
                                                    np.arange(start, end + 1, dtype=np.int64))
                except:
                    pass
                    #print("Error: vesc_faults not available (have been deprecated)")
            # loop over all trajectories and find the ones where vesc_fault
        #print(i)
        #print(len(indices_to_remove))
        # print(indices_to_remove)
        # plot the timeouts of the indices_to_remove
        #import matplotlib.pyplot as plt
        #print(indices_to_remove)
        #plt.plot(temp_dataset["timeouts"][indices_to_remove])
        #plt.show() 

        

        if len(remove_agents) > 0:
            # remove agents
            # Find all indices where the model_name matches any name in remove_agents
            _indices_to_remove = np.isin(root['model_name'][:], remove_agents)
            # Get the indices where the condition is true
            _indices_to_remove = np.where(_indices_to_remove)[0]
            indices_to_remove = np.union1d(indices_to_remove, _indices_to_remove)
        #print(indices_to_remove)
        if len(only_agents) > 0:
            indices_to_keep = np.isin(root['model_name'][:], only_agents)
        
            indices_to_keep = np.where(indices_to_keep)[0]

            #print(indices_to_keep)
            #print(np.unique(root['model_name'][indices_to_keep]))
            # Get all indices that are not in indices_to_keep
            all_indices = np.arange(root['model_name'].shape[0])
            indices_to_remove_only_agents = np.setdiff1d(all_indices, indices_to_keep)
            indices_to_remove = np.union1d(indices_to_remove, indices_to_remove_only_agents)
        #print(indices_to_remove)
        if self.timesteps_to_include is not None:
            clipped_indices, new_truncates = self.clip_trajectories(temp_dataset, 
                                                                    min_len=self.timesteps_to_include[0], 
                                                                    max_len=self.timesteps_to_include[1])
            
            temp_dataset["timeouts"] = temp_dataset["timeouts"] | new_truncates
            indices_to_remove = np.union1d(indices_to_remove, clipped_indices)
        


        # cast to int indices_to_remove
        indices_to_remove = indices_to_remove.astype(np.int64)
        #print(new_truncates)
        # print(indices_to_remove)
        if self.set_terminals:
            if self.use_compute_termination:
                temp_dataset["terminals"] =  temp_dataset["timeouts"] | temp_dataset["terminals"] | root['compute_termination'][:] | root['truncated'][:] | root['done'][:]
            else:
                temp_dataset["terminals"] = temp_dataset["terminals"] | root['done'][:] | root['truncated'][:]


            # print("timesteps", temp_dataset["observations"]["timestep"][:252])
        #print(temp_dataset["terminals"][152013:152065])
        #print(temp_dataset["timeouts"][152013:152065])
        #print(np.diff(np.sort(indices_to_remove)[-100:]))
        #print(np.sort(indices_to_remove)[-100:])
        
        dataset_removed = self._remove_indices_from_dataset(temp_dataset, indices_to_remove)
        #print(dataset_removed["terminals"][-2:])
        #print(dataset_removed["timeouts"][-2:])
        #exit()
        #
        if self.include_time_obs:
            dataset_removed["observations"]["timestep"] = self.add_timesteps(dataset_removed)
            # below is for legacy reasons
            dataset_removed["timesteps"] = dataset_removed["observations"]["timestep"][:].copy()
        debug = np.where(dataset_removed["timeouts"] == True)[0]
        # print("timeouts, dataset_removed", debug[1:] - debug[:-1])
        print("dataset original size: ", og_dataset_size)
        print("dataset after removing indices size: ", len(dataset_removed["observations"]["poses_x"]))
        print("remaining agents:", len(np.unique(dataset_removed["model_name"])))
        if self.flatten_obs:
            arrays_to_concat = [dataset_removed['observations'][key].reshape([dataset_removed['observations'][key].shape[0], -1]) for key in self.keys]
            concatenated_obs = np.concatenate(arrays_to_concat, axis=-1)
            dataset_removed['observations'] = concatenated_obs.reshape([concatenated_obs.shape[0], -1])
        dataset_removed["infos"]["obs_keys"] = self.keys

        return dataset_removed
    
    def _find_consecutive_terminals_indices(self, terminals):
        """
        Find indices of consecutive ones in a numpy array.

        Args:
        terminals (np.array): A numpy array of terminals (0s and 1s).

        Returns:
        np.array: Indices of consecutive ones, except for the first one in each sequence.
        """
        # Ensure terminals is a numpy array
        terminals = np.asarray(terminals)

        # Find where the changes occur
        change_points = np.where(np.diff(terminals) != 0)[0] + 1
        """
        print(np.sum(terminals))
        
        print("change_ps",change_points)
        print(terminals[211:250])
        import matplotlib.pyplot as plt
        print(len(change_points))
        print(terminals[213])
        print(terminals[250])
        print(terminals[251])
        print(terminals[252])
        plt.plot(terminals[211:250])
        plt.show()
        """
        # Add the start and end of the array for completeness
        change_points = np.concatenate([[0], change_points, [len(terminals) - 1]])

        # Initialize an empty list to store indices
        consecutive_ones_indices = []

        # Iterate through the change points
        for start, end in zip(change_points, change_points[1:]):
            if terminals[start] == 1:  # Check if the sequence starts with a 1
                # Add all indices after the first 1 in the sequence
                consecutive_ones_indices.extend(range(start + 1, end))

        return np.array(consecutive_ones_indices)

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
        return self.keys
    
    def clip_trajectories(self, data_dict, min_len=0, max_len=100):
        terminals =data_dict['timeouts'] | data_dict['terminals']
        start_indices = np.where(terminals[:-1] & ~terminals[1:])[0] + 1
        start_indices = np.concatenate(([0], start_indices))
        # print("Trajectories", len(start_indices))
        indices_to_remove = []
        truncates = np.zeros_like(data_dict['terminals'])
        for i in range(len(start_indices)):

            start  = start_indices[i]
            end = start_indices[i + 1] if i + 1 < len(start_indices) else len(terminals)
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
        start_indices = np.where(np.roll(terminals,1))[0]
        end_indices = np.where(terminals)[0] + 1
        # find longest trajectory
        longest_trajectory = end_indices - start_indices
        #print(terminals)
        longest_trajectory = np.max(longest_trajectory)
        #print(longest_trajectory)

        # Calculate normalized timesteps for each trajectory
        all_timesteps = []
        start_idx = 0
        timesteps = np.linspace(0, 1, longest_trajectory, endpoint=False)
        for start_idx, end_idx in zip(start_indices, end_indices):
            # Number of timesteps in the trajectory
            # Normalized timesteps between 0 and 1
            #print(len(timesteps[:end_idx - start_idx]))
            all_timesteps.append(timesteps[:end_idx - start_idx])
        # Concatenate all timesteps
        all_timesteps = np.concatenate(all_timesteps)
        #print(len(terminals))
        #print(len(all_timesteps))
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
    
    def get_specific_obs(self, obs, keys):
        if self.flatten_obs:
            # find position of keys in self.keys
            positions = [self.keys.index(key) for key in keys]
            # return the corresponding observations
            return obs[:,positions]
        else:
            raise NotImplementedError("Not implemented for non-flattened observations, just access only correct keys")

    def unflatten_batch(self, batch , keys=None):
        if keys is None:
            keys = self.state_space.spaces.items()
        else:
            key_ = [(key, self.state_space.spaces[key]) for key in keys]
            keys = key_
        batch = np.asarray(batch)

        assert len(batch.shape) == 2, "Batch should be 2D"

        batch_dict = {}
        
        start_idx = 0
        # print(batch.shape)
        for key, space in keys:
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
        batch = np.zeros((len(batch_dict["poses_x"]), len(self.keys)))
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
    
    def compute_reward_trajectories(self, trajectories, actions, terminations, reward_config, scans=None, precomputed_progress=False):
        """_summary_

        Args:
            states (_type_): shape (N, timesteps, state_dim)
            actions (_type_): (N, timesteps, action_dim)
            terminations (int): if < timestep indicates where the termination occurs 
            truncations (_type_): _description_
            reward_config (_type_): _description_
        """
        from f110_orl_dataset.config_new import RewardConfig
        from f110_orl_dataset.fast_reward import MixedReward
        config = RewardConfig(reward_config)
        # states are assumed to be unnormalized!
        rewardModel = MixedReward(self, config)
        if precomputed_progress:
            progress_obs = trajectories[:,:,self.keys.index("progress")]
        else:
            # print("Computing Progress")
            progress_obs = self.compute_progress(trajectories)
        #print(progress_obs.shape)
        #print("---")

        #print(trajectories.shape)
        actions = np.array(actions)
        if reward_config == "reward_lidar.json":
            if scans is None:
                scans = np.zeros((trajectories.shape[0],trajectories.shape[1],int(1080/20)))
                for trajectory_idx in range(len(trajectories)): 
                    scan_ = self.get_laser_scan(trajectories[trajectory_idx], 20)
                    scan_ = self.normalize_laser_scan(scan_)
                    scans[trajectory_idx] = scan_
        else:
            scans = np.zeros((trajectories.shape[0],trajectories.shape[1],int(1080/20)))
        rewards = np.zeros((trajectories.shape[0],trajectories.shape[1]))
        # create tqdm bar
       #  print("=")
        for i, (trajectory, termination, progress) in enumerate(zip(trajectories,terminations,progress_obs)):
            unflattened_states = self.unflatten_batch(trajectory, keys=self.keys)

            #print(unflattened_states["poses_x"].shape)
            #print(unflattened_states["poses_y"].shape)
            unflattened_states["progress"] = progress
            # col = np.zeros((len(trajectory),1))
            #print(actions[i].shape)
            reward , _ = rewardModel(unflattened_states, actions[i], int(termination), scans[i])
            rewards[i] = reward
        return rewards
        
    def compute_progress(self, trajectories):
        from f110_orl_dataset.compute_progress import Progress
        progress_obs_np = np.zeros((trajectories.shape[0],trajectories.shape[1],1))
        #print(self.track.centerline)
        # import matplotlib.pyplot as plt
        #plt.plot(self.track.centerline.xs, self.track.centerline.ys)
        #plt.plot(states[0,:,0],states[0,:,1])
        #plt.show()
        progress = Progress(self.track, lookahead=200)
        pose = lambda traj_num, timestep: np.array([(trajectories[traj_num,timestep,0],trajectories[traj_num,timestep,1])])
        for i in range(0,trajectories.shape[0]):
            # progress = Progress(states_inf[i,0,:])
            progress.reset(pose(i,0))
            for j in range(0,trajectories.shape[1]):
                progress_obs_np[i,j,0] = progress.get_progress(pose(i,j))
        return progress_obs_np
    
    def compute_trajectories(self, states, actions, terminations, truncations, model_names):
        """Generate a numpy array of trajectories of shape (N, timesteps, state_dim)
        Many helper functions accept this format as input

        Args:
            states (_type_): _description_
            terminations (_type_): _description_
            truncations (_type_): _description_
            model_names (_type_): _description_
        Returns:
            trajectories (_type_): (N, timesteps, state_dim)
            terminations (_type_): (N,) -> contains numbers where termination occured, incase no termination is timesteps + 1
            model_names (_type_): (N,) -> contains the model names of the trajectory
        """
        finished = terminations | truncations
        starts = np.where(np.roll(finished, 1) == 1)[0]
        ends = np.where(finished == 1)[0]

        # find largest difference between starts and ends
        horizon = np.max(ends + 1 -starts)
        trajectories = np.zeros((len(starts), horizon, states.shape[1]))
        terminations_ = np.zeros(len(starts))
        actions_ = np.zeros((len(starts), horizon, actions.shape[1]))
        model_names_ = []
        for i, (start, end) in enumerate(zip(starts, ends)):
            #print(start,end)
            #print(val_dataset.states[start:end+1].shape)
            trajectories[i, 0:end - start+ 1] =states[start:end+1]
            actions_[i,0:end-start + 1] = actions[start:end+1] 
            term = np.where(terminations[start:end+1])[0]
            if len(term )== 0:
                term = [horizon+1]

            terminations_[i] = int(term[0])
            model_names_.append(model_names[start])
        return trajectories, actions_, terminations_, np.array(model_names_)
    
    def get_target_log_probs(self, states, actions, actor, fn_unnormalize_states, scans= None, batch_size=5000,keys=None, subsample_laser=20):
        import torch
        num_batches = int(np.ceil(len(states) / batch_size))
        log_prob_list = []
        # batching, s.t. we dont run OOM
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            batch_states = states[start_idx:end_idx].clone()

            # unnormalize from the dope dataset normalization
            # print(batch_states.shape)
            batch_states_unnorm = fn_unnormalize_states(batch_states, keys=keys) # this needs batches
            del batch_states
            batch_states_unnorm = batch_states_unnorm.cpu().numpy()

            # get scans
            if scans is not None:
                laser_scan = scans[start_idx:end_idx].cpu().numpy()
            else:
                laser_scan = self.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                #print("Scan 1")
                #print(laser_scan)
                laser_scan = self.normalize_laser_scan(laser_scan)
            #print("Scan 2")
            #print(laser_scan)
            # back to dict
            #print(batch_states_unnorm.shape)
            model_input_dict = self.unflatten_batch(batch_states_unnorm, keys=keys)
            # normalize back to model input
            # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
            # now also append the laser scan
            # print(model_input_dict)
            model_input_dict['lidar_occupancy'] = laser_scan
            #print("model input dict")
            #print("after unflattening")
            #print(model_input_dict)
            log_probs = actor(
            model_input_dict,
            actions=actions[start_idx:end_idx],
            std=None)[2]
            #print(batch_actions)
            
            log_prob_list.append(log_probs)
        # tf.concat(actions_list, axis=0)
        # with torch
        # convert to torch tensor
        log_prob_list = [torch.from_numpy(prob) for prob in log_prob_list]
        log_prob = torch.concat(log_prob_list, axis=0)
        # print(actions)
        return log_prob.float()
    

    def get_target_actions(self, states, actor, fn_unnormalize_states, 
                           scans= None, action_timesteps=None, 
                           batch_size=5000,keys=None, 
                           subsample_laser=20, deterministic=False):
        import torch
        num_batches = int(np.ceil(len(states) / batch_size))
        actions_list = []
        # batching, s.t. we dont run OOM
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(states))
            batch_states = states[start_idx:end_idx].clone()

            # unnormalize from the dope dataset normalization
            # print(batch_states.shape)
            batch_states_unnorm = fn_unnormalize_states(batch_states, keys=keys) # this needs batches
            del batch_states
            batch_states_unnorm = batch_states_unnorm.cpu().numpy()

            # get scans
            if scans is not None:
                laser_scan = scans[start_idx:end_idx].cpu().numpy()
            else:
                laser_scan = self.get_laser_scan(batch_states_unnorm, subsample_laser) # TODO! rename f110env to dataset_env
                #print("Scan 1")
                #print(laser_scan)
                laser_scan = self.normalize_laser_scan(laser_scan)
            #print("Scan 2")
            #print(laser_scan)
            # back to dict
            #print(batch_states_unnorm.shape)
            model_input_dict = self.unflatten_batch(batch_states_unnorm, keys=keys)
            # normalize back to model input
            # model_input_dict = model_input_normalizer.normalize_obs_batch(model_input_dict)
            # now also append the laser scan
            # print(model_input_dict)
            model_input_dict['lidar_occupancy'] = laser_scan
            #print("model input dict")
            #print("after unflattening")
            #print(model_input_dict)
            batch_actions = actor(
            model_input_dict,
            std=None,
            deterministic=deterministic)[1]
            #print(batch_actions)
            
            actions_list.append(batch_actions)
        # tf.concat(actions_list, axis=0)
        # with torch
        # convert to torch tensor
        actions_list = [torch.from_numpy(action) for action in actions_list]
        actions = torch.concat(actions_list, axis=0)
        # print(actions)
        return actions.float()
    
    def plot_trajectories(self, trajectories, model_names, terminations, title="Agent trajectories", velocity_blobs=True, legend=True):
        """

        Args:
            trajectories (_type_): shape (N, timesteps, 2-X)
            model_names (_type_): shape (N,)
            terminations (_type_): shape (N,) # where the trajectory was terminated, if no termination then set -1
            timeouts (_type_): shape (N,) # where the trajectory was truncated (not necessary, otherwise just at end of trajectory)
        """
        from matplotlib.font_manager import FontProperties
        font = FontProperties(family='DejaVu Sans', style='normal')
        import matplotlib.cm as cm
        import matplotlib.lines as mlines
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.rcParams.update({
            'text.usetex': True,
            'font.family': 'serif',
        })
        sns.set_theme(style="white")
        map_array =  np.array(self.sim.agents[0].scan_simulator.map_img)
        resolution = self.sim.agents[0].scan_simulator.map_resolution

        origin = self.sim.agents[0].scan_simulator.origin
        plt.imshow(map_array, cmap='gray')
        # plt.show()

        assert len(trajectories) == len(model_names), f"Trajectories {len(trajectories)} and model names {len(model_names)} should have the same length"
        assert len(trajectories) == len(terminations)
        # assert len(trajectories) == len(timeouts)
        # sort the trajectories by model name, also sort terminations and timeouts the same way
        print(model_names)
        sort_indices = np.argsort(model_names)

        model_names = model_names[sort_indices]
        terminations = terminations[sort_indices]
        timeouts = np.ones((trajectories.shape[0])) * (trajectories.shape[1]-1) #timeouts[sort_indices]
        trajectories = trajectories[sort_indices]

        #model_names = ["StochasticContinousFTGAgent_0.45_3_0.5_0.03_0.1_5.0_0.3_0.5"]
        legend_handles = []
        colors = cm.rainbow(np.linspace(0, 1, len(np.unique(model_names))))
        crashes = 0
        crash_points = []
        timeout_points = []
        
        for idx, model_name in enumerate(np.unique(model_names)):
            legend_line = mlines.Line2D([], [], color=colors[idx], marker='x', linestyle='-', label=model_name)
            legend_handles.append(legend_line)
            #print(len(intersect))
            #print(model_name)
            
            model_trajectories = np.where(model_names == model_name)[0]
            for trajectory_idx in model_trajectories:
                #print(i)
                #print(np.where(root["timeouts"][i:i+252]==1))
                #print(np.where(root["terminals"][i:i+252]==1))
                #print(trajectories[trajectory_idx,0,0],trajectories[trajectory_idx,0,1])
                #print(trajectories[trajectory_idx].shape)
                pixel_poses = np.zeros((trajectories[trajectory_idx].shape[0],2))
                pixel_poses[:, 0] = (trajectories[trajectory_idx,:, 0] ) / resolution
                pixel_poses[:, 1] = ((trajectories[trajectory_idx,:, 1]) / resolution)
                
                plt.plot(pixel_poses[0,0],pixel_poses[0,1],'x',color=colors[idx], scalex=3.0)
                #print(end)
                ends = int(min(terminations[trajectory_idx], timeouts[trajectory_idx]))
                #print(ends)
                # print(ends)
                plt.plot(pixel_poses[:ends+1,0],pixel_poses[:ends+1,1],linestyle='--',color=colors[idx])
                if velocity_blobs:
                    plt.plot(pixel_poses[:ends:20, 0], pixel_poses[:ends:20, 1], 'o', color=colors[idx])

                if ends==terminations[trajectory_idx]:
                    crashes += 1
                    #plt.plot(root["observations"][i+end,0],root["observations"][i+end,1],'o',color="black", scalex=3.0)

                    crash_points.append((pixel_poses[ends], colors[idx]))
                    #plt.text(pixel_poses[ends,0], pixel_poses[ends,1], '\u2620',fontproperties=font, fontsize=16, color='black', ha='center', va='center')
                else:
                    timeout_points.append(pixel_poses[ends])
                    #plt.plot(pixel_poses[ends,0], pixel_poses[ends,1],'x',color="black", scalex=3.0)
                #break
                #plt.plot()
        for crash_point in crash_points:
            #plt.text(crash_point[0], crash_point[1], '\u2620', fontproperties=font, fontsize=16, color='black', ha='center', va='center')
            #\blacksquare
            #plt.text(crash_point[ 0], crash_point[1], r'$\blacksquare$', fontsize=16, color='black', ha='center', va='center', usetex=True)
            #lt.plot(crash_point[0], crash_point[1], 's', color="black", scalex=3.0)
            crash_point, color = crash_point
            plt.plot(crash_point[0], crash_point[1], '^', markeredgecolor='black', markerfacecolor=color, markersize=10)

        for timeout_point in timeout_points:
            plt.plot(timeout_point[0], timeout_point[1], 'x', color="black", scalex=3.0)
    # add x and y labels
        print(f"{title} Crashes/Trajectories", crashes, "/", len(trajectories))
        if legend == True:
            plt.legend(handles=legend_handles, fontsize='large', loc='upper left', bbox_to_anchor=(1, 1), title="Agents")

            plt.subplots_adjust(right=0.8)
        x_ticks_pixels = plt.xticks()[0][1:-1]
        y_ticks_pixels = plt.yticks()[0][1:-1]

        # Convert pixel ticks to meters by dividing by the resolution
        print(resolution)
        print(x_ticks_pixels)
        x_ticks_meters = x_ticks_pixels * resolution
        y_ticks_meters = y_ticks_pixels * resolution

        # Set new ticks in meters
        plt.xticks(x_ticks_pixels, [f"{tick:.2f}" for tick in x_ticks_meters])
        plt.yticks(y_ticks_pixels, [f"{tick:.2f}" for tick in y_ticks_meters])

        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.title(title)
        plt.savefig("plots/trajectories.pdf")
        plt.show()
        

    def simulate(self, agent, agent_name = None, render=True, 
                 debug=True, starting_states=None, save_data=False, 
                 num_episodes=1, episode_length=500, starting_speeds = None):
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
        from tqdm import tqdm
        if starting_states is None:
            starting_states = sh.get_start_position(self.track, start_positions=num_episodes)
        if starting_speeds is None:
            # like starting states 0.0
            starting_speeds = np.zeros((num_episodes,))
        #print(starting_states)
        starting_states = np.resize(starting_states, (num_episodes,starting_states.shape[1]))
        starting_speeds = np.resize(starting_speeds, (num_episodes,))
        #print(starting_states)
        progress = Progress(self.track, lookahead=200)
        if agent_name is None:
            agent_name = str(agent)
        log_dump = []
        obs_numpy = np.zeros((num_episodes, episode_length, len(self.keys)))

        for episode, (starting_state, starting_speed) in tqdm(enumerate(zip(starting_states, starting_speeds)), total=num_episodes):
            reset_options = dict(poses=np.array([starting_state]),velocity=np.array([starting_speed]))
            # print("reset")
            self.reset(options=reset_options)
            # print("reached")
            agent.reset()
            
            current_steer = 0.0
            current_vel = starting_speed
            # print(current_vel)

            obs, reward, done, truncated, info = self.step(np.array([[current_steer,current_vel]]))
            progress.reset(np.array([[obs["poses_x"][0], obs["poses_y"][0]]]))
            

            for timestep in range(episode_length):
                obs["previous_action_steer"] = np.array([current_steer])
                obs["previous_action_speed"] = np.array([current_vel])

                del obs["ego_idx"]
                del obs["lap_counts"]
                del obs["lap_times"]
                #print(obs["linear_vels_x"])
                # we need to add the timestep to the obs, transform the laser scan and 
                # add lidar_occupancy
                new_progress = progress.get_progress(np.array([[obs["poses_x"][0], obs["poses_y"][0]]]))
                obs["progress"] = np.array(new_progress)
                scan = obs["scans"]
                scan = self.normalize_laser_scan(scan)
                obs["lidar_occupancy"] = scan[:,::SUBSAMPLE]
                #print(obs["lidar_occupancy"].shape)
                #if timestep > 20:
                #    import matplotlib.pyplot as plt
                #    plt.plot(obs["lidar_occupancy"][0])
                #    plt.show()
                del obs["scans"]
                if self.encode_cyclic:
                    obs['progress_sin'] = np.array(np.sin(new_progress*2 * np.pi),dtype=np.float32)
                    obs['progress_cos'] = np.array(np.cos(new_progress*2 * np.pi),dtype=np.float32)
                    obs['theta_sin'] = np.array(np.sin(obs["poses_theta"]),dtype=np.float32)
                    obs['theta_cos'] = np.array(np.cos(obs["poses_theta"]),dtype=np.float32)
                    del obs["poses_theta"]
                    del obs["progress"]
                if self.include_time_obs:
                    obs["timestep"] = np.array([timestep/episode_length])
                # add progress
                #print(obs)
                # run agent
                _, action, log_prob = agent(obs) #timestep=np.array([timestep/episode_length]))
                #print(action)
                #exit()
                action_out = action.copy()
                action = action[0]
                
                if self.use_delta_actions:
                    #print("delta actions")
                    #print(action, current_steer, current_vel)
                    current_steer += action[0] 
                    current_vel += action[1]
                else:
                    current_steer = action[0]
                    current_vel = action[1]
                # clip current vel
                current_vel = np.clip(current_vel, 0.5, 7.0)
                #print("current steer", current_steer)
                #print("current vel", current_vel)
                #print("obs, linear vel", obs["linear_vels_x"])
                # do 5 steps
                del obs["collisions"]
                # create obs without lidar
                obs_no_lidar = obs.copy()
                del obs_no_lidar["lidar_occupancy"]
                obs_flattened = self.flatten_batch(obs_no_lidar)
                obs_numpy[episode, timestep, :] = obs_flattened
                

                action_raw = np.array([[current_steer, current_vel]])
                time_infos = {}
                time_infos["lidar_timestamp"] = 0.0
                time_infos["pose_timestamp"] = 0.0
                time_infos["action_timestamp"] = 0.0
                imu_data = dict()
                imu_data["lin_vel_x"] = [obs["linear_vels_x"][0]]
                imu_data["lin_vel_y"] = [0.0]
                imu_data["lin_vel_z"] = [0.0]
                imu_data["ang_vel_x"] = [0.0]
                imu_data["ang_vel_y"] = [0.0]
                imu_data["ang_vel_z"] = [0.0]
                imu_data["timestamp"] = [0.0]
                if timestep == episode_length - 1:
                    truncated = True
                if done:
                    truncated = True
                log_dump.append((action_out, obs, 0.0, done, truncated, log_prob, 
                                timestep, agent_name, done, action_raw, 
                                time_infos, imu_data))
                if done or truncated:
                    break
                for _ in range(5):
                    obs, reward, done, truncated, info = self.step(np.array([[current_steer,current_vel]]))
                    if done or truncated:
                        break
                if render:
                    self.render()

        
        return log_dump, obs_numpy # also return a nice observation numpy array             

    def termination_condition(self,scan, speed):    
        ranges = scan[17:36] *10
        results = ranges * np.cos(self.angles)
        return (results<0.3).any()

    def termination_scan(self,obs, scans, speed):
        terminations = np.zeros(len(scans), dtype=bool)

        # squeeze the scans if necessary
        if len(scans.shape) == 3:
            scans = np.squeeze(scans, axis=1)
        for i in range(len(scans)):

            terminations[i] = self.termination_condition(scans[i], speed[i])
        return terminations, None