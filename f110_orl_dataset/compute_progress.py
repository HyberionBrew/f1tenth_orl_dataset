import numpy as np
from typing import Tuple




class Progress:
    def __init__(self, track, lookahead: int = 200) -> None:
        # 
        xs = track.centerline.xs
        ys = track.centerline.ys
        self.centerline = np.stack((xs, ys), axis=-1)
        # append first point to end to make loop
        self.centerline = np.vstack((self.centerline, self.centerline[0]))

        self.segment_vectors = np.diff(self.centerline, axis=0)
        
        # print(segment_vectors.shape)
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        assert (self.segment_lengths > 0.0).all()
        # Extend segment lengths to compute cumulative distance
        self.cumulative_lengths = np.hstack(([0], np.cumsum(self.segment_lengths)))
        self.previous_closest_idx = 0
        self.max_lookahead = lookahead
        #print(self.centerline)
        #print(self.centerline.shape)
        #print("***********")

    def distance_along_centerline_np(self, pose_points):
        assert len(pose_points.shape) == 2 and pose_points.shape[1] == 2

        # centerpoints = np.array(centerpoints)
        #print(self.centerline.shape)
        #print(centerpoints[:-1])
        #print(pose_points)
        #print(".....")
        # assert pose points must be Nx2
        pose_points = np.array(pose_points)
        #print(pose_points.shape)
        #print(pose_points)
        #print("distance calc")
        #print(self.previous_closest_idx)
        def projected_distance(pose):
            rel_pose = pose - self.centerline[:-1]
            
            t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
            #print("sum of segment vectors")
            #print(self.segment_vectors)
            #print(np.sum(self.segment_vectors**2, axis=1))
            #print('i')
            t = np.clip(t, 0, 1)
            projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            points_len = self.centerline.shape[0]-1  # -1 because of last fake 
            lookahead_idx = (self.max_lookahead + self.previous_closest_idx) % points_len
            # wrap around
            if self.previous_closest_idx <= lookahead_idx:
                indices_to_check = list(range(self.previous_closest_idx, lookahead_idx + 1))
            else:
                # Otherwise, we need to check both the end and the start of the array
                indices_to_check = list(range(self.previous_closest_idx, points_len)) \
                    + list(range(0, lookahead_idx+1))
            # Extract the relevant distances using fancy indexing
            subset_distances = distances[indices_to_check]

            # Find the index of the minimum distance within this subset
            subset_idx = np.argmin(subset_distances)

            # Translate it back to the index in the original distances array
            closest_idx = indices_to_check[subset_idx]
            self.previous_closest_idx = closest_idx
            # print(closest_idx)
            return self.cumulative_lengths[closest_idx] + self.segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    
    # TODO is this not wrong? (the tuple)
    def get_progress(self, pose: Tuple[float, float]):
        #print("---get")
        #print(pose)
        progress =  self.distance_along_centerline_np(pose)
        # print(self.cumulative_lengths.shape)
        # print(self.cumulative_lengths[-1])
        #print('progress 1', progress)
        progress = progress / (self.cumulative_lengths[-1] + self.segment_lengths[-1])
        #print('progress 2', progress)
        #print(self.segment_vectors)
        # clip between 0 and 1 (it can sometimes happen that its slightly above 1)
        # print("progress", progress)
        return np.clip(progress, 0, 1)
    # input shape: tuple of (x,y)
    def reset(self, pose):
        # print(pose)
        rel_pose = pose - self.centerline[:-1]
        t = np.sum(rel_pose * self.segment_vectors, axis=1) / np.sum(self.segment_vectors**2, axis=1)
        t = np.clip(t, 0, 1)
        projections = self.centerline[:-1] + t[:, np.newaxis] * self.segment_vectors
        distances = np.linalg.norm(pose - projections, axis=1)
        
        closest_idx = np.argmin(distances)
        self.previous_closest_idx = closest_idx

class Raceline():
    def __init__(self):
        self.xs = []
        self.ys = []
        self.vxs = []

class Track():
    def __init__(self, file):
        self.track = self.load_track(file)
        self.centerline = Raceline()
        self.centerline.xs = self.track[:,0]
        self.centerline.ys = self.track[:,1]

        self.centerline.vxs = self.track[:,2]

    def load_track(self,file):
        # open and read in the track, a csv with x_m, x_y, vx_mps, delta_rad 
        
        track = np.loadtxt(file, delimiter=',')
        return track


if __name__ == "__main__":
    import argparse
    import zarr
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    # add argparser

    parser = argparse.ArgumentParser(description='Your script description')
    parser.add_argument('--path', type=str, default="dataset.zarr", help="dataset name")
    parser.add_argument('--track_path', type=str, default="Infsaal_centerline.csv", help="track name")
    args = parser.parse_args()
    import gymnasium as gym
    import f110_orl_dataset
    F110Env = gym.make('f110-real-v0',
    # only terminals are available as of tight now 
        **dict(name='f110-real-v0',
            config = dict(map="Infsaal", num_agents=1),
            render_mode="human")
    )
    root = zarr.open(args.path, mode='wr')
    track = Track(args.track_path)
    progress = Progress(track, lookahead=200)

    # now lets loop over the dataset and compute the progress
    #pose_0 = np.array(root['observations']["poses_x"][0],root['observations']["poses_y"][0])
    # the above as a lambda function for i being the index
    pose = lambda i: np.array([(root['observations']["poses_x"][i],root['observations']["poses_y"][i])])
    print(pose(0))
    progress.reset(pose(0))
    progr = []
    first = 0


    root["observations"]["progress"] = np.zeros_like(root['observations']["poses_x"])
    for i in tqdm(range(len(root['observations']["poses_x"]))):
        #print(progress.get_progress(pose(i)))
        curr_progress = progress.get_progress(pose(i))
        curr_angle = curr_progress * 2 * np.pi
        #progr.append(curr_angle)
        root["observations"]["progress_sin"][i] = np.sin(curr_angle)
        root["observations"]["progress_cos"][i] = np.cos(curr_angle)
        root["observations"]["progress"][i] = curr_progress
        if root["done"][i] or root["truncated"][i]:
            #print(root["done"][i], root["truncated"][i])
            if i < len(root['observations']["poses_x"])-1:
                progress.reset(pose(i+1))
            """
            plt.plot(root["observations"]["progress"][first:first+250])
            plt.show()
            plt.scatter(root['observations']["poses_x"][first:first+250], root['observations']["poses_y"][first:first+250])
            # plot x at start
            plt.scatter(root['observations']["poses_x"][first], root['observations']["poses_y"][first], color="red")
            plt.show()
            print(i)
            first = i+1
            if i > 300:
                exit()
            """
            #plt.plot(np.sin(np.array(progr)), label="new_sin")
            #plt.plot(np.cos(np.array(progr)), label="new_cos")
            #plt.legend()
            #plt.show()
            #plt.scatter(root['observations']["poses_x"][i:i+249], root['observations']["poses_y"][i:i+249])
            #plt.show()
    print(np.array(root["observations"]["progress"]))
    plt.plot(np.array(root["observations"]["progress"]))
    plt.show()