import numpy as np


def get_start_position(track, start_positions:int=25):
    cl_x = track.centerline.xs
    cl_y = track.centerline.ys
    track_length = len(track.centerline.xs)
    start_indices = np.linspace(0, track_length, start_positions, dtype=int, endpoint=False)
    xy = np.vstack([cl_x[start_indices], cl_y[start_indices]]).T
    xy_next = np.vstack([cl_x[(start_indices+1)%track_length], cl_y[(start_indices+1)%track_length]]).T
    theta = np.arctan2(xy_next[:,1] - xy[:,1],
                                xy_next[:,0] - xy[:,0])
    reset_poses = np.hstack([xy, theta[:,None]])
    print(reset_poses)
    return reset_poses

