import argparse
import f110_gym
import f110_orl_dataset
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description='Your script description')




if __name__ == "__main__":
    F110Env = gym.make('f110-real-v1',
                   encode_cyclic=False,
                   flatten_obs=True,
                   timesteps_to_include=(0,250),
                    use_delta_actions=True, # control if actions are deltas or absolute
                   reward_config="reward_progress.json",
        **dict(name='f110-real-v1',
            config = dict(map="Infsaal2", num_agents=1),
              render_mode="human")
    ) 
    dataset = F110Env.get_dataset(only_agents="pure_pursuit2_0.8_1.2_raceline_og_3_0.6")
    poses_x = dataset["observations"][:30,0]
    poses_y = dataset["observations"][:30,1]
    timestep_to_check = 45
    timesteps = dataset["infos"]["pose_timestamp"][:30]
    # denote with o's the points
    #plt.plot(poses_x, poses_y, "-o")
    #plt.plot(poses_x, poses_y, "--")
    #plt.show()
    #plt.plot(timesteps)
    #plt.show()
    timesteps_test = 55
    poses_x = dataset["observations"][:timesteps_test, 0]
    poses_y = dataset["observations"][:timesteps_test, 1]
    poses_theta = dataset["observations"][:timesteps_test,2]
    timesteps = dataset["infos"]["pose_timestamp"][:timesteps_test]
    from scipy.interpolate import UnivariateSpline
    # Normalize timestamps to start from zero
    timesteps_normalized = timesteps - timesteps[0]

    # Create splines for x and y
    spline_x = UnivariateSpline(timesteps_normalized, poses_x, s=0.02) # s is the smoothing factor
    spline_y = UnivariateSpline(timesteps_normalized, poses_y, s=0.02)

    # Generate smoothed positions
    smoothed_x = spline_x(timesteps_normalized)
    smoothed_y = spline_y(timesteps_normalized)

    # Plot original and smoothed data for comparison
    """
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(timesteps_normalized, poses_x, 'o', label='Original X')
    plt.plot(timesteps_normalized, smoothed_x, label='Smoothed X')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(timesteps_normalized, poses_y, 'o', label='Original Y')
    plt.plot(timesteps_normalized, smoothed_y, label='Smoothed Y')
    plt.legend()

    plt.show()
    """
    action_timestep =dataset["infos"]["action_timestamp"][:timesteps_test]
    normalized_action_timestep = action_timestep - timesteps[0]

    interpolated_x = spline_x(normalized_action_timestep)
    interpolated_y = spline_y(normalized_action_timestep)

    print("Interpolated X at timestep", normalized_action_timestep, "is:", interpolated_x)
    print("Interpolated Y at timestep", normalized_action_timestep, "is:", interpolated_y)
    
    plt.plot(interpolated_x, interpolated_y, "-o", label="Interpolated")
    plt.plot(poses_x, poses_y, '-o', label='Original X')
    plt.plot(poses_x[timestep_to_check], poses_y[timestep_to_check], 'o', scalex=3.0, color="red")
    plt.plot(interpolated_x[timestep_to_check], interpolated_y[timestep_to_check], 'o', scalex=3.0, color="black")
    plt.legend()
    plt.show()
    plt.plot(action_timestep)
    plt.plot(timesteps)
    plt.legend(["action_timestep", "pose_timestep"])
    plt.show()

    plt.plot(action_timestep - timesteps)
    plt.show()
    
    # Calculate errors
    error_x = poses_x - spline_x(timesteps_normalized)
    error_y = poses_y - spline_y(timesteps_normalized)

    # Plot histogram for x errors
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(error_x, bins='auto', color='red', alpha=0.7)
    plt.title('Histogram of X Errors')
    plt.xlabel('Error in X')
    plt.ylabel('Frequency')
    plt.grid(True)

    # Plot histogram for y errors
    plt.subplot(1, 2, 2)
    plt.hist(error_y, bins='auto', color='green', alpha=0.7)
    plt.title('Histogram of Y Errors')
    plt.xlabel('Error in Y')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    """

    dx = spline_x.derivative()(normalized_action_timestep)
    dy = spline_y.derivative()(normalized_action_timestep)

    # Calculate angles from the derivatives
    # Note: np.arctan2 handles the quadrant of the angle correctly
    calculated_theta = np.arctan2(dy, dx)

    # Calculate error in theta
    # It's important to handle the periodicity of angles
    print(poses_theta)
    print(calculated_theta)
    theta_error = np.unwrap(poses_theta - calculated_theta)

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import norm

    # Calculate mean and standard deviation for theta error
    mean_theta_error = np.mean(theta_error)
    std_theta_error = np.std(theta_error)

    # Generate values for the fitted normal distribution
    theta_error_fit = np.linspace(theta_error.min(), theta_error.max(), 100)
    theta_error_pdf = norm.pdf(theta_error_fit, mean_theta_error, std_theta_error)

    # Plot histogram and the fitted normal distribution
    plt.hist(theta_error, bins='auto', density=True, alpha=0.6, color='purple')
    plt.plot(theta_error_fit, theta_error_pdf, 'k--')
    plt.title('Theta Error with Fitted Normal Distribution')
    plt.xlabel('Error in Theta (radians)')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

    # Calculate mean and standard deviation for x position error
    mean_error_x = np.mean(error_x)
    std_error_x = np.std(error_x)

    # Generate values for the fitted normal distribution
    error_x_fit = np.linspace(error_x.min(), error_x.max(), 100)
    error_x_pdf = norm.pdf(error_x_fit, mean_error_x, std_error_x)

    # Plot histogram and the fitted normal distribution for x error
    plt.hist(error_x, bins='auto', density=True, alpha=0.6, color='red')
    plt.plot(error_x_fit, error_x_pdf, 'k--')
    plt.title('X Position Error with Fitted Normal Distribution')
    plt.xlabel('Error in X')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

    # Calculate mean and standard deviation for y position error
    mean_error_y = np.mean(error_y)
    std_error_y = np.std(error_y)

    # Generate values for the fitted normal distribution
    error_y_fit = np.linspace(error_y.min(), error_y.max(), 100)
    error_y_pdf = norm.pdf(error_y_fit, mean_error_y, std_error_y)

    # Plot histogram and the fitted normal distribution for y error
    plt.hist(error_y, bins='auto', density=True, alpha=0.6, color='green')
    plt.plot(error_y_fit, error_y_pdf, 'k--')
    plt.title('Y Position Error with Fitted Normal Distribution')
    plt.xlabel('Error in Y')
    plt.ylabel('Density')
    plt.grid(True)
    plt.show()

    import numpy as np

    # Number of samples to generate
    num_samples = 100

    mean_theta_error = mean_theta_error
    std_theta_error = std_theta_error#0.2

    # Generate samples for x, y, and theta errors
    samples_error_x = np.random.normal(mean_error_x, std_error_x, num_samples)
    samples_error_y = np.random.normal(mean_error_y, std_error_y, num_samples)
    samples_theta_error = np.random.normal(mean_theta_error, std_theta_error, num_samples)
    # now lets load an agent and test if for different samples and then lets fit a normal distribution to it
    # first lets pick 10 samples from the dataset

    sample1 = normalized_action_timestep[timestep_to_check]
    # get the x and y position
    x1 = spline_x(sample1)
    y1 = spline_y(sample1)
    theta1 = np.arctan2(dy, dx)[timestep_to_check]
    # now lets add the error in a loop
    print(samples_error_x)
    print(x1)
    print(y1)
    xs = []
    ys = []
    thetas = []
    for i in range(num_samples):
        # add the error
        xs.append(x1 + samples_error_x[i])
        ys.append(y1 + samples_error_y[i])
        thetas.append(theta1 + samples_theta_error[i])
        # now lets plot the position
    plt.plot(xs, ys, "o")
    # plot the thetas as arrows at the xs
    plt.plot(x1, y1, "o", color="red")
    plt.plot(poses_x[timestep_to_check], poses_y[timestep_to_check], "o", color="black")
    plt.quiver(xs, ys, np.cos(thetas), np.sin(thetas))
    plt.quiver(x1, y1, np.cos(theta1), np.sin(theta1), color="red")
    plt.quiver(poses_x[timestep_to_check], poses_y[timestep_to_check], np.cos(poses_theta[timestep_to_check]), np.sin(poses_theta[timestep_to_check]), color="black")
    # now we execute the agent for each of there positions
    # print()
    plt.show()
    model = dataset["model_name"][0]
    print(model)
    from f110_agents.agent import Agent
    actor = Agent().load(f"/home/fabian/msc/f110_dope/ws_release/config_1501/config/agent_configs/pure_pursuit_0.6_1.0_raceline_og_3.json")
    start = normalized_action_timestep[timestep_to_check]
    end = normalized_action_timestep[timestep_to_check-1]
    # sample 10 points between start and end
    sampled = np.linspace(start, end, 10)
    print(sampled)
    x_vals = spline_x(sampled)
    y_vals = spline_y(sampled)
    action_speed = []
    action_steering = []
    xss = []
    yss = []
    thetass = []
    for sample in sampled:
        curr_obs = dataset["observations"][timestep_to_check]
        # add fake batch dim
        curr_obs = np.expand_dims(curr_obs, axis=0)
        unflattened = F110Env.unflatten_batch(curr_obs)
        # print(unflattened)
        _, action_og, _ = actor(unflattened)
        print(action_og)
        action_og = [dataset["actions"][timestep_to_check]]
        print(action_og)

        # print the previous_action steer and speed
        print(unflattened["previous_action_steer"])
        print(unflattened["previous_action_speed"])
        for i in range(num_samples):
            x = spline_x(sample)
            y = spline_y(sample)
            theta = theta1
            # now add the error
            x += np.random.normal(mean_error_x, std_error_x, 1)[0]
            y += np.random.normal(mean_error_y, std_error_y, 1)[0]
            theta += samples_theta_error[i]
            unflattened['poses_x'] = np.array([x])
            unflattened['poses_y'] = np.array([y])
            unflattened['poses_theta'] = np.array(theta)
            xss.append(x)
            yss.append(y)
            thetass.append(theta)
            # print(unflattened)
            _, action, _ = actor(unflattened)
            action_speed.append(action[0][1])
            action_steering.append(action[0][0])
    plt.plot(xss, yss, "o")
    # plot the thetas as arrows at the xs
    plt.plot(spline_x(sampled),spline_y(sampled) , "o", color="red")
    for i in range(-5,5):
        plt.plot(poses_x[timestep_to_check+i], poses_y[timestep_to_check+i], "o", color="black")
        plt.quiver(poses_x[timestep_to_check+i], poses_y[timestep_to_check+i], np.cos(poses_theta[timestep_to_check+i]), np.sin(poses_theta[timestep_to_check+i]), color="black")
    plt.quiver(xss, yss, np.cos(thetass), np.sin(thetass))
    # plt.quiver(spline_x(sample), spline_y(sample), np.cos(theta1), np.sin(theta1), color="red")
    
    # now we execute the agent for each of there positions
    # print()
    plt.show()
    mean_action_speed = np.mean(action_speed)
    std_action_speed = np.std(action_speed)

    # Generate values for the fitted normal distribution
    action_speed_fit = np.linspace(min(action_speed), max(action_speed), 100)
    action_speed_pdf = norm.pdf(action_speed_fit, mean_action_speed, std_action_speed)

    mean_action_steering = np.mean(action_steering)
    std_action_steering = np.std(action_steering)
    print(mean_action_steering)
    print(std_action_steering)
    action_steering_fit = np.linspace(min(action_steering), max(action_steering), 100)
    action_steering_pdf = norm.pdf(action_steering_fit, mean_action_steering, std_action_steering)

    # Create a figure with 2 subplots
    plt.figure(figsize=(12, 6))

    # Plot histogram for action_steering
    plt.subplot(1, 2, 1)
    plt.hist(action_steering, bins='auto', density=True, alpha=0.6, color='green')
    plt.plot(action_steering_fit, action_steering_pdf, 'k--')
    plt.axvline(action_og[0][0], color='red', linestyle='dashed', linewidth=2)
    plt.title('Action Steering Histogram')
    plt.xlabel('Action Steering')
    plt.ylabel('Density')

    # Plot histogram and fitted normal distribution for action_speed
    plt.subplot(1, 2, 2)
    plt.hist(action_speed, bins='auto', density=True, alpha=0.6, color='blue')
    plt.plot(action_speed_fit, action_speed_pdf, 'k--')
    plt.axvline(action_og[0][1], color='red', linestyle='dashed', linewidth=2)
    plt.title('Action Speed with Fitted Normal Distribution')
    plt.xlabel('Action Speed')
    plt.ylabel('Density')

    # Show the plot
    plt.tight_layout()
    plt.show()