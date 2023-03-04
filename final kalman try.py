"""
import numpy as np
from scipy.optimize import linear_sum_assignment
import read_depth as rd
def extended_kalman_filter(depth_measured, depth_ground_truth,
                           state_mean_prior, state_covariance_prior,
                           transition_matrix, observation_matrix,
                           transition_covariance, observation_covariance):
    # Initialize state vector and covariance matrix
    state_mean = state_mean_prior
    state_covariance = state_covariance_prior

    # Initialize depth map with ground truth depth
    depth_estimate = depth_ground_truth.copy()

    # Iterate over each pixel in the depth maps
    for i in range(depth_measured.shape[0]):
        for j in range(depth_measured.shape[1]):
            # Predict state mean and covariance
            state_mean_predicted = np.dot(transition_matrix.T, state_mean)
            state_covariance_predicted = np.dot(transition_matrix, np.dot(state_covariance.T, transition_matrix)) + transition_covariance

            # Compute Kalman gain
            kalman_gain = np.dot(state_covariance_predicted, np.dot(observation_matrix.T,
                                                                    np.linalg.inv(np.dot(observation_matrix, np.dot(state_covariance_predicted.T, observation_matrix)) + observation_covariance)))

            #
        # Update state mean and covariance
            state_mean = state_mean_predicted + np.dot(kalman_gain, (depth_measured[i, j] - np.dot(observation_matrix, state_mean_predicted)))
            state_covariance = np.dot((np.eye(state_mean.shape[0]) - np.dot(kalman_gain, observation_matrix)), state_covariance_predicted)

    # Update depth estimate
            depth_estimate[i, j] = state_mean[0]

    return depth_estimate


import numpy as np

# Set up the parameters for the EKF
state_mean_prior = np.zeros((375, 1242))  # Prior mean for the state vector (initial depth estimate)
state_covariance_prior = np.ones((375, 1242))  # Prior covariance for the state vector
transition_matrix = np.ones((375, 1242))  # Transition matrix for the state model
observation_matrix = np.ones((375, 1242))  # Observation matrix for the measurement model
transition_covariance = np.ones((375, 1242)) * 0.1  # Covariance matrix for the state model noise
observation_covariance = np.ones((375, 1242)) * 0.5  # Covariance matrix for the measurement model noise

# Generate some example depth maps
depth_measured = depth_map = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\depth maps generated\0000000005.png")
depth_ground_truth = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\lidar ground truth\0000000005.png")
print(depth_measured.shape)

# Run the EKF algorithm
depth_estimate = extended_kalman_filter(depth_measured, depth_ground_truth,
                                        state_mean_prior, state_covariance_prior,
                                        transition_matrix, observation_matrix,
                                        transition_covariance, observation_covariance)

print(depth_estimate)  # Output: estimated depth map of size 375x1242
"""
import numpy as np

def extended_kalman_filter(depth_measured, depth_ground_truth,
                           state_mean_prior, state_covariance_prior,
                           transition_matrix, observation_matrix,
                           transition_covariance, observation_covariance):
    # Initialize depth map with ground truth depth
    depth_estimate = depth_ground_truth.copy()

    # Iterate over each pixel in the depth maps
    for i in range(depth_measured.shape[0]):
        for j in range(depth_measured.shape[1]):
            # Initialize state vector and covariance matrix
            state_mean = state_mean_prior[i, j]
            state_covariance = state_covariance_prior[i, j]

            # Predict state mean and covariance
            state_mean_predicted = np.dot(transition_matrix, state_mean)
            state_covariance_predicted = np.dot(transition_matrix, np.dot(state_covariance, transition_matrix.T)) + transition_covariance

            # Compute Kalman gain
            kalman_gain = np.dot(state_covariance_predicted, np.dot(observation_matrix.T,
                                                                    np.linalg.inv(np.dot(observation_matrix, np.dot(state_covariance_predicted.T, observation_matrix)) + observation_covariance)))

            # Update state mean and covariance
            state_mean = state_mean_predicted + np.dot(kalman_gain, (depth_measured[i, j] - np.dot(observation_matrix, state_mean_predicted)))
            state_covariance = np.dot((np.eye(state_mean.shape[0]) - np.dot(kalman_gain, observation_matrix)), state_covariance_predicted)

            # Update depth estimate
            depth_estimate[i, j] = state_mean[0]

    return depth_estimate


import numpy as np
import read_depth as rd
import cv2
# Set up the parameters for the EKF
state_mean_prior = np.zeros((375, 1242))  # Prior mean for the state vector (initial depth estimate)
state_covariance_prior = np.ones((375, 1242))  # Prior covariance for the state vector
transition_matrix = np.ones((1, 1))  # Transition matrix for the state model
observation_matrix = np.ones((1, 1))  # Observation matrix for the measurement model
transition_covariance = np.ones((1, 1)) * 1  # Covariance matrix for the state model noise
observation_covariance = np.ones((1, 1)) * 0.05 # Covariance matrix for the measurement model noise
# Generate some example depth maps
depth_measured = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\depth maps generated\0000000005.png")
depth_ground_truth = rd.depth_read(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\lidar ground truth\0000000005.png")
print(depth_measured.shape)
# Generate some example depth maps
#depth_measured = np.random.rand(375, 1242)  # Measured depth map
#depth_ground_truth = np.random.rand(375, 1242)  # Ground truth depth map

# Run the EKF algorithm
depth_estimate = extended_kalman_filter(depth_measured, depth_ground_truth,
                                        state_mean_prior, state_covariance_prior,
                                        transition_matrix, observation_matrix,
                                        transition_covariance, observation_covariance)

print(depth_estimate)
depth_uint16 =  (depth_estimate * 256.0).astype(np.uint16)
cv2.imwrite(r"D:\dataspell project\FinalYearProject\FinalYearProject\2011_09_26_drive_0001_syncimage_02\Fused depth maps\0000000005.png", depth_uint16)
a=depth_estimate# Output: estimated depth map of size (375, 1242)
