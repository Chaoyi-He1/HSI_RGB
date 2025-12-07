import torch
import numpy as np
import itertools
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import os
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

mean_distance = np.zeros((5, 3))
# Generate all pairs of voltage combinations
for total_num in [1]:
    # load data
    data_path = 'path/rgbTable.mat'
    data = sio.loadmat(data_path)
    voltages = np.reshape(data['interp_voltage'], (-1,))[:300]
    rgb_values = data['rgb']

    # Prepare the training data: input is combined RGB, output is two voltage indices
    X_train = []  # Combined RGB values
    y_train = []  # Corresponding voltage indices
    combined_rgb_set = set()  # Set to track unique RGB combinations
    
    for pick_num in range(1, total_num + 1):
        for indices in itertools.combinations(range(len(voltages)), pick_num):
            rgb_values_selected = rgb_values[list(indices)]
            combined_rgb = np.sum(rgb_values_selected, axis=0) 
            combined_rgb = np.clip(combined_rgb, 0, 255)
            
            # Convert combined_rgb to a tuple for use in the set
            combined_rgb_tuple = tuple(combined_rgb)
            
            # Check if combined_rgb is already in the set
            if combined_rgb_tuple not in combined_rgb_set:
                combined_rgb_set.add(combined_rgb_tuple)  # Add to set for fast lookup
                X_train.append(combined_rgb)  # Add to the training list
                y_train.append(indices)  # Add the corresponding indices

    # Convert data to numpy arrays
    X_train = np.array(X_train)
    # y_train = np.array(y_train)

    # save X_train as a image
    plt.imsave('weights/rgb_recover_visual/X_train.png', X_train[:len(X_train)//128 * 128].reshape(-1, 128, 3).astype(np.uint8))

    def rgb_to_voltage(rgb_values):
        # find the closest rgb values in X_train for each rgb value in rgb_values
        # rgb_values: (N, 3) numpy array; X_train: (M, 3) numpy array
        # Returns: (N, 3) numpy array of the corresponding argmin rgb values
        
        # cut rgb_values into mini-batches to avoid memory error
        batch_size = 10000
        n_batches = len(rgb_values) // batch_size
        
        if len(rgb_values) % batch_size != 0:
            n_batches += 1
        argmin_rgb_values = []
        
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, len(rgb_values))
            argmin_rgb_values.append(X_train[np.argmin(np.linalg.norm(X_train[:, None] - rgb_values[start:end], axis=-1), axis=0)])
        return np.concatenate(argmin_rgb_values, axis=0)


    # Test the model on the test images
    test_img_list = [f for f in os.listdir('path/test') if f.endswith('.png')]
    test_img_list.sort()
    # read the test .png images and convert them to numpy arrays as [0, 255] integers
    test_imgs = [(plt.imread(os.path.join('path/test', f)) * 255).astype(np.int64) for f in test_img_list] 
    test_imgs_names = [f for f in test_img_list]

    test_img_list = [f for f in os.listdir('path/test') if f.endswith('.jpg')]
    test_img_list.sort()
    test_imgs += [plt.imread(os.path.join('path/test', f)).astype(np.int64) for f in test_img_list]
    test_imgs_names += [f for f in test_img_list]

    # Directory to save results
    output_dir = 'weights/rgb_recover_visual/predict_results_from_V'
    os.makedirs(output_dir, exist_ok=True)
    
    idx_img = 0

    for rgb_img, img_name in tqdm(zip(test_imgs, test_imgs_names), total=len(test_imgs)):
        # Remove the alpha channel if it exists
        if rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        h, w, _ = rgb_img.shape
        rgb_img = rgb_img.reshape(-1, 3)  # Normalize the RGB values
        
        # Predict the combined RGB values
        combined_rgb = rgb_to_voltage(rgb_img)
        
        combined_rgb = combined_rgb.reshape(h, w, 3)
        # Save the combined RGB image
        combined_rgb = np.clip(combined_rgb, 0, 255).astype(np.uint8)
        plt.imsave(os.path.join(output_dir, f'{img_name}.png'), combined_rgb)
        
        # Calculate the mean distance between the predicted and original RGB values
        # mean_distance[idx_img, total_num - 1] = np.mean(np.linalg.norm(rgb_img - combined_rgb, axis=-1))
        # idx_img += 1

# Save the mean distances to a .csv file with given column and row names
# mean_distance = np.round(mean_distance, 2)
# np.savetxt('weights/rgb_recover_visual/mean_distance.csv', mean_distance, delimiter=',', header='1_volt,2_volt,3_volt', comments='', fmt='%s')
