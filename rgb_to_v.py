import torch
import numpy as np
import itertools
import scipy.io as sio
import torch.nn as nn
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import torch.nn.functional as F


# load data
data_path = 'path/rgbTable.mat'
data = sio.loadmat(data_path)
voltages = np.reshape(data['interp_voltage'], (-1,))[:300]
rgb_values = data['rgb']

# Prepare the training data: input is combined RGB, output is two voltage indices
X_train = []  # Combined RGB values
y_train = []  # Corresponding voltage indices

# Generate all pairs of voltage combinations
for (i, j, k) in itertools.combinations(range(len(voltages)), 3):
    rgb_x = rgb_values[i]
    rgb_y = rgb_values[j]
    rgb_z = rgb_values[k]
    
    # Combine RGB values (50/50 mix)
    combined_rgb = rgb_x + rgb_y + rgb_z
    combined_rgb = np.clip(combined_rgb, 0, 255)  # Clip to valid RGB range
    
    # Store the combined RGB and the corresponding voltage indices
    X_train.append(combined_rgb)
    y_train.append([i, j, k])  # Store the indices of Vx and Vy

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 256)  # Input layer: RGB (3 units)
        self.fc2 = nn.Linear(256, 1024)  # Hidden layer 1
        self.fc3 = nn.Linear(1024, 1024)  # Hidden layer 2
        self.fc4 = nn.Linear(1024, 512)  # Hidden layer 2
        self.fc5 = nn.Linear(512, 3)  # Output layer: 2 units (Vx index and Vy index)

    def forward(self, x):
        x = F.silu(self.fc1(x))
        x = F.silu(self.fc2(x))
        x = F.silu(self.fc3(x))
        x = F.silu(self.fc4(x))
        x = self.fc5(x)
        return x

# Instantiate the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MLP().to(device)

# Define the loss function (MSE loss) and the optimizer (Adam optimizer)
criterion = nn.MSELoss().to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001)

# Set training parameters
num_epochs = 1000
batch_size = 128

# Training loop
model.train()  # Set the model to training mode
for epoch in range(num_epochs):
    # Shuffle the training data
    indices = torch.randperm(X_train_tensor.size(0))
    X_train_tensor_shuffled = X_train_tensor[indices]
    y_train_tensor_shuffled = y_train_tensor[indices]

    # Mini-batch training
    loss_total = 0
    for i in range(0, X_train_tensor.size(0), batch_size):
        # Get mini-batch data
        inputs = X_train_tensor_shuffled[i:i+batch_size].to(device)
        labels = y_train_tensor_shuffled[i:i+batch_size].to(device)

        # Forward pass
        outputs = model(inputs)

        # Compute the loss
        loss = criterion(outputs, labels)
        loss_total += loss.item()

        # Zero gradients, backward pass, and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # print(f'Epoch {epoch+1}/{num_epochs}, Batch {i//batch_size+1}, Loss: {loss.item()}', flush=True)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss Total: {loss_total / (X_train_tensor.size(0) // batch_size)}')

# Function to predict the best voltage pair for a new RGB value
def predict_best_voltages(new_rgb):
    # Convert the new RGB to a PyTorch tensor
    new_rgb_tensor = torch.tensor(new_rgb, dtype=torch.float32).to(device)
    
    # Set the model to evaluation mode and predict
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # No need to calculate gradients during evaluation
        predicted_indices = model(new_rgb_tensor)

    # Round the predicted indices to the nearest integers (since we're predicting indices)
    Vx_idx = torch.round(predicted_indices[:, 0]).to(torch.int32).cpu().numpy()
    Vy_idx = torch.round(predicted_indices[:, 1]).to(torch.int32).cpu().numpy()
    Vz_idx = torch.round(predicted_indices[:, 2]).to(torch.int32).cpu().numpy()
    Vx_idx = np.clip(Vx_idx, 0, len(voltages) - 1)  # Clip to valid index range
    Vy_idx = np.clip(Vy_idx, 0, len(voltages) - 1)  # Clip to valid index range
    Vz_idx = np.clip(Vz_idx, 0, len(voltages) - 1)  # Clip to valid index range
    
    # Get the actual voltages corresponding to these indices
    Vx = voltages[Vx_idx]
    Vy = voltages[Vy_idx]
    Vz = voltages[Vz_idx]
    
    # get the RGB values corresponding to the predicted voltages
    rgb_x = rgb_values[Vx_idx]
    rgb_y = rgb_values[Vy_idx]
    rgb_z = rgb_values[Vz_idx]
    combined_rgb = rgb_x + rgb_y + rgb_z
    
    return Vx, Vy, Vz, combined_rgb

test_img_list = [f for f in os.listdir('weights/rgb_recover_visual/predict_results') if f.endswith('.mat') and "HVI" in f]
test_imgs = [sio.loadmat(os.path.join('weights/rgb_recover_visual/predict_results', f))['pred_rgb'] for f in test_img_list]
test_imgs_names = [f.replace('HVI', '') for f in test_img_list]

# test the model on the test images
for rgb_img, img_name in zip(test_imgs, test_imgs_names):
    h, w, _ = rgb_img.shape
    rgb_img = rgb_img.reshape(-1, 3)  # Normalize the RGB values
    Vx, Vy, Vz, combined_rgb = predict_best_voltages(rgb_img)
    
    combined_rgb = combined_rgb.reshape(h, w, 3)
    # Save the combined RGB image
    combined_rgb = np.clip(combined_rgb, 0, 255)
    combined_rgb = combined_rgb.astype(np.uint8)
    os.makedirs('weights/rgb_recover_visual/predict_results_from_V', exist_ok=True)
    plt.imsave(f'weights/rgb_recover_visual/predict_results_from_V/{img_name}.png', combined_rgb)
    