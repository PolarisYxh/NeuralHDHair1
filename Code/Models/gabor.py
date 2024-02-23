import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image
from torchvision.utils import save_image
def create_gabor_filter_bank(n_orientations, size, sigma, lambd, gamma):
    filters = []
    
    for theta in range(n_orientations):
        theta = theta / n_orientations * np.pi
        kernel = cv2.getGaborKernel((size, size), sigma, theta, lambd, gamma)
        filters.append(kernel)
    
    return np.array(filters)

class GaborFilter(nn.Module):
    def __init__(self, n_orientations, size, sigma, lambd, gamma):
        super(GaborFilter, self).__init__()
        
        self.n_orientations = n_orientations
        self.size = size
        self.sigma = sigma
        self.lambd = lambd
        self.gamma = gamma
        
        # self.conv_layer = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        # Create the Gabor filter bank
        filter_bank = create_gabor_filter_bank(n_orientations, size, sigma, lambd, gamma)
        self.register_buffer('filter_bank', torch.from_numpy(filter_bank).float())
    
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        orientations = self.n_orientations
        
        # Reshape the input tensor for convolution
        x_reshaped = x.view(1, batch_size * num_channels, height, width)
        self.filter_bank = self.filter_bank[:,None,:,:]
        outputs = []
        for filter in self.filter_bank:
            filtered_image = F.conv2d(x_reshaped, filter.unsqueeze(0), padding=[self.size//2])
            save_image(filtered_image,"2.png")
            outputs.append(filtered_image)

        # 合并输出结果
        orientations = torch.cat(outputs, dim=1)

        # 取模和归一化
        orientations = torch.sqrt(torch.sum(orientations**2, dim=1))
        orientations /= torch.max(orientations)
        # # Apply the Gabor filters
        # filtered_responses = F.conv2d(x_reshaped, self.filter_bank, padding=[self.size//2], stride=[1], dilation=[1],groups=batch_size * num_channels)
        # # Reshape the filtered responses
        # filtered_responses = filtered_responses.view(batch_size, num_channels, orientations, height, width)
        
        return filtered_responses
def preprocess_image(image_path):
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((256, 256))  # Resize if necessary
    
    tensor = TF.to_tensor(image)
    tensor = tensor.unsqueeze(0)  # Add a batch dimension
    
    return tensor


# Define the Gabor filter parameters
n_orientations = 8
size = 5
sigma = 5
lambd = 10
gamma = 0.5

# Load and preprocess the image
image_path = '/home/yxh/Documents/company/NeuralHDHair/data/Train_input1/img1/0bb661d0f8b65162465e8c098af3123d.png'
image_tensor = preprocess_image(image_path)

# Create the Gabor filter module
gabor_filter = GaborFilter(n_orientations, size, sigma, lambd, gamma)

# Apply the Gabor filters to the image
filtered_responses = gabor_filter(image_tensor)

# Compute the orientation from the filtered responses
orientations = torch.argmax(torch.sum(filtered_responses, dim=3), dim=2)

# Print the orientations
print(orientations)

