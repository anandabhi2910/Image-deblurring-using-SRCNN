import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm



class DeblurCNN(nn.Module):
    def __init__(self):
        super(DeblurCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 1024, kernel_size=9, padding=2)
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=1, padding=2)
        self.conv3 = nn.Conv2d(1024, 3, kernel_size=5, padding=2)
    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.relu(self.conv2(x))
        x = self.conv3(x)
        return x



# Define a custom transform to convert BGR to RGB
class BGRToRGB(object):
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Define transforms
transform = transforms.Compose([
    BGRToRGB(),  # Convert BGR to RGB
    transforms.ToPILImage(),
    # transforms.Resize((448, 256)),
    transforms.ToTensor(),
])



# Define dataset class for testing
class TestDataset(Dataset):
    def __init__(self, blur_dir, blur_paths, transforms=None):
        self.blur_dir = blur_dir
        self.X = blur_paths
        self.transforms = transforms
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        filename = self.X[i]
        blur_image = cv2.imread(os.path.join(self.blur_dir, filename))
        if self.transforms:
            blur_image = self.transforms(blur_image)
        return blur_image, filename  # returning only the filename


# Load saved model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DeblurCNN().to(device)
model.load_state_dict(torch.load('/home/ayushh/deblur/model_abhi.pth'))
model.eval()


# Load test data
test_blur_dir = '/home/ayushh/deblur/mp2_test/custom_test/blur'
test_images = os.listdir(test_blur_dir)
test_images.sort()
test_data = TestDataset(test_blur_dir, test_images, transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)


# Prediction loop
output_dir = '/home/ayushh/deblur/test_deblurred'
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for blur_image, filename in tqdm(test_loader, total=len(test_data)):
        blur_image = blur_image.to(device)
        output = model(blur_image)
        filename = filename[0]  # extract the filename from the tuple
        output_image = transforms.ToPILImage()(output.squeeze(0).cpu()).convert("RGB")
        output_image.save(os.path.join(output_dir, filename))