import os
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import save_image


# os.makedirs('/kaggle/working/gaussian_blurred', exist_ok=True)
src_dir = '/home/ayushh/train_sharp_resized'
images = os.listdir(src_dir)
dst_dir = '/home/ayushh/train_blurred/train_blurred'



def save_decoded_image(img, name):
    img = img.view(img.size(0), 3, 448, 256)
    save_image(img, name)


image_dir = '/home/ayushh/deblur/saved_images'
os.makedirs(image_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


batch_size = 2

gauss_blur = os.listdir('/home/ayushh/train_blurred/train_blurred')
gauss_blur.sort()
sharp = os.listdir('/home/ayushh/train_sharp_resized')
sharp.sort()
x_blur = []
for i in range(len(gauss_blur)):
    x_blur.append(gauss_blur[i])
y_sharp = []
for i in range(len(sharp)):
    y_sharp.append(sharp[i])


# Define a custom transform to convert BGR to RGB
class BGRToRGB(object):
    def __call__(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# define transforms
transform = transforms.Compose([
    BGRToRGB(),  # Convert BGR to RGB
    transforms.ToPILImage(),
    transforms.Resize((448, 256)),
    transforms.ToTensor(),
])

class DeblurDataset(Dataset):
    def __init__(self, blur_paths, sharp_paths=None, transforms=None, isTrain=True):
        self.X = blur_paths
        self.y = sharp_paths
        self.transforms = transforms
        self.isTrain=isTrain
    def __len__(self):
        if(self.isTrain==True):
            return (int(0.8*len(self.X)))
        else:
            return len(self.X)-(int(0.8*len(self.X)))
    
    def __getitem__(self, i):
        if(self.isTrain==False):
            i+=int(0.8*len(self.X))
        blur_image = cv2.imread(f"/home/ayushh/train_blurred/train_blurred/{self.X[i]}")
        # print(f"/home/ayushh/train_blurred/{self.X[i]}")
        
        if self.transforms:
            blur_image = self.transforms(blur_image)
            
        if self.y is not None:
            sharp_image = cv2.imread(f"/home/ayushh/train_sharp_resized/{self.y[i//3]}")
#             print(f"/kaggle/input/sharp-image/test_sharp/{self.y[i//3]}")
            sharp_image = self.transforms(sharp_image)
            return (blur_image, sharp_image)
        else:
            return blur_image
        

train_data = DeblurDataset(x_blur, y_sharp, transform, isTrain=True)
val_data = DeblurDataset(x_blur, y_sharp, transform, isTrain=False)
trainloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)


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
model = DeblurCNN().to(device)
# print(model)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

total_params = count_parameters(model)
print(f"Total number of trainable parameters: {total_params}")


# the loss function
criterion = nn.MSELoss()
# the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
        optimizer,
        mode='min',
        patience=5,
        factor=0.5,
        verbose=True
    )


to_image=transforms.ToPILImage(mode='RGB')


def fit(model, dataloader, epoch):
    model.train()
    running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data)/dataloader.batch_size)):
        blur_image = data[0]
        sharp_image = data[1]
        blur_image = blur_image.to(device)
        sharp_image = sharp_image.to(device)
        optimizer.zero_grad()
        outputs = model(blur_image)
        loss = criterion(outputs, sharp_image)
        # backpropagation
        loss.backward()
        # update the parameters
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(dataloader.dataset)
    print(f"Train Loss: {train_loss:.5f}")
    
    return train_loss



def validate(model, dataloader, epoch):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data)/dataloader.batch_size)):
            blur_image = data[0]
            sharp_image = data[1]
            blur_image = blur_image.to(device)
            sharp_image = sharp_image.to(device)
            outputs = model(blur_image)
            loss = criterion(outputs, sharp_image)
            running_loss += loss.item()
            if epoch == 0 and i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(sharp_image.cpu().data, name=f"/home/ayushh/deblur/saved_images/sharp{epoch}.jpg")
                save_decoded_image(blur_image.cpu().data, name=f"/home/ayushh/deblur/saved_images/blur{epoch}.jpg")
            if i == int((len(val_data)/dataloader.batch_size)-1):
                save_decoded_image(outputs.cpu().data, name=f"/home/ayushh/deblur/saved_images/val_deblurred{epoch}.jpg")
        val_loss = running_loss/len(dataloader.dataset)
        print(f"Val Loss: {val_loss:.5f}")
        
        return val_loss
    

#Training
train_loss  = []
val_loss = []
start = time.time()
for epoch in range(3):
    print(f"Epoch {epoch+1} of {3}")
    train_epoch_loss = fit(model, trainloader, epoch)
    val_epoch_loss = validate(model, valloader, epoch)
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    scheduler.step(train_epoch_loss)
end = time.time()
print(f"Took {((end-start)/60):.3f} minutes to train")


# loss plots
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(val_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/home/ayushh/deblur/loss.png')
plt.show()
# save the model to disk
print('Saving model...')
torch.save(model.state_dict(), '/home/ayushh/deblur/model_a.pth')