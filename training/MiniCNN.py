import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class MiniCNN(nn.Module):
    def __init__(self, num_classes=2,):
        super(MiniCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.batch1 = nn.BatchNorm2d(self.conv1.out_channels)
        self.max1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=self.conv1.out_channels, out_channels=32, kernel_size=3, padding=1)
        self.batch2 = nn.BatchNorm2d(self.conv2.out_channels)
        self.max2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=self.conv2.out_channels, out_channels=64, kernel_size=3, padding=1)
        self.batch3 = nn.BatchNorm2d(self.conv3.out_channels)
        self.adapt_pool = nn.AdaptiveMaxPool2d((4, 4))
        
        #self.fc1 = nn.Linear(in_features=64 * 4 * 4,  out_features=768)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)
        self.fc2 = nn.Linear(in_features=self.fc1.out_features, out_features=num_classes)
        
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv2(x)
        x = self.batch2(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv3(x)
        x = self.batch3(x)
        x = F.relu(x)
        
        x = self.adapt_pool(x)
        
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x