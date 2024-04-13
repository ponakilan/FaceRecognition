import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEmbeddingModel(nn.Module):

    def __init__(self, dynamic_dropout_prob):

        super(ImageEmbeddingModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=4)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout_dynamic = nn.Dropout(p=dynamic_dropout_prob) 
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=40, out_channels=60, kernel_size=3)
        self.conv4 = nn.Conv2d(in_channels=60, out_channels=80, kernel_size=2)

        self.linear1 = nn.Linear(in_features=1200, out_features=160)
        self.linear2 = nn.Linear(in_features=960, out_features=160)
    
    def forward(self, x):
       
        x1 = self.pool(F.relu(self.conv1(x)))
        x1 = self.dropout_dynamic(x1)
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.dropout_dynamic(x1)
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = self.dropout_dynamic(x1)

        x2 = F.relu(self.conv4(x1))
        x2 = self.dropout_dynamic(x2)
       
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

        x = torch.cat((x1, x2), 1)

        x = F.normalize(x)
        return x
    