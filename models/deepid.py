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

        self.linear1 = nn.Linear(in_features=self._calculate_linear_input(47, 55), out_features=160)
        self.linear2 = nn.Linear(in_features=self._calculate_linear_input(47, 55), out_features=160)

    def _calculate_linear_input(self, h, w):
       
        def conv2d_size_out(size, kernel_size, stride=1):
            return (size - (kernel_size - 1) - 1)         
        def pool2d_size_out(size, kernel_size, stride):
            return (size - (kernel_size - 1) - 1)         
        
        h = conv2d_size_out(h, 4)
        w = conv2d_size_out(w, 4)
        h = pool2d_size_out(h, 2, 2)
        w = pool2d_size_out(w, 2, 2)
        
        
        h = conv2d_size_out(h, 3)
        w = conv2d_size_out(w, 3)
        h = pool2d_size_out(h, 2, 2)
        w = pool2d_size_out(w, 2, 2)
        
        h = conv2d_size_out(h, 3)
        w = conv2d_size_out(w, 3)
        h = pool2d_size_out(h, 2, 2)
        w = pool2d_size_out(w, 2, 2)
        
       
        h = conv2d_size_out(h, 2)
        w = conv2d_size_out(w, 2)
        
        return h * w * 80  
    def forward(self, x):
       
        x1 = self.pool(F.relu(self.conv1(x)))
        x1 = self.dropout_dynamic(x1)
        x1 = self.pool(F.relu(self.conv2(x1)))
        x1 = self.dropout_dynamic(x1)
        x1 = self.pool(F.relu(self.conv3(x1)))
        x1 = self.dropout_dynamic(x1)

        
        x2 = F.relu(self.conv4(x))
        x2 = self.dropout_dynamic(x2)
        
       
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        
        
        x1 = self.linear1(x1)
        x2 = self.linear2(x2)

       
        x = torch.cat((x1, x2), 1)
        return x
