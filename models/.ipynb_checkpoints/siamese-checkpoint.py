import torch
from torch import nn

from .embedder import InceptionResnetEmbedding


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(Siamese, self).__init__()
        self.encoder = InceptionResnetEmbedding(
            weights='vggface2',
            weights_path='TrainedWeights'
        )
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

        self.encoder.requires_grad_(False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = torch.sigmoid(x)
        return x
