import torch
from torch import nn
import lightning as L

from .embedder import InceptionResnetEmbedding


class SiameseNetwork(L.LightningModule):
    def __init__(self, pretrained=False):
        super(SiameseNetwork, self).__init__()
        self.encoder = InceptionResnetEmbedding()
        if pretrained:
            self.encoder = InceptionResnetEmbedding(
                weights='vggface2',
                weights_path='models/TrainedWeights'
            )
        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward(self, x, y):
        cat = torch.cat([x, y], dim=0)
        embeddings = self.encoder(cat)
        x = embeddings.reshape((1, 1024))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x
