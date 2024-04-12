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

    def training_step(self, batch, batch_idx):

        anchor, positive, negative = batch

        a_e = self.forward(anchor)
        p_e = self.forward(positive)
        n_e = self.forward(negative)

        pos_dist = torch.norm(a_e[0] - p_e[0], p=2).detach().tolist()
        neg_dist = torch.norm(a_e[0] - n_e[0], p=2).detach().tolist()
        self.run.log({"pos_dist": pos_dist, "neg_dist": neg_dist})

        loss = F.triplet_margin_loss(a_e, p_e, n_e, 1.2)
        self.log("train_loss", loss, sync_dist=True)
        self.run.log({"train_loss": loss})

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        anchor, positive, negative = batch

        a_e = self.forward(anchor)
        p_e = self.forward(positive)
        n_e = self.forward(negative)

        loss = F.triplet_margin_loss(a_e, p_e, n_e, 1.2)
        self.log("val_loss", loss, sync_dist=True)
        self.run.log({"val_loss": loss})
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=1e-3)
        return optimizer
