import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from networks.inceptionv1 import BasicConv2d, Block35, Block17, Mixed_6a, Mixed_7a, Block8


class InceptionResnetEmbedding(L.LightningModule):

    def __init__(self, dropout_prob, wandb_run):
        super().__init__()

        self.run = wandb_run

        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        x = F.normalize(x, p=2., dim=1)
        return x
    
    def training_step(self, batch, batch_idx):

        anchor, positive, negative = batch

        a_e = self.forward(anchor)
        p_e = self.forward(positive)
        n_e = self.forward(negative)

        pos_dist = torch.norm(a_e - p_e, p=2, dim=1).detach().tolist()[0]
        neg_dist = torch.norm(a_e - n_e, p=2, dim=1).detach().tolist()[0]

        loss = F.triplet_margin_loss(a_e, p_e, n_e, 1.2)
        self.log("train_loss", loss, sync_dist=True)
        self.log("pos_dist", pos_dist, sync_dist=True)
        self.log("neg_dist", neg_dist, sync_dist=True)
        self.run.log({"train_loss": loss, "pos_dist": pos_dist, "neg_dist": neg_dist})

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


