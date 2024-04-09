import torch
from torch import nn
from torch.nn import functional as F
import lightning as L

from networks.inceptionv1 import BasicConv2d, Block35, Block17, Mixed_6a, Mixed_7a, Block8


class InceptionResnetEmbedding(L.LightningModule):

    def __init__(self, weights=None, weights_path=None, dropout_prob=0.6, device=None):
        super().__init__()

        self.weights = weights

        if weights == 'vggface2':
            tmp_classes = 8631
        elif weights == 'casia-webface':
            tmp_classes = 10575

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

        if weights is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, weights, weights_path)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

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
        loss = F.triplet_margin_loss(a_e, p_e, n_e)
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def load_weights(mdl, name, weights_path):
    if name == 'vggface2':
        path = f'{weights_path}/vggface2.pt'
    elif name == 'casia-webface':
        path = f'{weights_path}/casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    state_dict = torch.load(path)
    mdl.load_state_dict(state_dict)
