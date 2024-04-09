import lightning as L
import torch.nn.functional as F
import numpy as np
import torch
from torch import nn

class Encoder(L.LightningModule):
    def __init__(self, image_size, channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.shape_before_flattening = None
        flattened_size = (image_size // 8) * (image_size // 8) * 128
        self.fc = nn.Linear(flattened_size, embedding_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        self.shape_before_flattening = x.shape[1:]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class Decoder(L.LightningModule):
    def __init__(self, embedding_dim, shape_before_flattening, channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(embedding_dim, np.prod(shape_before_flattening))
        self.reshape_dim = shape_before_flattening
        self.deconv1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(32, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), *self.reshape_dim)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = torch.sigmoid(self.conv1(x))
        return x

class AutoEncoder(L.LightningModule):
    def __init__(self, image_size: int, channels: int, embedding_dim: int):
        super().__init__()
        self.encoder = Encoder(image_size, channels, embedding_dim)
        self.encoder(torch.rand((1, 3, 160, 160)))
        self.decoder = Decoder(embedding_dim, self.encoder.shape_before_flattening, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer