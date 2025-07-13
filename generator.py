import torch
import torch.nn as nn
import torch.nn.functional as F
from discriminator import Discriminator

discriminator = Discriminator()
def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

class Generator(nn.Module):
    def __init__(self,  z_dim=100, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (N, 100, 1, 1)
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, img_channels, 4, 2, 1),  # 128x128
            nn.Tanh()
        )

    def forward(self, x, y=False):

        out_img = self.net(x)
        if y :
            probs = discriminator(out_img)
            loss = probs[0].log()
            return loss
        else:
            return out_img

    def _train(self, n_iter, lr, data):
        optim = torch.optim.AdamW(self.parameters(), lr)

        for i in range(n_iter):
            loss = self(data, True)

            optim.zero_grad()
            loss.backward()

            optim.step()
