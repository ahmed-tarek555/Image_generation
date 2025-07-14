import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(conv_block(3, 32, 3, 1),
                                  conv_block(32, 256, 3, 2),
                                  conv_block(256, 512, 3, 2))

        self.fc = nn.Sequential(nn.Linear(460800, 2),
                                nn.ReLU())


    def forward(self, x, y=None):
        x = self.conv(x)
        A, B, C, D = x.shape
        x = x.view(A, B*C*D)
        logits = self.fc(x)

        if y is not None:
            loss = F.cross_entropy(logits, y)
            return loss
        else:
            return logits
