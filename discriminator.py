import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_dataset import load_dataset

data, targets = load_dataset('real', 1)

def get_batch(batch_size, data, targets):
    batch = torch.randint(0, data.shape[0], (batch_size, ))
    return data[batch], targets[batch]

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(conv_block(3, 64, 3, 1),
                                  conv_block(64, 512, 3, 2))

        # self.fc = nn.Sequential(nn.Linear(),
        #                         nn.ReLU())


    def forward(self, x, y=None):
        x = self.conv(x)

        logits = x
        return logits

model = Discriminator()
x, y = get_batch(32, data, targets)


logits = model(x)
print(logits.shape)
