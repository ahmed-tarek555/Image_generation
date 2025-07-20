import torch.nn as nn
import torch.nn.functional as F

# CONVOLUTIONAL LAYER FORMULA: [(in−K+2P)/S]+1

def conv_block(in_channels, out_channels, kernel_size, stride=1, padding=0):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU())

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # input 32x32
        self.conv = nn.Sequential(conv_block(3, 64, 3, 1),  #30x30
                                  nn.Dropout(p=0.2),
                                  conv_block(64, 128, 3, 1), #28x28
                                  nn.Dropout(p=0.2),
                                  conv_block(128, 256, 4, 2),#13x13
                                  nn.Dropout(p=0.2),
                                  conv_block(256, 512, 3, 2),#6x6
                                  )

        self.projection = nn.Conv2d(3, 512, 17, 3)

        self.fc = nn.Sequential(nn.Linear(6*6*512, 2),
                                nn.Dropout(p=0.2),
                                )


    def forward(self, x, y=None):
        iden = x
        x = self.conv(x)
        iden = self.projection(iden)
        x = x + iden
        A, B, C, D = x.shape
        x = x.view(A, B*C*D)
        logits = self.fc(x)

        if y is not None:
            loss = F.cross_entropy(logits, y)
            return loss
        else:
            return logits
