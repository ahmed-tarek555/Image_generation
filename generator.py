import torch.nn as nn

# deconvolution formula = (input - 1) × stride - 2 × padding + kernel_size + output_padding

class Generator(nn.Module):
    def __init__(self,  z_dim=100, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (N, 100, 1, 1)
            nn.ConvTranspose2d(z_dim, 16, 4, 1, 0), # 4x4
            nn.BatchNorm2d(16),
            nn.ReLU(),

            nn.ConvTranspose2d(16, 32, 2, 2, 0), # 8x8
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(32, 64, 2, 2, 0),  # 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, img_channels, 4, 2, 1), # 32x32
            nn.Tanh()
        )
        self.projection = nn.ConvTranspose2d(z_dim, img_channels, kernel_size=16,stride=18, output_padding=16)

    def forward(self, x):
        iden = x
        x = self.net(x)
        iden = self.projection(iden)
        out_img = x + iden
        return out_img

