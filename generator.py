import torch.nn as nn

# deconvolution formula = (input - 1) × stride - 2 × padding + kernel_size + output_padding

def convtranspos2d_block(in_channels, out_channels, kernel_size, stride, padding=0, output_padding=0):
    return nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
                         nn.BatchNorm2d(out_channels),
                         nn.ReLU())

class Generator(nn.Module):
    def __init__(self,  z_dim=100, img_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # Input: (N, 100, 1, 1)
            convtranspos2d_block(z_dim, 16, 4, 1, 0), # 4x4

            convtranspos2d_block(16, 32, 2, 2, 0), # 8x8

            convtranspos2d_block(32, 64, 2, 2, 0),  # 16x16

            convtranspos2d_block(64, 128, 5, 1, 0),  # 20x20

            nn.ConvTranspose2d(128, img_channels, 2, 2, 4), # 32x32
            nn.Tanh()
        )
        self.projection = nn.ConvTranspose2d(z_dim, img_channels, kernel_size=16,stride=18, output_padding=16)

    def forward(self, x):
        iden = x
        x = self.net(x)
        iden = self.projection(iden)
        out_img = x + iden
        return out_img

