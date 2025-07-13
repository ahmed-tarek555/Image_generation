import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.load_dataset import load_dataset
from utils.process_img import process_img

batch_size = 32
# data, targets = load_dataset('real', 1)

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

        self.fc = nn.Sequential(nn.Linear(512*62*62, 2),
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
            probs = F.softmax(logits, dim=1)
            return probs

    def _train(self, n_iter, lr, data, targets):
        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr)

        for i in range(n_iter):
            x, y = get_batch(batch_size, data, targets)
            loss = self(x, y)

            optim.zero_grad()
            loss.backward()

            optim.step()



model = Discriminator()

def discriminate(img):
    img = torch.load(img, weights_only=False)
    img  = torch.stack((img, ))
    probs = model(img)
    idx = torch.argmax(probs)
    return probs, idx

test_img = os.listdir('real')[0]
img_path = os.path.join('real', test_img)


print(discriminate(img_path))