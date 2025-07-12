import os
import torch

def load_dataset(path, flag):
    x = []
    y = []
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        x.append(torch.load(img_path, weights_only=False))
        y.append(flag)
    x = torch.stack(x)
    y = torch.tensor(y)
    return x, y
