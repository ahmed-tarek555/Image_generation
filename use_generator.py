import torch
from generator import Generator
from torchvision.transforms.functional import to_pil_image

generator = Generator()
generator.load_state_dict(torch.load('generator_model.pth'))
generator.eval()

img = generator(torch.randn(1, 100, 1, 1))
img = img.squeeze(0)
img  = (img+1)/2
pil_img = to_pil_image(img)
pil_img.show()
