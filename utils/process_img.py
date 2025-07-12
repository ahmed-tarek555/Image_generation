from PIL import Image
from torchvision import transforms

transform = transforms.ToTensor()
format = 'RGB'
target_size = (128, 128)

def process_img(path):
    if isinstance(path, Image.Image):
        img = path.convert(format)
    else:
        img = Image.open(path).convert(format)
    img = img.resize(target_size)
    img = transform(img)
    return img