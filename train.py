import torch
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from generator import Generator
from discriminator import Discriminator
from utils.load_dataset import  load_dataset

discriminator = Discriminator()
generator = Generator()
data, targets = load_dataset('real', 1)
data = F.interpolate(data, size=(32, 32), mode='bilinear', align_corners=False)
data = data*2 - 1
batch_size = 32
gen_lr = 3e-5
disc_lr = 5e-6

def get_batch(batch_size, data, targets):
    batch = torch.randint(0, data.shape[0], (batch_size, ))
    return data[batch], targets[batch]

def train(n_iter, gen_lr, disc_lr):
    global fake_img
    gen_optim = torch.optim.AdamW(generator.parameters(), gen_lr)
    disc_optim = torch.optim.AdamW(discriminator.parameters(), disc_lr)

    # gen_scheduler = torch.optim.lr_scheduler.StepLR(gen_optim, step_size=3000, gamma=0.5)
    # disc_scheduler = torch.optim.lr_scheduler.StepLR(disc_optim, step_size=3000, gamma=0.5)

    current_iter = 0
    for i in range(n_iter):
        init = torch.randn(batch_size, 100, 1, 1)
        fake_img = generator(init)
        logits = discriminator(fake_img)
        gen_loss = F.cross_entropy(logits, torch.ones(batch_size, dtype=torch.long))
        gen_optim.zero_grad()
        gen_loss.backward()
        print(generator.net[0][0].weight.grad)
        print(f'gen loss is {gen_loss}')
        gen_optim.step()

        x_real, y_real = get_batch(batch_size, data, targets)
        x_fake = fake_img.detach()
        y_fake = torch.zeros(batch_size, dtype=torch.long)

        disc_optim.zero_grad()
        disc_loss_real = discriminator(x_real, y_real)
        disc_loss_fake = discriminator(x_fake, y_fake)
        disc_loss = disc_loss_fake + disc_loss_real
        disc_loss.backward()
        print(f'discriminators wights {discriminator.fc[0].weight.grad}')
        print(f'disc loss is {disc_loss}')
        disc_optim.step()

        current_iter += 1
        print(int((current_iter / n_iter) * 100))

        if current_iter % 500 == 0:
            img = generator(torch.randn(1, 100, 1, 1))
            img = img.squeeze(0)
            img = (img + 1) / 2
            pil_img = to_pil_image(img)
            pil_img.show()



train(10000, gen_lr, disc_lr)

# torch.save(generator.state_dict(), f'path.pth')
# torch.save(discriminator.state_dict(), f'path.pth')