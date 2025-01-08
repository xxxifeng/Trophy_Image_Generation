import argparse
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import os
import numpy as np
from torchvision.utils import save_image

# 参数解析
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoches", type=int, default=1000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=20, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=256, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
    parser.add_argument("--model_interval", type=int, default=50)
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--resume", type=bool, default=False, help="if continue training")
    parser.add_argument("--sample", type=bool, default=False, help="if testing")
    parser.add_argument("--train", type=bool, default=False, help="if training")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--img_num", type=int, default=200, help="num of the generated images")
    return parser.parse_args()


class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape[0], self.img_shape[1], self.img_shape[2])
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity


class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator_CNN, self).__init__()

        self.init_size = img_shape[1] // 4
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))  # 100 ——> 128 * 8 * 8 = 8192

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, img_shape[0], 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator_CNN(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator_CNN, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1),
                     nn.LeakyReLU(0.2, inplace=True),
                     nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(img_shape[0], 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),

        )

        ds_size = img_shape[1] // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())  # 128 * 2 * 2 ——> 1

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        return validity


def train(opt):
    # 数据处理步骤
    transform = transforms.Compose(
        [
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    # 获取数据
    data = datasets.ImageFolder('./dataset', transform=transform)

    # mnist_data = datasets.MNIST(
    #     "mnist-data",
    #     train=True,
    #     download=True,
    #     transform=transform
    # )

    # 获取训练数据
    train_loader = torch.utils.data.DataLoader(
        data,
        batch_size=opt.batch_size,
        shuffle=True)

    # 得到图像形状
    img_shape = (opt.channels, opt.img_size, opt.img_size)

    # 构建图像生成器和判别器
    generator = Generator_CNN(opt.latent_dim, img_shape)
    discriminator = Discriminator_CNN(img_shape)

    if opt.resume:
        generator.load_state_dict(torch.load(f"checkpoints/DCGAN_3/generator/generator_epoch_{opt.start_epoch}.pth"))
        discriminator.load_state_dict(torch.load(f"checkpoints/DCGAN_3/discriminator/discriminator_epoch_{opt.start_epoch}.pth"))

    # 构建损失函数和优化器

    adversarial_loss = torch.nn.BCELoss()

    cuda = True if torch.cuda.is_available() else False
    print(cuda)
    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # 构建损失函数和优化器

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=(opt.lr * 8 / 9), betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    print(generator)
    print(discriminator)

    # 开始训练
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    for epoch in range(opt.n_epoches):
        count = 0
        for i, (imgs, _) in enumerate(train_loader):
            # adversarial ground truths
            valid = torch.ones(imgs.shape[0], 1).type(Tensor)
            fake = torch.zeros(imgs.shape[0], 1).type(Tensor)

            real_imgs = imgs.type(Tensor)

            #############    训练生成器    ################
            optimizer_G.zero_grad()

            # 采样噪声，作为生成器输入
            z = torch.tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))).type(Tensor)

            # 生成一批次图像
            gen_imgs = generator(z)
            # G-Loss
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)
            g_loss.backward()
            optimizer_G.step()

            #############  训练判别器 ################
            optimizer_D.zero_grad()

            # D-Loss
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G Loss: %f]"
                % (epoch+opt.start_epoch+1, opt.n_epoches+opt.start_epoch, i, len(train_loader), d_loss.item(), g_loss.item())
            )

            batches_done = (epoch+opt.start_epoch) * len(train_loader) + i
            os.makedirs("images_1", exist_ok=True)
            # 固定批次保存一次图像
            if batches_done % opt.sample_interval == 0:
                with torch.no_grad():
                    z = torch.tensor(np.random.normal(0, 1, (70, opt.latent_dim))).type(Tensor)
                    imgs = generator(z)
                    save_image(imgs.data, "images_1/%d.png" % (batches_done), nrow=10, normalize=True)
        # 固定轮数保存模型
        if (epoch+1+opt.start_epoch) % opt.model_interval == 0:
            torch.save(generator.state_dict(), f"checkpoints/DCGAN/generator/generator_epoch_{epoch+1+opt.start_epoch}.pth")
            torch.save(discriminator.state_dict(), f"checkpoints/DCGAN/discriminator/discriminator_epoch_{epoch+1+opt.start_epoch}.pth")
        torch.save(generator.state_dict(), f"checkpoints/DCGAN/generator/last.pth")
        torch.save(discriminator.state_dict(), f"checkpoints/DCGAN/discriminator/last.pth")



if __name__ == '__main__':
    opt = args_parse()
    # 训练模式
    if opt.train:
        train(opt)
    # 采样推理模式
    if opt.sample:
        with torch.no_grad():
            # 加载生成模型
            img_shape = (opt.channels, opt.img_size, opt.img_size)
            generator = Generator_CNN(opt.latent_dim, img_shape)
            generator.load_state_dict(torch.load("checkpoints/DCGAN/generator/generator_epoch_1000.pth"))
            generator.eval()
            cuda = True if torch.cuda.is_available() else False
            generator.cuda()
            # 随机采样，分批生成，保存图像
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            num_batches = (opt.img_num // opt.batch_size) + 1
            for batch_idx in range(num_batches):
                start_idx = batch_idx * opt.batch_size
                end_idx = min((batch_idx + 1) * opt.batch_size, opt.img_num)
                batch_size_current = end_idx - start_idx
                z = torch.tensor(np.random.normal(0, 1, (batch_size_current, opt.latent_dim))).type(Tensor)
                gen_imgs = generator(z)
                # 保存图像
                for i,img in enumerate(gen_imgs.data):
                    global_index = start_idx + i
                    save_image(img, f"result_DCGAN/1/{global_index}.png", normalize=True)