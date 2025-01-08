import torch.nn.functional as F
from torchvision.utils import save_image
import torch
from torch import nn
import math
from torch.optim import Adam
import logging
import argparse
from torchvision import transforms, datasets
import os
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 参数解析
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epoches", type=int, default=600, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--n_cpu", type=int, default=1, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=20, help="interval between image sampling")
    parser.add_argument("--model_interval", type=int, default=100)
    parser.add_argument("--resume", type=bool, default=False, help="if continue training")
    parser.add_argument("--sample", type=bool, default=False, help="if testing")
    parser.add_argument("--train", type=bool, default=False, help="if training")
    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch")
    parser.add_argument("--img_num", type=int, default=200, help="num of the generated images")
    return parser.parse_args()


def get_loss(model, x_0, t, device):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    # return F.l1_loss(noise, noise_pred)
    return F.mse_loss(noise, noise_pred)


# 线性betadiaper
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    # 返回一个长度为timestep的beta值序列
    return torch.linspace(start, end, timesteps)


# 从值列表中提取对应时间步的值
def get_index_from_list(vals, time_step, x_shape):
    batch_size = time_step.shape[0]
    out = vals.gather(-1, time_step.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(time_step.device)


# 正向扩散采样
def forward_diffusion_sample(x_0, time_step, device="cpu"):
    noise = torch.randn_like(x_0)  # 生成与x_0形状相同的随机噪声
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, time_step, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, time_step, x_0.shape)
    # 计算扩散后的图像
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(
        device
    ), noise.to(device)


T = 1000  # 设置最大时间步数
betas = linear_beta_schedule(timesteps=T)  # 生成beta序列
alphas = 1.0 - betas  # 计算alpha序列
alphas_cumprod = torch.cumprod(alphas, axis=0)  # alpha累积乘积
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)  # 前一时间步累积乘积
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)  # alpha倒数平方根
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)  # alpha累积乘积平方根
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)  # 1-alpha累积乘积的平方根
posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)  # 后验方差


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)  # 时间嵌入全连接层
        if up:
            # 上采样
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)  # 上采样
        else:
            # 下采样
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)  # 下采样

        # 第二卷积层和两个批量归一化层
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()  # 激活函数

    def forward(self, x, t):
        # 第一个卷积 + 批量归一化 + 激活函数
        h = self.bnorm1(self.relu(self.conv1(x)))
        # 计算时间嵌入
        time_emb = self.relu(self.time_mlp(t))
        # 扩展时间嵌入的维度
        time_emb = time_emb[(...,) + (None,) * 2]
        # 将时间嵌入添加到卷积后的特征图中
        h = h + time_emb
        # 第二个卷积 + 批量归一化 + 激活函数
        h = self.bnorm2(self.relu(self.conv2(h)))
        # 完成上采样或下采样
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim  # 嵌入维度

    # 计算正弦位置嵌入
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)  # 计算位置嵌入的缩放因子
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)  # 创建缩放因子的指数序列
        embeddings = time[:, None] * embeddings[None, :]  # 计算每个位置的嵌入
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)  # 计算正弦和余弦嵌入
        return embeddings


class SimpleUnet(nn.Module):
    def __init__(self):
        super().__init__()
        image_channels = 3  # 图像通道数
        down_channels = (64, 128, 256, 512, 1024)  # 每层下采样通道数
        up_channels = (1024, 512, 256, 128, 64)  # 每层上采样通道数
        out_dim = 3  # 输出图像的通道数
        time_emb_dim = 32  # 时间嵌入维度
        # 时间嵌入MLP网络
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),  # 使用正弦位置嵌入
            nn.Linear(time_emb_dim, time_emb_dim),  # 全连接层
            nn.ReLU()  # 激活函数
        )
        # 初始卷积层，将输入图像的通道数映射到第一个下采样层的通道数
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)
        # 下采样模块
        self.downs = nn.ModuleList(
            [Block(down_channels[i], down_channels[i + 1], time_emb_dim) for i in range(len(down_channels) - 1)]
        )
        # 上采样模块
        self.ups = nn.ModuleList(
            [Block(up_channels[i], up_channels[i + 1], time_emb_dim, up=True) for i in range(len(up_channels) - 1)]
        )
        # 输出卷积层，将最后一个上采样层的通道数映射到输出通道数
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # 计算时间嵌入
        t = self.time_mlp(timestep)
        # 初始卷积
        x = self.conv0(x)

        residual_inputs = []
        # 下采样路径
        for down in self.downs:
            x = down(x, t)  # 对每个下采样块进行处理
            residual_inputs.append(x)  # 保存中间结果用于上采样
        # 上采样路径
        for up in self.ups:
            residual_x = residual_inputs.pop()  # 获取对应残差
            x = torch.cat((x, residual_x), dim=1)  # 连接特征图
            x = up(x, t)  # 对每个上采样快进行处理
        return self.output(x)


@torch.no_grad()  # 推理阶段禁用梯度计算
def sample_timestep(model, x, t):
    betas_t = get_index_from_list(betas, t, x.shape)  # 获取该时间步对应的beta
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)  # sqrt(1-alpha累乘)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)  # sqrt(1/alpha)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)  # 获取当前时间步的模型均值
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)  # 当前时间步后验方差

    if t == 0:
        # 若时间步为0，直接返回均值
        return model_mean
    else:
        # 否则加入噪声
        noise = torch.randn_like(x)  # 生成与x形状相同的噪声
        return model_mean + torch.sqrt(posterior_variance_t) * noise  # 返回带有噪声的样本


def sample_plot_image(model, device, img_size, T, batch_size):
    # 初始化随机噪声
    img = torch.randn((batch_size, 3, img_size, img_size), device=device)
    # 反向迭代采样图像
    for i in reversed(range(0, T)):
        t = torch.tensor([i], device=device, dtype=torch.long)
        img = sample_timestep(model, img, t)  # 采样该时间步图像
        img = torch.clamp(img, 0, 1.0)  # 限制图象值在0~1

    return img


if __name__ == "__main__":
    opt = args_parse()

    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 采样推理
    if opt.sample:
        # 加载模型
        with torch.no_grad():
            model = SimpleUnet()
            model.load_state_dict(torch.load("./checkpoints/DDPM/model_epoch_2000.pth"))
            model.to(device)
            model.eval()
            # 分批采样图像
            num_batches = (opt.img_num // opt.batch_size) + 1
            for batch_idx in range(num_batches):
                start_idx = batch_idx * opt.batch_size
                end_idx = min((batch_idx + 1) * opt.batch_size, opt.img_num)
                batch_size_current = end_idx - start_idx
                imgs = sample_plot_image(model=model, device=device, img_size=opt.img_size, T=1000,
                                         batch_size=batch_size_current)
                for i, img in enumerate(imgs):
                    global_index = start_idx + i
                    save_image(img, f"result_DDPM/1/{global_index}.png", normalize=True)
    # 训练
    if opt.train:
        # 加载数据集
        transform = transforms.Compose([
            transforms.Resize((opt.img_size, opt.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        dataset = datasets.ImageFolder('./dataset', transform=transform)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

        # 初始化模型、优化器
        model = SimpleUnet()
        model.to(device)
        optimizer = Adam(model.parameters(), lr=opt.lr)

        # 如果继续上次的训练，加载模型参数
        if opt.resume:
            model.load_state_dict(torch.load(f"checkpoints/DDPM/model_epoch_{opt.start_epoch}.pth"))
            model.to(device)

        # 开始训练
        global_batch_count = 0
        for epoch in range(opt.n_epoches):
            count = 0
            for batch_idx, (batch, _) in enumerate(dataloader):
                optimizer.zero_grad()

                t = torch.randint(0, opt.time_steps, (batch.shape[0],), device=device).long()
                x_noisy, noise = forward_diffusion_sample(batch, t, device)
                noise_pred = model(x_noisy, t)
                loss = F.mse_loss(noise, noise_pred)
                loss.backward()
                optimizer.step()
                print(
                    f"[Epoch {epoch + 1 + opt.start_epoch}/{opt.n_epoches + opt.start_epoch}] [Batch {batch_idx + 1}/{len(dataloader)}] [Loss: {loss.item()}]")

            # 保存模型
            if (epoch + 1 + opt.start_epoch) % opt.model_interval == 0:
                torch.save(model.state_dict(), f"checkpoints/DDPM/model_epoch_{epoch + 1 + opt.start_epoch}.pth")
            torch.save(model.state_dict(), f"checkpoints/DDPM/last.pth")


