from DCGAN import Discriminator_CNN
import torch
import numpy as np
from scipy import linalg
from torchvision import transforms, datasets
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import argparse

# 参数解析
def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_path", type=str, default='./dataset', help="path to real dataset")
    parser.add_argument("--fake_path", type=str, default='./result_DCGAN', help="path to fake dataset")
    parser.add_argument("--classifier_path", type=str, default='./checkpoints/DCGAN/discriminator/discriminator_epoch_1000.pth', help="path to classifier")
    parser.add_argument("--distribution_name", type=str, default='DCGAN', help="name of the dataset")

    return parser.parse_args()


# 提取特征
def extract_features(dataloader, discriminator, pca_components=2048, device='cuda'):
    features = []
    for batch in tqdm(dataloader, desc='Extracting features'):
        images = batch[0].to(device)
        with torch.no_grad():
            outputs = discriminator.model(images)
            features_batch = outputs.cpu().view(outputs.size(0), -1).numpy()
            features.append(features_batch)

    # 将所有批次的特征合并为一个大数组
    features = np.concatenate(features, axis=0)

    # 使用整个数据集来拟合PCA模型，并进行降维
    pca = PCA(n_components=pca_components)
    features_reduced = pca.fit_transform(features)

    return features_reduced


def calculate_reality(dataloader, discriminator):
    # 计算平均真实性
    reality = 0.0
    for batch_idx, (batch, _) in enumerate(dataloader):
        batch = batch.to(device)
        out = discriminator(batch)
        out = out.mean().item()
        reality += out
    return reality / len(dataloader)


def calculate_fid(real_dataloader, fake_dataloader, device='cuda'):
    # 计算均值和协方差
    def calculate_statistics(features, batch_size=20):
        # 初始化均值和协方差矩阵
        mean = np.mean(features, axis=0)
        cov = np.cov(features, rowvar=False)
        return mean, cov

    # 计算FID
    def calculate_fid_score(real_mean, real_cov, fake_mean, fake_cov, eps=1e-6):
        # 计算两个分布之间的Fréchet距离
        diff = real_mean - fake_mean
        covmean, _ = linalg.sqrtm(real_cov.dot(fake_cov), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(real_cov.shape[0]) * eps
            covmean = linalg.sqrtm((real_cov + offset).dot(fake_cov + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(real_cov) + np.trace(fake_cov) - 2 * tr_covmean
        return fid

    real_features = extract_features(real_dataloader, discriminator)
    fake_features = extract_features(fake_dataloader, discriminator)
    # 提取特征维度过大，需先进行降维
    real_mean, real_cov = calculate_statistics(real_features)  # 分批处理以防爆内存
    fake_mean, fake_cov = calculate_statistics(fake_features)
    fid = calculate_fid_score(real_mean, real_cov, fake_mean, fake_cov)
    return fid


def visual_distribution(fake_dataloader, real_dataloader, img_pth, device='cuda'):
    real_features = extract_features(real_dataloader, discriminator, 2)
    fake_features = extract_features(fake_dataloader, discriminator, 2)

    plt.figure(figsize=(8, 6))
    plt.scatter(real_features[:, 0], real_features[:, 1], c='dodgerblue', label='Real', s=50, marker='o', alpha=0.3)
    plt.scatter(fake_features[:, 0], fake_features[:, 1], c='orangered', label='Fake', s=50, marker='*', alpha=0.3)
    plt.title('Features Distribution')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')

    plt.legend()
    plt.savefig(img_pth)
    plt.close()

def draw_line_chart(x,y_1,y_2,y1_name,y2_name,savename,x_name,y_name,title):
    fig, ax = plt.subplots()
    ax.plot(x, y_1, marker='D', linestyle='-', color='#1f77b4', label=y1_name)
    ax.plot(x, y_2, marker='^', linestyle='--', color='#ff7f0e', label=y2_name)
    for i in range(len(x)):
        ax.text(x[i], y_1[i], f'{y_1[i]:.2f}', ha='center', va='bottom', fontsize=8)
        ax.text(x[i], y_2[i], f'{y_2[i]:.2f}', ha='center', va='top', fontsize=8)
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('results/'+savename+'.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    # 获取参数
    opt = args_parse()
    # 读取数据
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    real_dataset = datasets.ImageFolder(opt.real_path, transform=transform)
    real_dataloader = torch.utils.data.DataLoader(real_dataset, batch_size=20, shuffle=True)
    fake_dataset = datasets.ImageFolder(opt.fake_path, transform=transform)
    fake_dataloader = torch.utils.data.DataLoader(fake_dataset, batch_size=20, shuffle=True)

    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = Discriminator_CNN(img_shape=(3, 256, 256)).to(device)
    discriminator.load_state_dict(torch.load(opt.classifier_path))
    discriminator.eval()

    # # 读取图片(测试)
    # img = Image.open("results/DCGAN/98.png")
    # img = img.resize((256, 256))
    # img = transforms.ToTensor()(img)
    # img = img.unsqueeze(0)
    # img = img.to(device)
    # # 生成判别结果
    # out = discriminator(img)
    # print(out)

    # 计算图像生成平均真实性
    reality = calculate_reality(fake_dataloader, discriminator)
    print(f"Reality: {reality:.6f}")

    # 计算fid
    fid = calculate_fid(real_dataloader,fake_dataloader,device)
    print(f"FID: {fid:.6f}")

    # 可视化特征降维分布
    visual_distribution(fake_dataloader,real_dataloader,f"results/{opt.distribution_name}.png")

    # # 绘制折线图
    # epoch = [100,200,300,400,500,600,700,800,900,1000]
    # FID_1 = [9.567455,69.162735,8.835607,67.653735,24.707522,38.874411,19.883835,18.996005,6.267735,49.837314]
    # FID_2 = [20.668537,2.974801,12.400005,31.775700,1.772049,2.753478,0.296034,1.196453,3.362758,5.337171]
    # draw_line_chart(epoch, FID_1, FID_2, "256*256", "64*64", "FID", "Epoch", "FID Score", "FID Scores Over Epochs")
    # Reality_1 = [33.1504,20.4234,14.1992,21.9044,19.8887,21.0417,21.6477,24.3193,31.5976,33.3616]
    # Reality_2 = [54.4241,48.6337,50.5698,51.1380,49.0357,49.7345,51.1758,53.3582,54.1275,56.7434]
    # draw_line_chart(epoch, Reality_1, Reality_2, "256*256", "64*64", "Reality", "Epoch", "Reality", "Reality Over Epochs")