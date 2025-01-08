# 奖杯图像生成
## 项目介绍
本项目分别使用了DCGAN与DDPM两种图像生成模型，分别对收集的奖杯数据进行训练生成奖杯图像

## 环境依赖
- python3.7
- pytorch1.7.1
- torchvision0.8.2

## 数据预处理
项目使用了通过网络爬取得到的将被数据集，每张图像大小不一名称不同，在进行训练前需要将数据集进行预处理，统一图像大小为256*256，图像名称为对应的序号。终端运行下面的命令即可完成预处理：
```
python load_data.py
```

## 模型训练
终端运行下面的命令,即可从0开始对模型进行训练(例：对DCGAN从头开始进行训练)：
```
python DCGAN.py \
--train True
```
若需要从指定的epoch开始继续训练,则按照下面的格式编写指令(例：从第100轮开始继续进行训练100轮DCGAN模型),更详细参数可参考代码本身
``` 
python DCGAN.py \ 
--resume True  \
--train True  \
--start_epoch 100 \
--n_epoches 100
```
若要对DDPM模型进行训练,则将"DCGAN.py"文件名更改为"DDPM.py"


# 生成图像
终端运行下面的命令,即可生成图像(例：基于DCGAN模型生成100张图像)
```
python DCGAN.py \
--sample True \
--img_num 100
```
训练好的模型保存在"checkpoint"文件夹下，使用DCGAN，则将生成器和判别器对应的checkpoint分存储在"DCGAN/generator"和"DCGAN/discriminator"文件夹下，使用DDPM，则将checkpoint存储在"DDPM"文件夹下,已上传部分训练好的模型到[百度网盘]()，密码：

# 生成图像评估
注意此处生成图像数量需要提前生成好，并与真实数据集图像数量保持一致，图像路径的父文件夹需要包含在输入路径下，运行下面命令即可输出生成图像的真实性、FID、特征分布情况对比图(例：指定最后训练的判别器作为鉴别器，生成图像路径为"./result_DCGAN"，真实数据集路径为"./dataset"):
``` 
python evaluate.py \
--real_path "./dataset" \ 
--fake_path "./result_DCGAN" \ 
--classifier_path "./checkpoint/DCGAN/discriminator/last.pth"
```

### 评估结果展示
