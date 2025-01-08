from PIL import Image
import os
import json


def process_image(input_floder, output_floder):
    # 确保输出文件夹存在
    if not os.path.exists(output_floder):
        os.makedirs(output_floder)

    # 遍历图像
    count = 0
    for image_file in os.listdir(input_floder):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            # 读取图像
            img_path = os.path.join(input_floder, image_file)
            img = Image.open(img_path)
            # 如果图像是 'P' 模式，转换为 'RGB'
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # 等比缩放图像
            target_height = 256
            original_width, original_height = img.size
            ratio = target_height / original_height
            new_width = int(original_width * ratio)
            img = img.resize((new_width, target_height), Image.LANCZOS)
            # 裁剪或填充
            if new_width > 256:
                img = img.crop(((new_width - 256) // 2, 0, new_width - (new_width - 256) // 2, target_height))
            elif new_width < 256:
                new_img = Image.new('RGB', (256, target_height), (255, 255, 255))  # 创建白色背景
                new_img.paste(img, (256 // 2 - new_width // 2, 0))  # 将图像粘贴到白色背景中心
                img = new_img

            # 保存图像
            output_path = os.path.join(output_floder, str(count) + '.jpg')
            img.save(output_path)

            count += 1


if __name__ == '__main__':
    # 数据预处理
    process_image('./奖杯素材', './dataset/cup')