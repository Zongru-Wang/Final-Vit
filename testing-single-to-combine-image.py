import os
import random
import numpy as np
from PIL import Image
import itertools

#################################################
# 这个脚本的存在是为了将  256*256的图片的但局放的PRPD图像组合成多标签的PRPD图像
# 输出的图像可以放在Traning 和testing 文件夹中用于训练和验证

################################################
# 设置基本目录路径
single_image_dir = r"Testing-single-image"
combined_image_dir = r"Testing-combined-Images"

# 标签映射
label_mapping = {
    'Corona': [1, 0, 0, 0],
    "Floating Potential": [0, 1, 0, 0],
    "Free Particle": [0, 0, 1, 0],  
    "Insulation": [0, 0, 0, 1],
    "Noise": [0, 0, 0, 0]
}

# 获取所有单标签图像的路径
def get_single_label_images(single_image_dir):
    single_label_images = {}
    for label in label_mapping.keys():
        label_dir = os.path.join(single_image_dir, label)
        if os.path.exists(label_dir):
            single_label_images[label] = [os.path.join(label_dir, img) for img in os.listdir(label_dir) if img.endswith(".png")]
    return single_label_images

# 随机组合不同类型的图像
def combine_images(image_paths, target_shape):
    combined_image = np.zeros(target_shape + (3,), dtype=np.uint8)
    for img_path in image_paths:
        img = Image.open(img_path).resize(target_shape)
        img_array = np.array(img)
        combined_image = np.add(combined_image, img_array)  # 合并图像
    return Image.fromarray(combined_image)

# 创建文件夹并保存组合后的图像
def save_combined_images(single_label_images, combined_image_dir, num_images_per_combination, target_shape):
    labels = list(single_label_images.keys())
    all_combinations = []

    # 生成所有可能的标签组合（至少两个标签）
    for r in range(2, len(labels) + 1):
        combinations = list(itertools.combinations(labels, r))
        all_combinations.extend(combinations)

    for combination_labels in all_combinations:
        combination_name = '+'.join(combination_labels)
        save_dir = os.path.join(combined_image_dir, combination_name)
        os.makedirs(save_dir, exist_ok=True)

        for i in range(num_images_per_combination):
            image_paths = [random.choice(single_label_images[label]) for label in combination_labels]
            combined_image = combine_images(image_paths, target_shape)
            
            # 生成组合文件名
            original_filenames = [os.path.basename(path).replace('.png', '') for path in image_paths]
            combined_filename = '_'.join(original_filenames) + f'_{i}.png'
            
            save_path = os.path.join(save_dir, combined_filename)
            combined_image.save(save_path)

# 设置目标形状
target_shape = (256, 256)

# 获取单标签图像
single_label_images = get_single_label_images(single_image_dir)

# 创建组合图像并保存
num_images_per_combination = 150  # 每种组合生成的图像数量
save_combined_images(single_label_images, combined_image_dir, num_images_per_combination, target_shape)
