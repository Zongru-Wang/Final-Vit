import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import collections
from transformers import TFAutoModel
from tensorflow.keras.callbacks import TensorBoard


#################################################
# 这个脚本的存在是为了将 dat文件中读取的 PRPD的 csv文件转换为 256*256的图片


################################################
# 配置GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)

# 标签映射
label_mapping = {
    'Corona': [1, 0, 0, 0],
    "Floating Potential": [0, 1, 0, 0],
    "Free Particle": [0, 0, 1, 0],  
    "Insulation": [0, 0, 0, 1],
    "Noise": [0,0,0,0]
}

# 设置基本目录路径
output_base_dir = r"Raw2D-SinglePRPD-CSV"
image_output_dir = r"Raw2D-CSV-to-Images"

# 获取所有组合文件的路径
def get_combined_files(output_base_dir, graph_type):
    combined_files = []
    for root, dirs, files in os.walk(os.path.join(output_base_dir, graph_type)):
        for file in files:
            if file.endswith(".csv"):
                combined_files.append(os.path.join(root, file))
    return combined_files

# 读取CSV文件并转换为二维数组
def load_csv_to_array(file_path):
    print("Loading file: ", file_path)
    data = pd.read_csv(file_path, header=None).values
    return data

# 使用零填充或裁剪统一数组形状
def pad_or_crop_to_shape(array, target_shape):
    result = np.zeros(target_shape)
    min_shape = np.minimum(array.shape, target_shape)
    result[:min_shape[0], :min_shape[1]] = array[:min_shape[0], :min_shape[1]]
    return result

# 将二维数组转换为带有自定义颜色映射的RGB图像
def array_to_custom_color_image(array):
    cdict = {
        'red':   [(0.0, 1.0, 1.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 1.0, 1.0)],
        'green': [(0.0, 1.0, 1.0),
                  (0.25, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)],
        'blue':  [(0.0, 1.0, 1.0),
                  (0.25, 1.0, 1.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0)]
    }
    custom_cmap = LinearSegmentedColormap('custom_cmap', cdict)
    array = array[1:, :]
    min_nonzero = np.min(array[np.nonzero(array)])
    adjusted_array = np.where(array == 0, min_nonzero * 0.1, array)
    log_array = np.log1p(adjusted_array)
    norm_array = (log_array - np.min(log_array)) / (np.max(log_array) - np.min(log_array))

    fig, ax = plt.subplots(figsize=(256/80, 256/80), dpi=80)
    ax.imshow(norm_array, cmap=custom_cmap, aspect='auto')
    ax.axis('off')

    # 将绘制的图像转换为 PIL 图像
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = Image.fromarray(img)

    plt.close(fig)
    return img

# 保存图像到本地
def save_image_from_csv(file_path, target_shape, output_dir):
    array = load_csv_to_array(file_path)
    array = pad_or_crop_to_shape(array, target_shape)
    img = array_to_custom_color_image(array)
    directory_name = os.path.basename(os.path.dirname(file_path))
    save_dir = os.path.join(output_dir, directory_name)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, os.path.basename(file_path).replace('.csv', '.png'))
    img.save(save_path)

# 批量处理所有CSV文件并保存图像
def process_and_save_all_images(output_base_dir, image_output_dir, graph_type, target_shape):
    combined_files = get_combined_files(output_base_dir, graph_type)
    for file_path in combined_files:
        save_image_from_csv(file_path, target_shape, image_output_dir)

# 设置目标形状
target_shape = (256, 256)
process_and_save_all_images(output_base_dir, image_output_dir, 'PRPD', target_shape)
