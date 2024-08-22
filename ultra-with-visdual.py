import os
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import logging
logging.getLogger('tensorflow').disabled = True
import numpy as np
import collections
import datetime
from PIL import Image, ImageDraw
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import TFAutoModel
from tensorflow.keras.callbacks import Callback, TensorBoard
import tensorflow_addons as tfa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import os
import shutil

###############################
# 打开conda promot 输入 
# conda env list 查看所有环境

# 使用 conda activate 环境名 进入环境
# 使用 conda deactivate 退出环境
# 使用 conda remove -n 环境名 --all 删除环境


# 使用 conda activate directml-AMD 激活环境，该环境的备份已经保存在桌面上
# 激活后 进入从 conda promot 的commandline 中进入这个文件夹
# 使用 python ultra-with-visdual.py 运行这个训练模型训练脚本

###############################
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
    "Insulation": [0, 0, 0, 1]
}

# 设置基本目录路径
image_output_dir = r"C:\Users\GLB\Desktop\Transformer\Training-combined-Images"

# 按类别分类文件
def categorize_files(files):
    categories = collections.defaultdict(list)
    for file in files:
        directory_name = os.path.basename(os.path.dirname(file))
        categories[directory_name].append(file)
    return categories

def generate_label_from_directory(directory_name):
    categories = set(directory_name.split('+'))  # 使用 set 来确保每个标签只考虑一次
    label = np.zeros(4)
    for category in categories:
        if category == "Noise":
            continue  # 跳过 Noise 标签
        if category in label_mapping:
            label += np.array(label_mapping[category])
        else:
            raise ValueError(f"Unknown category: {category}")
    return label
def load_image_and_label(file_path):
    directory_name = os.path.basename(os.path.dirname(file_path))
    label = generate_label_from_directory(directory_name)
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_data = np.array(img) / 255.0
    # print("Loading ... ...", img_data.shape, label)
    return img_data, label
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# 数据生成器
def add_grid(image):
    draw = ImageDraw.Draw(image)
    for i in range(0, image.width, 16):  # Adjust the step size as needed
        draw.line((i, 0, i, image.height), fill="black", width=1)
    for i in range(0, image.height, 16):
        draw.line((0, i, image.width, i), fill="black", width=1)
    return image

def add_sine_wave(image):
    draw = ImageDraw.Draw(image)
    frequency = 2  # Adjust the frequency as needed
    amplitude = 10  # Adjust the amplitude as needed
    for x in range(image.width):
        y = int(image.height / 2 + amplitude * np.sin(2 * np.pi * frequency * x / image.width))
        draw.point((x, y), fill="black")
    return image
def visualize_predictions(batch_data, batch_labels, batch_predictions, batch_index, model, epoch_dir):
    fig, axes = plt.subplots(min(len(batch_data), 4), 3, figsize=(15, 5 * min(len(batch_data), 4)))
    
    for i in range(min(len(batch_data), 4)):
        # Original Image
        axes[i, 0].imshow(batch_data[i])
        axes[i, 0].set_title("Original Image")

        # After ViT
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('tf_vi_t_model').output)
        vit_output = intermediate_model.predict(np.expand_dims(batch_data[i], axis=0))
        vit_output = vit_output.last_hidden_state  # Extract the last hidden state
        vit_output_patches = vit_output[0, 1:, :]  # Remove the CLS token and reshape patches

        projection_dim = vit_output_patches.shape[-1]
        patch_size = 16
        num_patches = 14  # Since 224/16 = 14
        vit_output_reshaped = vit_output_patches.reshape((num_patches, num_patches, projection_dim))
        channels = 3  # Number of channels to visualize
        
        # Only take the first 3 channels for visualization
        reconstructed_image = np.zeros((num_patches * patch_size, num_patches * patch_size, channels))

        for j in range(num_patches):
            for k in range(num_patches):
                patch = vit_output_reshaped[j, k, :channels]
                reconstructed_image[j * patch_size:(j + 1) * patch_size, k * patch_size:(k + 1) * patch_size, :] = patch

        axes[i, 1].imshow(np.clip(reconstructed_image, 0, 1))
        axes[i, 1].set_title("After ViT")
        # vit_output_reshaped = vit_output_patches.reshape((num_patches, num_patches, -1))
        
        # # Reconstruct the image from patches
        # reconstructed_image = np.zeros((num_patches * patch_size, num_patches * patch_size, 3))
        # for j in range(num_patches):
        #     for k in range(num_patches):
        #         patch = vit_output_reshaped[j, k].reshape((patch_size, patch_size, 3))
        #         reconstructed_image[j * patch_size:(j + 1) * patch_size, k * patch_size:(k + 1) * patch_size, :] = patch
        # axes[i, 1].imshow(reconstructed_image, cmap='viridis')
        # axes[i, 1].set_title("After ViT")

        # After Transformer Block
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('transformer_block_2').output)
        transformer_output = intermediate_model.predict(np.expand_dims(batch_data[i], axis=0))
        transformer_output_patches = transformer_output[0, 1:, :]  # Remove the CLS token and reshape patches


        if transformer_output_patches.shape[0] == 0:  # Handle case where there are no patches
            transformer_output_patches = transformer_output[0, :, :]  # Do not remove the CLS token
        transformer_output_patches = transformer_output_patches.reshape((-1, projection_dim))
        transformer_output_reshaped = transformer_output_patches[:num_patches*num_patches, :].reshape((num_patches, num_patches, projection_dim))
        
        # Reconstruct the image from patches
        reconstructed_image = np.zeros((num_patches * patch_size, num_patches * patch_size, channels))

        for j in range(num_patches):
            for k in range(num_patches):
                patch = transformer_output_reshaped[j, k, :channels]
                reconstructed_image[j * patch_size:(j + 1) * patch_size, k * patch_size:(k + 1) * patch_size, :] = patch
        
        axes[i, 2].imshow(np.clip(reconstructed_image, 0, 1))
        axes[i, 2].set_title("After Transformer Block")
        # Show predicted probabilities
        actual_label = batch_labels[i]
        predicted_label = batch_predictions[i]
        
        # Display probabilities for each class
        prob_text = "\n".join([f"Class {j}: {pred:.2f}" for j, pred in enumerate(predicted_label)])
        axes[i, 2].text(1.05, 0.5, prob_text, transform=axes[i, 2].transAxes, verticalalignment='center')
        
        # Display actual vs predicted labels
        actual_text = "\n".join([f"Actual Class {j}: {act}" for j, act in enumerate(actual_label)])
        predicted_text = "\n".join([f"Pred Class {j}: {pred:.2f}" for j, pred in enumerate(predicted_label)])
        axes[i, 0].text(0.5, -0.1, actual_text, transform=axes[i, 0].transAxes, verticalalignment='top', color='green')
        axes[i, 2].text(0.5, -0.1, predicted_text, transform=axes[i, 2].transAxes, verticalalignment='top', color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(epoch_dir, f'batch_{batch_index + 1}.png'))
    plt.close(fig)





def image_data_generator(files, batch_size=32, model=None, epoch=0):
    datagen = ImageDataGenerator(
        width_shift_range=0.3,
        fill_mode='nearest'
    )
    
    
    
    batch_index = 0
    while True:
        np.random.shuffle(files)
        batch_data = []
        batch_labels = []
        for file_path in files:
            img_data, label = load_image_and_label(file_path)
            batch_data.append(img_data)
            batch_labels.append(label)
            if len(batch_data) == batch_size:
                augmented_data, augmented_labels = next(datagen.flow(np.array(batch_data), np.array(batch_labels), batch_size=batch_size))
                if model is not None:
                    predictions = model.predict(augmented_data)
                    if batch_index % 20 == 0:
                        visualize_predictions(augmented_data, augmented_labels, predictions, batch_index, model, epoch_dir)
                yield augmented_data, augmented_labels
                batch_data = []
                batch_labels = []
                batch_index += 1
        if len(batch_data) > 0:
            augmented_data, augmented_labels = next(datagen.flow(np.array(batch_data), np.array(batch_labels), batch_size=len(batch_data)))
            if model is not None:
                predictions = model.predict(augmented_data)
                if batch_index % 20 == 0:
                    visualize_predictions(augmented_data, augmented_labels, predictions, batch_index, model, epoch_dir)
            yield augmented_data, augmented_labels
            batch_data = []
            batch_labels = []
            batch_index += 1






# 获取图像文件列表
def get_image_files(image_output_dir):
    image_files = []
    for root, _, files in os.walk(image_output_dir):
        for file in files:
            if file.endswith(".png"):
                image_files.append(os.path.join(root, file))
    return image_files


# 自定义的 TransformerBlock
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        self.embed_dim = embed_dim  # 添加这一行
        self.num_heads = num_heads  # 添加这一行
        self.ff_dim = ff_dim  # 添加这一行
        self.rate = rate  # 添加这一行
        super(TransformerBlock, self).__init__(**kwargs)  # 调整位置
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [tf.keras.layers.Dense(ff_dim, activation="relu"), tf.keras.layers.Dense(embed_dim)]
        )
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,  
            "num_heads": self.num_heads,  
            "ff_dim": self.ff_dim,  
            "rate": self.rate,  
        })
        return config


def build_model(input_shape, num_classes, transformer_blocks=4, freeze_backbone=True, weight_decay=1e-3):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.transpose(inputs, perm=[0, 3, 1, 2])  # (batch_size, height, width, channels) -> (batch_size, channels, height, width)
    vit_model = TFAutoModel.from_pretrained('./vit-base-patch16-224-in21k')
    if freeze_backbone:
        vit_model.trainable = False  # 冻结 backbone

    x = vit_model(pixel_values=x).last_hidden_state
    patch_dim = x.shape[-1]
    for _ in range(transformer_blocks):
        x = TransformerBlock(embed_dim=patch_dim, num_heads=4, ff_dim=patch_dim*4)(x)  # 设置 FFN 维度为 patch_dim 的 4 倍

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    model = tf.keras.Model(inputs, outputs)
    return model

image_files_prpd = get_image_files(image_output_dir)
categorized_files = categorize_files(image_files_prpd)
train_files = []
val_files = []

for category, files in categorized_files.items():
    train, val = train_test_split(files, test_size=0.2, random_state=42)
    train_files.extend(train)
    val_files.extend(val)

batch_size = 32
train_steps = len(train_files) // batch_size
val_steps = len(val_files) // batch_size


input_shape = (224, 224, 3)
num_classes = 4

strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

model = build_model(input_shape, num_classes)
# train_gen = image_data_generator(train_files, batch_size, model, epoch=0)
# val_gen = image_data_generator(val_files, batch_size, model, epoch=0)
def create_optimizer(model, hyp, opt):
    pg1, pg2 = [], []  # optimizer parameter groups
    for layer in model.layers:
        if hasattr(layer, 'bias') and isinstance(layer.bias, tf.Variable):
            pg2.append(layer.bias)  # biases
        if hasattr(layer, 'kernel') and isinstance(layer.kernel, tf.Variable):
            pg1.append(layer.kernel)  # apply decay

    # Define the optimizer
    if opt['adam']:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=hyp['lr0'], 
            beta_1=hyp['momentum'], 
            beta_2=0.999
        )
    else:
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=hyp['lr0'], 
            momentum=hyp['momentum'], 
            nesterov=True
        )

    return optimizer, pg1, pg2

# Hyperparameters and optimizer configuration
hyp = {
    'lr0': 1e-4,
    'momentum': 0.9,
    'weight_decay': 1e-4
}

opt = {
    'adam': True  # Change to False if you want to use SGD
}
optimizer, pg1, pg2 = create_optimizer(model, hyp, opt)

# 设置日志目录并清除旧日志
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)  # 清除旧日志
os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


class_names = ['Corona', 'Floating Potential', 'Free Particle', 'Insulation']
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./vit-BackBone-newGroupoptimizer-Ultra-refine-traning-2.h5', 
        save_best_only=True, 
        monitor='val_loss', 
        mode='min',
        save_weights_only=False
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=3, 
        mode='min'
    ),
    tensorboard_callback,
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.2, 
        patience=3, 
        min_lr=1e-6
    )
]

def add_regularization(model, weight_decay):
    if weight_decay and weight_decay > 0:
        for layer in model.layers:
            if hasattr(layer, 'kernel_regularizer') and layer.kernel_regularizer is None:
                layer.kernel_regularizer = tf.keras.regularizers.l2(weight_decay)
            if hasattr(layer, 'bias_regularizer') and layer.bias_regularizer is None:
                layer.bias_regularizer = tf.keras.regularizers.l2(weight_decay)
    return model

model = add_regularization(model, hyp['weight_decay'])
# with strategy.scope():
#     model = build_model(input_shape, num_classes)
#     model = add_regularization(model, hyp['weight_decay'])

#     optimizer, pg1, pg2 = create_optimizer(model, hyp, opt)

#     model.compile(
#         optimizer=optimizer,
#         loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#         metrics=[
#             tf.keras.metrics.BinaryAccuracy(),
#             tf.keras.metrics.AUC(),
#             tf.keras.metrics.Precision(),
#             tf.keras.metrics.Recall(),
#             tfa.metrics.F1Score(num_classes=num_classes, average='macro')
#         ]
#     )


model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

model.summary()

# 调试数据生成器
print("Checking data generator...")

# 测试生成一个批次的数据

num_epochs = 100

target_shape = (256, 256)


for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    # Create the directory for this epoch
    epoch_dir = os.path.join("visualizations-2", f"epoch_{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    train_gen = image_data_generator(train_files, batch_size, model, epoch)
    val_gen = image_data_generator(val_files, batch_size, model, epoch)

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=epoch + 1,  # Increment epoch number
        initial_epoch=epoch,  # Set initial epoch to resume from
        validation_data=val_gen,
        validation_steps=val_steps,
        callbacks=callbacks
    )
model.save('final_model_refine_traning-testing-2.h5')