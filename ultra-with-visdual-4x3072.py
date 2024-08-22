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
from tensorflow.keras.callbacks import Callback, TensorBoard
import tensorflow_addons as tfa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

#########################################
# 这是一个实验性脚本，本质和之前的脚本一样，只是在模型的构建上进行了一些调整
# 1. 使用了更大的 Transformer 模型
# 2. 添加了 CNN 模块
# 3. 使用了更大的全连接层
# 4. 使用了更大的学习率
# 5. 使用了更大的 batch size
# 6. 使用了更大的权重衰减
# 7. 使用了更大的 Transformer Blocks
# 8. 使用了更大的 Transformer Block 大小
# 9. 使用了更大的 Transformer Block 的全连接层

# 同时通过 transformer block 和 CNN 模块的组合，可以更好地捕获图像的空间信息和全局信息
# 目前来看普通的CNN卷积无法快速拟合，未来应考虑使用 GLU 模块
# 同时也是为了在论文中展示CNN和ViT的 效果的区别。
# 多数情况下该代码没用

#############################################



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

import matplotlib.pyplot as plt

def visualize_predictions(batch_data, batch_labels, batch_predictions, batch_index, model, epoch_dir):
    fig, axes = plt.subplots(min(len(batch_data), 4), 4, figsize=(20, 5 * min(len(batch_data), 4)))

    for i in range(min(len(batch_data), 4)):
        # Original Image
        axes[i, 0].imshow(np.clip(batch_data[i], 0, 1))
        axes[i, 0].set_title("Original Image")

        # After ViT
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('vision_transformer').output)
        vit_output = intermediate_model.predict(np.expand_dims(batch_data[i], axis=0))
        vit_output_patches = vit_output[0, 1:, :]  # Remove the CLS token and reshape patches
        patch_size = 16
        num_patches = 14  # Since 224/16 = 14
        projection_dim = vit_output_patches.shape[-1]
        
        # Reconstruct the image from patches
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

         # After CNN
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('tf.concat').output)
        combined_features_output = intermediate_model.predict(np.expand_dims(batch_data[i], axis=0))
        cnn_output = combined_features_output[:, :, projection_dim:]  # 提取 CNN 的部分
        #print("cnn_output.shape", cnn_output.shape)
        # 处理 cnn_output 以显示为色块
         # 处理 cnn_output 以显示为色块
        cnn_patch_size = 16
        cnn_num_patches = 14  # Since 224/16 = 14
        cnn_channels = cnn_output.shape[-1]  # Number of CNN output channels

        # Normalize cnn_output to [0, 1] for visualization
        cnn_output_min = np.min(cnn_output)
        cnn_output_max = np.max(cnn_output)
        cnn_output_normalized = (cnn_output - cnn_output_min) / (cnn_output_max - cnn_output_min)
        
        cnn_output_reshaped = cnn_output_normalized.reshape((cnn_num_patches, cnn_num_patches, cnn_channels))
        
        # 将每个补丁的前三个通道取出来进行可视化
        cnn_reconstructed_image = np.zeros((cnn_num_patches * cnn_patch_size, cnn_num_patches * cnn_patch_size, 3))
        
        for j in range(cnn_num_patches):
            for k in range(cnn_num_patches):
                patch = cnn_output_reshaped[j, k, :3]  # 取前三个通道
                cnn_reconstructed_image[j * cnn_patch_size:(j + 1) * cnn_patch_size, k * cnn_patch_size:(k + 1) * cnn_patch_size, :] = patch

        axes[i, 2].imshow(np.clip(cnn_reconstructed_image, 0, 1))
        axes[i, 2].set_title("After CNN")

        # After Transformer Block
        intermediate_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer('transformer_block_2').output)
        transformer_output = intermediate_model.predict(np.expand_dims(batch_data[i], axis=0))
        transformer_output_patches = transformer_output[0, 1:, :]  # Remove the CLS token if present
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
        
        axes[i, 3].imshow(np.clip(reconstructed_image, 0, 1))
        axes[i, 3].set_title("After Transformer Block")
        
        # Show predicted probabilities
        actual_label = batch_labels[i]
        predicted_label = batch_predictions[i]
        
        # Display probabilities for each class
        prob_text = "\n".join([f"Class {j}: {pred:.2f}" for j, pred in enumerate(predicted_label)])
        axes[i, 3].text(1.05, 0.5, prob_text, transform=axes[i, 3].transAxes, verticalalignment='center')
        
        # Display actual vs predicted labels
        actual_text = "\n".join([f"Actual Class {j}: {act}" for j, act in enumerate(actual_label)])
        predicted_text = "\n".join([f"Pred Class {j}: {pred:.2f}" for j, pred in enumerate(predicted_label)])
        axes[i, 0].text(0.5, -0.1, actual_text, transform=axes[i, 0].transAxes, verticalalignment='top', color='green')
        axes[i, 3].text(0.5, -0.1, predicted_text, transform=axes[i, 3].transAxes, verticalalignment='top', color='red')

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
            "embed_dim": self.embed_dim,  # 添加这一行
            "num_heads": self.num_heads,  # 添加这一行
            "ff_dim": self.ff_dim,  # 添加这一行
            "rate": self.rate,  # 修改这一行
        })
        return config

class VisionTransformer(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim, transformer_layers, num_heads, ff_dim, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.transformer_layers = transformer_layers
        self.num_heads = num_heads
        self.ff_dim = ff_dim

        self.patch_embeddings = tf.keras.layers.Dense(self.projection_dim)
        self.position_embeddings = tf.keras.layers.Embedding(input_dim=self.num_patches + 1, output_dim=self.projection_dim)
        self.cls_token = self.add_weight("cls_token", shape=[1, 1, self.projection_dim])

        self.transformer_blocks = [TransformerBlock(embed_dim=self.projection_dim, num_heads=self.num_heads, ff_dim=self.ff_dim) for _ in range(self.transformer_layers)]

    def call(self, patches, training):
        batch_size = tf.shape(patches)[0]
        cls_tokens = tf.broadcast_to(self.cls_token, [batch_size, 1, self.projection_dim])
        patches_embeddings = self.patch_embeddings(patches)
        patches_embeddings += self.position_embeddings(tf.range(self.num_patches))
        tokens = tf.concat([cls_tokens, patches_embeddings], axis=1)
        for transformer in self.transformer_blocks:
            tokens = transformer(tokens, training)
        return tokens

    def get_config(self):
        config = super(VisionTransformer, self).get_config()
        config.update({
            "num_patches": self.num_patches,
            "projection_dim": self.projection_dim,
            "transformer_layers": self.transformer_layers,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_cnn(input_shape):
    cnn_input = tf.keras.Input(shape=input_shape, name='cnn_input')
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(cnn_input)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    cnn_model = tf.keras.Model(cnn_input, x, name='cnn_model')
    return cnn_model

def build_vit(input_shape, num_patches, projection_dim, transformer_layers, num_heads, ff_dim):
    inputs = tf.keras.Input(shape=input_shape)
    patches = tf.image.extract_patches(
        images=inputs,
        sizes=[1, 16, 16, 1],
        strides=[1, 16, 16, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patches = tf.reshape(patches, [-1, num_patches, 16*16*input_shape[-1]])
    vit = VisionTransformer(num_patches=num_patches, projection_dim=projection_dim, transformer_layers=transformer_layers, num_heads=num_heads, ff_dim=ff_dim)
    return vit(patches)

def build_model(input_shape, num_classes, transformer_blocks=4, weight_decay=1e-3):
    model_input = tf.keras.Input(shape=input_shape, name='model_input')
    
    num_patches = (input_shape[0] // 16) * (input_shape[1] // 16)
    projection_dim = 768  # Set projection_dim to 768
    transformer_layers = 4
    num_heads = 4
    ff_dim = 3072

    patches = tf.image.extract_patches(
        images=model_input,
        sizes=[1, 16, 16, 1],
        strides=[1, 16, 16, 1],
        rates=[1, 1, 1, 1],
        padding="VALID"
    )
    patches = tf.reshape(patches, [-1, num_patches, 16 * 16 * input_shape[-1]])

    vit = VisionTransformer(num_patches=num_patches, projection_dim=projection_dim, transformer_layers=transformer_layers, num_heads=num_heads, ff_dim=ff_dim, name='vision_transformer')
    vit_output = vit(patches)
    
    # 添加 CNN 模块
    cnn_model = build_cnn(input_shape)
    cnn_output = cnn_model(model_input)
    
    # 调整 cnn_output 形状
    cnn_output = tf.keras.layers.Dense(projection_dim)(cnn_output)  # 添加全连接层，使其具有相同的通道数
    cnn_output = tf.expand_dims(cnn_output, axis=1)  # 添加一个维度
    cnn_output = tf.tile(cnn_output, [1, num_patches, 1])  # 将其重复以匹配 vit_output 的形状
    
    # 组合 ViT 和 CNN 输出
    combined_features = tf.concat([vit_output[:, 1:, :], cnn_output], axis=-1)  # 保留所有补丁并连接
    
    for i in range(transformer_blocks):
        combined_features = TransformerBlock(embed_dim=combined_features.shape[-1], num_heads=4, ff_dim=combined_features.shape[-1]*4, name=f'transformer_block_{i}')(combined_features)

    x = tf.keras.layers.GlobalAveragePooling1D()(combined_features)
    x = tf.keras.layers.Dropout(0.1)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    model = tf.keras.Model(inputs=model_input, outputs=outputs)
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

log_dir = os.path.join(os.getcwd(), "logs", "fit")
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
class_names = ['Corona', 'Floating Potential', 'Free Particle', 'Insulation']
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='./vit-BackBone-newGroupoptimizer-Ultra-self-vit.h5', 
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

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tfa.metrics.F1Score(num_classes=num_classes, average='macro')])

model.summary()

# 调试数据生成器
print("Checking data generator...")

num_epochs = 100
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    # Create the directory for this epoch
    epoch_dir = os.path.join("visualizations-self-vit", f"epoch_{epoch+1}")
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
model.save('final_model_refine_self-vit.h5')
