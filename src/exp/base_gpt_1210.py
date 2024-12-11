import os

from skimage import io, exposure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import layers, models


# %%
# 1. 画像の読み込み
# データの確認
def load_tif_image(file_path):
    image = io.imread(file_path)  # (height, width, channels)
    return image.astype(np.float32)  # 必要なら型を統一


file_path = "../input/Satellite/train_1/train/train_1.tif"

image = load_tif_image(file_path)


# %%
# 2. データの正規化
# データの前処理
# .tif 形式の画像を NumPy 配列に変換し、正規化 (0～1 の範囲にスケーリング)。
# 学習用データをシャッフルして、トレーニングデータと検証データに分割。
def normalize_with_rescale(image):
    # 値を0～1の範囲にスケーリング
    normalized_image = exposure.rescale_intensity(
        image, in_range="image", out_range=(0, 1)
    )
    return normalized_image


normalize_img = normalize_with_rescale(image)

# %%
# 正規化できたかの確認
# 正規化前
image = load_tif_image("example.tif")
plt.figure()
plt.title("Before Normalization")
plt.imshow(image[:, :, 0], cmap="gray")  # チャネル0を可視化
plt.colorbar()

# 正規化後
normalize_image = normalize_with_rescale(image)
plt.figure()
plt.title("After Normalization")
plt.imshow(normalize_image[:, :, 0], cmap="gray")
plt.colorbar()

plt.show()


# %%
# 3. ラベルの対応付け
def load_dataset_skimage(file_paths, labels):
    images = []
    for file_path in file_paths:
        image = load_tif_image(file_path)
        image = normalize_image(image)  # 正規化
        images.append(image)
    return np.array(images), np.array(labels)


# %% 4. データセットの準備
# 学習用、検証用、テスト用に分割します。
# ファイルパスとラベルを取得
INPUT_PATH = "../input/Satellite/"


label_df = pd.read_csv(INPUT_PATH + "train_master.tsv", sep="\t")

# 学習用画像が格納されているディレクトリを指定する
data_path = "../input/Satellite/train_1/train/"



file_paths = [data_path + fn for fn in label_df["file_name"]]
labels = label_df["flag"].values

# シャッフルと分割
train_files, val_files, train_labels, val_labels = train_test_split(
    file_paths, labels, test_size=0.1, random_state=123
)

# 画像とラベルの読み込み
train_images, train_labels = load_dataset_skimage(train_files, train_labels)
val_images, val_labels = load_dataset_skimage(val_files, val_labels)


# %%
# 5. skimage を使ったデータ拡張
# skimage の変換機能を使ってデータ拡張を行うことも可能です。
# from skimage.transform import rotate
# from skimage.util import random_noise

# def augment_image_skimage(image):
#     augmented_images = []
#     augmented_images.append(rotate(image, angle=90))  # 90度回転
#     augmented_images.append(rotate(image, angle=180))  # 180度回転
#     augmented_images.append(random_noise(image, mode='gaussian'))  # ノイズ追加
#     return augmented_images


#%%
def data_generator_skimage(file_paths, labels, batch_size):
    num_samples = len(file_paths)
    while True:  # 無限ループでバッチを生成
        for offset in range(0, num_samples, batch_size):
            batch_files = file_paths[offset:offset + batch_size]
            batch_labels = labels[offset:offset + batch_size]

            images = []
            for file_path in batch_files:
                image = load_tif_image(file_path)
                image = normalize_image(image)
                images.append(image)

            yield np.array(images), np.array(batch_labels)

# %%
# IoU
class MeanIoU(tf.keras.metrics.Metric):
    def __init__(self, name="mean_iou", **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(y_pred > 0.5, tf.bool)

        intersection = tf.reduce_sum(
            tf.cast(tf.logical_and(y_true, y_pred), tf.float32)
        )
        union = tf.reduce_sum(tf.cast(tf.logical_or(y_true, y_pred), tf.float32))

        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return tf.math.divide_no_nan(self.intersection, self.union)

    def reset_states(self):
        self.intersection.assign(0)
        self.union.assign(0)


# ベースラインモデル（CNN）
def create_baseline_model(input_shape):
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(1, activation="sigmoid"),  # バイナリ分類
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=[MeanIoU()])
    return model

#%%
train_gen = data_generator_skimage(train_files, train_labels, batch_size=64)
val_gen = data_generator_skimage(val_files, val_labels, batch_size=64)

model.fit(train_gen,
          steps_per_epoch=len(train_files) // 64,
          validation_data=val_gen,
          validation_steps=len(val_files) // 64,
          epochs=10)
# %%
# 学習
model = create_baseline_model(
    input_shape=(32, 32, 7)
)  # Landsatデータのチャネル数に合わせる
history = model.fit(
    train_images,
    train_labels,
    validation_data=(val_images, val_labels),
    epochs=10,
    batch_size=64,
)


# %%
# 予測、評価
predictions = model.predict(test_images)
predictions_binary = (predictions > 0.5).astype(int)

# 提出用ファイル作成
submission = pd.DataFrame(
    {"file_name": test_file_names, "label": predictions_binary.flatten()}
)
submission.to_csv("submission.tsv", sep="\t", header=False, index=False)
