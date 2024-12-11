#%%
# ライブラリの読み込み
import os

from skimage import io, exposure

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


import tensorflow as tf
from tensorflow.keras import layers, models

#%%
# ファイルパスとラベルを取得
INPUT_PATH = "../input/Satellite/"
label_file = INPUT_PATH + "train_master.tsv"
# "C:\Users\user026\Desktop\Satellite\src\input\Satellite\train_1"
# 学習用画像が格納されているディレクトリを指定する
data_dir = "../input/Satellite/train_1/train"
#%%
# 1. データの読み込み関数
# .tif 形式の画像を NumPy 配列に変換
def load_data(data_dir, label_file):
    labels = pd.read_csv(label_file, sep='\t')
    images = []
    for file_name in labels['file_name']:
        try:
            img_path = os.path.join(data_dir, file_name)
            img = io.imread(img_path).astype(np.float32)  # tifファイルの読み込み　# 必要なら型を統一
            images.append(img)
        except FileNotFoundError:
            print(f"ファイル '{img_path}' が見つかりません")\
        

    return np.array(images), np.array(labels['flag'].values)

# 2. データの正規化関数
# データの前処理
# 正規化 (0～1 の範囲にスケーリング)。

def normalize_with_rescale(image):
    # 値を0～1の範囲にスケーリング
    normalized_image = exposure.rescale_intensity(
        image, in_range="image", out_range=(0, 1)
    )
    return normalized_image

# 画像全体の正規化
def normalize_images(images):
    normalized_images = np.array([normalize_with_rescale(img) for img in images])
    return normalized_images


#%%
# データの読み込み、numpy変換、ラベルの対応づけ、正規化
# 3. データとラベルの対応付け
train_images, train_labels = load_data(data_dir, label_file)
#%%
train_images = normalize_images(train_images)

#%%
# 正規化できたかの確認

image = train_images[0]
plt.figure()
plt.title("Normalization")
plt.imshow(image[:, :, 0], cmap="gray")  # チャネル0を可視化
plt.colorbar()
print('最大値:', image.max())
print('最小値:', image.min())

#%%
# 4. データのシャッフルと分割
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.1, random_state=123)


# %%
# . skimage を使ったデータ拡張
# from skimage.transform import rotate
# from skimage.util import random_noise

# def augment_image_skimage(image):
#     augmented_images = []
#     augmented_images.append(rotate(image, angle=90))  # 90度回転
#     augmented_images.append(rotate(image, angle=180))  # 180度回転
#     augmented_images.append(random_noise(image, mode='gaussian'))  # ノイズ追加
#     return augmented_images


#%%
# 5. バッチ生成
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))

#%%
# バッチサイズ設定
batch_size = 32
train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

#%%
# 6. パイプラインの統合
print(f"Train dataset: {train_dataset}")
print(f"Validation dataset: {val_dataset}")