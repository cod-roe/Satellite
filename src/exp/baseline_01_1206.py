# %% [markdown]
## チュートリアル！
# =================================================
# ベースライン作成：
# chainerではなくtensorflowを使用
# 畳み込み層2つ、後半が全結合層2つ
# Sequential APIでまずは行う。
# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
# import json
import logging

# import re
import os
from skimage import io, exposure
import sys
import pickle
from IPython.display import display
import warnings
import zipfile

import numpy as np
import pandas as pd


# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


# lightGBM
import lightgbm as lgb

# sckit-learn
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
# import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, train_test_split  # , KFold
# from sklearn.metrics import mean_squared_error,accuracy_score, roc_auc_score ,confusion_matrix
# %%
# import keras
# from keras import layers

import tensorflow as tf

import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Sequential, Model  # type:ignore

from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization  # type:ignore

from tensorflow.keras.layers import Flatten, Conv2D, MaxPooling2D  # type:ignore
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
)  # type:ignore
from tensorflow.keras.optimizers import Adam, SGD  # type:ignore

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %%
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

# import rasterio
# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 0  # スプレッドシートAの番号

######################
# Data #
######################
comp_name = "Satellite"
# 評価：IOU 回帰 分類

skip_run = False  # 飛ばす->True，飛ばさない->False

######################
# filename
######################
# vscode用
abs_path = os.path.abspath(__file__)  # /tmp/work/src/exp/_.py'
name = os.path.splitext(os.path.basename(abs_path))[0]
# Google Colab等用（取得できないためファイル名を入力）
# name = 'run001'

######################
# set dirs #
######################
DRIVE = os.path.dirname(os.getcwd())  # このファイルの親(scr)
INPUT_PATH = f"../input/{comp_name}/"  # 読み込みファイル場所
OUTPUT = os.path.join(DRIVE, "output")
OUTPUT_EXP = os.path.join(OUTPUT, name)  # 情報保存場所
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")  # 学習済みモデル保存

######################
# Dataset #
######################
# target_columns = "bikes_available"
# sub_index = "id"

######################
# ハイパーパラメータの設定
######################


# %%
# Utilities #
# =================================================


# 今の日時
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now


# stdout と stderr をリダイレクトするクラス
class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass


# make dirs
# =================================================
def make_dirs():
    for d in [EXP_MODEL]:
        os.makedirs(d, exist_ok=True)
    print("フォルダ作成完了")


# ファイルの確認
# =================================================
def file_list(input_path):
    file_list = []
    for dirname, _, _filenames in os.walk(input_path):
        for i, _datafilename in enumerate(_filenames):
            print("=" * 20)
            print(i, _datafilename)
            file_list.append([_datafilename, os.path.join(dirname, _datafilename)])
    file_list = pd.DataFrame(file_list, columns=["ファイル名", "ファイルパス"])
    display(file_list)
    return file_list


# メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:  # noqa: E721
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astypez(np.float64)
        else:
            pass

    end_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage after optimization is: {end_mem:.2f} MB")
    print(f"Decreased by {100*((start_mem - end_mem) / start_mem):.2f}%")

    return df


# ファイルの読み込み
# =================================================
def load_data(file_index):
    # file_indexを引数に入力するとデータを読み込んでくれる
    if file_list["ファイル名"][file_index][-3:] == "csv":
        print(f"読み込んだファイル：{file_list['ファイル名'][file_index]}")
        df = reduce_mem_usage(
            pd.read_csv(file_list["ファイルパス"][file_index], encoding="shift-jis")
        )
        print(df.shape)
        display(df.head())

    elif file_list["ファイル名"][file_index][-3:] == "pkl" or "pickle":
        print(f"読み込んだファイル：{file_list['ファイル名'][file_index]}")
        df = reduce_mem_usage(pd.read_pickle(file_list["ファイルパス"][file_index]))
        print(df.shape)
        display(df.head())
    return df


# %%
# 前処理の定義 カテゴリ変数をcategory型に
# =================================================
def data_pre00(df):
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].astype("category")
    print("カテゴリ変数をcategory型に変換しました")
    df.info()
    return df


# %% [markdown]
## Main 分析start!
# ==========================================================
# %%
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
sns.set(font="IPAexGothic")
#!%matplotlib inline
pd.options.display.float_format = "{:10.4f}".format  # 表示桁数の設定

# フォルダの作成
make_dirs()
# ファイルの確認
file_list = file_list(INPUT_PATH)

# utils
# ログファイルの設定
logging.basicConfig(
    filename=f"{OUTPUT_EXP}/log_{name}.txt", level=logging.INFO, format="%(message)s"
)
# ロガーの作成
logger = logging.getLogger()


# 出力表示数増やす
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)


# %%
# zipファイルの扱い
# zip_file_path = "<zipファイルのパス>"
# target_file_path = "<zipファイルの中にある目的のtxtファイル>"

# with zipfile.ZipFile(file_path, "r") as zip_ref:
#     with zip_ref.open(target_file_path, "r") as file:
#         for line in file:
#             line_decoded = line.decode("utf-8")
#             # line_decodedに対してやりたいことを追記する


# import zipfile

# # zipファイルを読み込む
# zip_f = zipfile.ZipFile('sample.zip')

# # zipファイルの中身を取得
# lst = zip_f.namelist()

# # 取得したzipファイルの中身を出力
# print(lst)


# %% ファイルの読み込み
# Load Data
# =================================================
#  train_1
image_path = "../input/Satellite/train_1/train/train_1.tif"
image = io.imread(image_path)

print(image.shape)

# %%
# 画像データの確認、可視化
fig, (ax0, ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(
    nrows=1, ncols=7, figsize=(10, 3)
)
ax0.imshow(image[:, :, 0])
ax0.set_title("1")
ax0.axis("off")
ax0.set_adjustable("box")

ax1.imshow(image[:, :, 1])
ax1.set_title("B")
ax1.axis("off")
ax1.set_adjustable("box")

ax2.imshow(image[:, :, 2])
ax2.set_title("G")
ax2.axis("off")
ax2.set_adjustable("box")

ax3.imshow(image[:, :, 3])
ax3.set_title("R")
ax3.axis("off")
ax3.set_adjustable("box")

ax4.imshow(image[:, :, 4])
ax4.set_title("5")
ax4.axis("off")
ax4.set_adjustable("box")

ax5.imshow(image[:, :, 5])
ax5.set_title("6")
ax5.axis("off")
ax5.set_adjustable("box")

ax6.imshow(image[:, :, 6])
ax6.set_title("7")
ax6.axis("off")
ax6.set_adjustable("box")

fig.tight_layout()

# %%
data_df = pd.read_csv(INPUT_PATH + "train_master.tsv", sep="\t")
data_df.head()

# %%
# 3.画像データの前処理 正規化
image_rescaled = exposure.rescale_intensity(image)
# %%
# def normalize_image(image):
#     return image / 65535.0  # Landsat-8の値域は0～65535
# image_rescaled2 = normalize_image(image)
# %%


# %%
# 前処理を行う前
print("最大値：", image.max())
print("最大値：", image.min())

# %%
# 前処理を行った後
print("最大値：", image_rescaled.max())
print("最大値：", image_rescaled.min())


#
# %%
# 1枚の画像データを渡して画像処理後のデータを出力する
def preprocess(image, mode="train"):
    """
    image: shape = (h, w, channel)を想定。
    mode: 'train', 'val', 'test'を想定。
    """
    if mode == "train":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    elif mode == "val":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    elif mode == "test":
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)
    else:
        # その他いろいろな前処理メソッドを実装してみてください
        if image.max() != image.min():
            image = exposure.rescale_intensity(image)

    return image


# %%
tr1_data_path = "../input/Satellite/train_1/train"


# %%
def generate(data_path, data_df, mode="train"):
    images = []
    if mode == "train" or mode == "val":
        labels = []
    for data in data_df.iterrows():
        try:
            im_path = os.path.join(data_path, data[1]["file_name"])
            image = io.imread(im_path)

            # preprocess image
            image = preprocess(image, mode=mode)
            # image = image.transpose((2, 0, 1))

            if mode == "train" or mode == "val":
                labels.append(data[1]["flag"])

            images.append(image)
        except FileNotFoundError:
            print(f"ファイル '{data[1]['file_name']}' が見つかりません")

    images = np.array(images)
    if mode == "train" or mode == "val":
        labels = np.array(labels)

        return images, labels
    else:
        return images


# %%
images, labels = generate(tr1_data_path, data_df, mode="train")
# train_images, train_labelsに後で変更しておく

# %%
print(len(images))
print(len(labels))

# %%
# データを学習用と検証用に分ける関数
# def split_data(data, ratio=0.95):
#     train_index = np.random.choice(data.index, int(len(data)*ratio), replace=False)
#     val_index = list(set(data.index).difference(set(train_index)))
#     train = data.iloc[train_index].copy()
#     val = data.iloc[val_index].copy()

#     return train, val

# %%
# 4. データのシャッフルと分割

# %%
x_tr, x_va, y_tr, y_va = train_test_split(
    images, labels, test_size=0.1, random_state=123
)

# %%
datagen = ImageDataGenerator()
train_generator = datagen.flow(x_tr, y_tr, batch_size=64)
valid_generator = datagen.flow(x_va, y_va, batch_size=64)


# %%
# 今回の評価関数となっているIOU(intersection over union)を実装します。正解データy_trueと予測されたデータy_predを渡してIOUを返します。
def IOU(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    p_true_index = np.where(y_true == 1)[0]
    p_pred_index = np.where(y_pred == 1)[0]
    union = set(p_true_index).union(set(p_pred_index))
    intersection = set(p_true_index).intersection(set(p_pred_index))
    if len(union) == 0:
        return 0
    else:
        return len(intersection) / len(union)


# %%
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


# %%
# モデリング
model = Sequential()

model.add(Conv2D(32, (3, 3), strides=2, activation="relu", input_shape=(32, 32, 7)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), strides=2, activation="relu"))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()


# %%
model.compile(optimizer="Adam", loss="binary_crossentropy", metrics=[MeanIoU()])


history = model.fit(
    x_tr,
    y_tr,  # train_generator,
    validation_data=(x_va, y_va),
    epochs=20,
    batch_size=64,
    callbacks=[
        ModelCheckpoint(
            filepath=EXP_MODEL + "/model_cnn01.weights.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
    ],
    verbose=1,
)

# %%
test_loss, test_iou = model.evaluate(x_va, y_va)
print(f"IOU数値{test_iou:.2}")


# %%
param = [["IOU", "mean_iou", "val_mean_iou"], ["誤差", "loss", "val_loss"]]

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc="best")
plt.show()

# %%
# train2学習
# =================================================

tr2_data_path = "../input/Satellite/train_2/train"

images_2, labels_2 = generate(tr2_data_path, data_df, mode="train")

x_tr2, x_va2, y_tr2, y_va2 = train_test_split(
    images_2, labels_2, test_size=0.1, random_state=123
)

history = model.fit(
    x_tr2,
    y_tr2,  # train_generator,
    validation_data=(x_va2, y_va2),
    epochs=20,
    batch_size=64,
    callbacks=[
        ModelCheckpoint(
            filepath=EXP_MODEL + "/model_cnn02.weights.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
    ],
    verbose=1,
)

# %%
test_loss, test_iou = model.evaluate(x_va, y_va)
print(f"IOU数値{test_iou:.2}")


# %%
param = [["IOU", "mean_iou", "val_mean_iou"], ["誤差", "loss", "val_loss"]]

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc="best")
plt.show()

# %%
# train3学習
# =================================================

tr3_data_path = "../input/Satellite/train_3/train"

images_3, labels_3 = generate(tr3_data_path, data_df, mode="train")
x_tr3, x_va3, y_tr3, y_va3 = train_test_split(
    images_3, labels_3, test_size=0.1, random_state=123
)

history = model.fit(
    x_tr3,
    y_tr3,  # train_generator,
    validation_data=(x_va3, y_va3),
    epochs=20,
    batch_size=64,
    callbacks=[
        ModelCheckpoint(
            filepath=EXP_MODEL + "/model_cnn03.weights.h5",
            monitor="val_loss",
            mode="min",
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            min_delta=0,
            patience=10,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
    ],
    verbose=1,
)

# %%
test_loss, test_iou = model.evaluate(x_va, y_va)
print(f"IOU数値{test_iou:.2}")


# %%
param = [["IOU", "mean_iou", "val_mean_iou"], ["誤差", "loss", "val_loss"]]

plt.figure(figsize=(10, 4))
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.title(param[i][0])
    plt.plot(history.history[param[i][1]], "o-")
    plt.plot(history.history[param[i][2]], "o-")
    plt.xlabel("学習回数")
    plt.legend(["訓練", "テスト"], loc="best")
plt.show()


# %%
# 推論処理
# =================================================


# test1
# =================================================
# 1-1.ファイル呼び込み
test_path1 = "../input/Satellite/test_1/test"
test1_sub = pd.DataFrame()
test1_sub["file_name"] = os.listdir(test_path1)
test1_sub.head()
# %%
# 1-2.前処理

images_test1 = generate(test_path1, test1_sub, mode="test")


# %%
# 1-3モデル評価
y_va_pred1 = model.predict(images_test1, batch_size=64, verbose=1)

test1_sub["pred"] = y_va_pred1
test1_sub.head()


# %%
# test2
# =================================================
# 2-1.ファイル呼び込み
test_path2 = "../input/Satellite/test_2/test"
test2_sub = pd.DataFrame()
test2_sub["file_name"] = os.listdir(test_path2)
# %%
# 2-2.前処理
images_test2 = generate(test_path2, test2_sub, mode="test")


# %%
# 2-3.モデル評価
y_va_pred2 = model.predict(images_test2, batch_size=64, verbose=1)

test2_sub["pred"] = y_va_pred2
test2_sub.head()


# %%
# test3
# =================================================
# 3-1.ファイル呼び込み

test_path3 = "../input/Satellite/test_3/test"
test3_sub = pd.DataFrame()
test3_sub["file_name"] = os.listdir(test_path3)
test3_sub.head()
# %%
# 3-2.前処理
images_test3 = generate(test_path3, test3_sub, mode="test")


# %%
# 3-3モデル評価
y_va_pred3 = model.predict(images_test3, batch_size=64, verbose=1)

test3_sub["pred"] = y_va_pred3

test3_sub.head()


# %%
# test4
# =================================================
# 4-1.ファイル呼び込み

test_path4 = "../input/Satellite/test_4/test"
test4_sub = pd.DataFrame()
test4_sub["file_name"] = os.listdir(test_path4)
test4_sub.head()
# %%
# 4-2.前処理
images_test4 = generate(test_path4, test4_sub, mode="test")


# %%
# 4-3モデル評価
y_va_pred4 = model.predict(images_test4, batch_size=64, verbose=1)

test4_sub["pred"] = y_va_pred4

test4_sub.head()


# %%

df_submit = pd.concat([test1_sub, test2_sub, test3_sub, test4_sub], axis=0)

df_submit.head()
# %%
df_submit.shape
# %%
df_submit.describe()
# %%
# サンプルの確認
sample_df = pd.read_csv(INPUT_PATH + "sample_submit.tsv", sep="\t")
sample_df.shape
# %%
df_submit["pred"] = (np.where(df_submit["pred"] >= 0.5, 1, 0)).astype(int)

df_submit.sort_values("file_name", inplace=True)
df_submit.head()


df_submit.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.tsv", index=False, header=False)
)


# %%
# テスト3までのファイル
df_submit.to_csv(os.path.join(OUTPUT_EXP, f"submission_{name}.csv"))

# %%


# %%


# %%
# %%
# def seed_everything(seed):
#     import random

#     random.seed(seed)
#     os.environ["PYTHONHASHSEED"] = str(seed)
#     np.random.seed(seed)
#     tf.random.set_seed(seed)
#     session_conf = tf.compat.v1.ConfigProto(
#         intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
#     )
#     sess = tf.compat.v1.Session(
#         graph=tf.compat.v1.get_default_graph(), config=session_conf
#     )
#     # tf.compat.v1.keras.backend.set_session(sess)
#     # K.set_session(sess)

# seed_everything(seed=123)


# %%
# 8-61: 推論用関数
def predict_tf(
    input_x,
    input_id,
    list_nfold=[1, 2, 3],
):
    # 推論値を入れる変数の作成
    test_pred = np.zeros((len(input_x), 4))

    for nfold in list_nfold:
        # modelのロード
        filepath = os.path.join(EXP_MODEL, f"model_cnn{nfold}.weights.h5")
        model = create_model(col_num=col_num, col_cat=col_cat, show=False)
        model.load_weights(filepath)

        # validの推論値取得
        pred = model.predict([x_num_test, x_cat_test], batch_size=512, verbose=0)
        test_pred += pred / len(list_nfold)

    # 予測値の格納
    df_test_pred = pd.concat(
        [
            input_id,
            pd.DataFrame(
                test_pred, columns=["target1", "target2", "target3", "target4"]
            ),
        ],
        axis=1,
    )

    return df_test_pred


# test4kopi
# =================================================
# 4-1.ファイル呼び込み

test_path4 = "../input/Satellite/test_4/test"
test4_sub = pd.DataFrame()
test4_sub["file_name"] = os.listdir(test_path4)
test4_sub.head()
# %%
# 4-2.前処理
images_test4 = generate(test_path4, test4_sub, mode="test")


# %%
# 4-3モデル評価
y_va_pred4 = model.predict(images_test4, batch_size=64, verbose=1)

test4_sub["pred"] = y_va_pred4

test4_sub.head()


# %%

df_submit = pd.concat([test1_sub, test2_sub, test3_sub, test4_sub], axis=0)

df_submit.head()
# %%
df_submit.shape
# %%
df_submit.describe()
# %%
# サンプルの確認
sample_df = pd.read_csv(INPUT_PATH + "sample_submit.tsv", sep="\t")
sample_df.shape
# %%
df_submit["pred"] = (np.where(df_submit["pred"] >= 0.5, 1, 0)).astype(int)

df_submit.sort_values("file_name", inplace=True)
df_submit.head()


df_submit.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.tsv", index=False, header=False)
)


# %%
# テスト3までのファイル
df_submit.to_csv(os.path.join(OUTPUT_EXP, f"submission_{name}.csv"))

#%%
# 8-55: ニューラルネットワークのモデル定義
def create_model(
    col_num=["heightInches", "weight"],
    col_cat=["playerId", "teamId", "dayofweek"],
    show=False,
):
    input_num = Input(shape=(len(col_num),))
    input_cat = Input(shape=(len(col_cat),))

    # numeric
    x_num = input_num  # Dense(30,activation='relu)(input_num)
    # category
    for i, col in enumerate(col_cat):
        tmp_cat = input_cat[:, i]
        input_dim = dict_cat[col]["num_label"]
        output_dim = int(input_dim / 2)
        tmp_cat = Embedding(input_dim=input_dim, output_dim=output_dim)(tmp_cat)
        tmp_cat = Dropout(0.2)(tmp_cat)
        tmp_cat = Flatten()(tmp_cat)
        if i == 0:
            x_cat = tmp_cat
        else:
            x_cat = Concatenate()([x_cat, tmp_cat])

    # concat
    x = Concatenate()([x_num, x_cat])

    x = Dense(128, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    output = Dense(4, activation="linear")(x)

    model = Model(inputs=[input_num, input_cat], outputs=output)
    model.compile(optimizer="Adam", loss="mae", metrics=["mae"])

    if show:
        print(model.summary())
    else:
        return model


# モデリング
model = Sequential()

model.add(Conv2D(32, (3, 3), strides=2, activation="relu", input_shape=(32, 32, 7)))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), strides=2, activation="relu"))
model.add(MaxPooling2D((2, 2), strides=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.summary()



# %%
# 8-56:モデル構造の確認
create_model(col_num=col_num, col_cat=col_cat, show=True)