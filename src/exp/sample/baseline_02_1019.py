# %% [markdown]
## DeepAR使ってみる！
# =================================================
# DeepARを使う
# %%
# %%
# ライブラリ読み込み
# =================================================
import datetime as dt
from datetime import timedelta
from datetime import datetime

# import gc
from IPython.display import display

# import json
import logging
import math
import os
import pickle

# import re
import sys
from statistics import mean
import warnings
# import zipfile

import numpy as np
import pandas as pd


# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib


# sckit-learn
# 前処理
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
)  # LabelEncoder, OneHotEncoder

# バリデーション、評価測定
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import mean_squared_error
# ,accuracy_score, roc_auc_score ,confusion_matrix


# パラメータチューニング
# import optuna


# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.optimizers import Adam


import time
import copy


# 正義
# ==========================
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import (
    Baseline,
    DeepAR,
    TimeSeriesDataSet,
    TemporalFusionTransformer,
)  # TFT
from pytorch_forecasting.data import NaNLabelEncoder

# from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import (
    RMSE,
    MAE,
    SMAPE,
    NormalDistributionLoss,
)

import pytorch_forecasting as ptf
from lightning.pytorch.tuner import Tuner

# =============================


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 29  # スプレッドシートAの番号

######################
# Data #
######################
comp_name = "Share_bicycle"
# 評価：RMSE（二乗平均平方根誤差） 回帰

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
# lgbm初期値
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}


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


# %%
# ファイルの読み込み
# =================================================
def load_data(file_index):
    # file_indexを引数に入力するとデータを読み込んでくれる
    if file_list["ファイル名"][file_index][-3:] == "csv":
        print(f"読み込んだファイル：{file_list['ファイル名'][file_index]}")
        df = reduce_mem_usage(pd.read_csv(file_list["ファイルパス"][file_index]))
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

# os.chdir("../../..")
# %% ファイルの読み込み
# Load Data
# =================================================
# status
status_df = load_data(2)
# status_df = pd.read_csv(file_list["ファイルパス"][2])


# %%
status_df["date"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    + " "
    + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date"] = pd.to_datetime(status_df["date"])  # .dt.strftime("%Y-%m-%d %H")


# date yyyy-mm-dd
status_df["date2"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    # + " "
    # + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date2"] = pd.to_datetime(status_df["date2"])  # .dt.strftime("%Y-%m-%d")


# dayofweek 曜日カラムの作成 Sun
status_df["dayofweek"] = status_df["date2"].dt.strftime("%a")

status_df.head()
# %%
status_df = status_df.sort_values(by=["station_id", "date"])

# %%
# 0時の特徴量
t = status_df.groupby(["station_id", "date2"]).first()["bikes_available"].reset_index()

t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
t.columns = ["staion_id", "date2", "bikes_available_at0"]

status_df["bikes_available_at0"] = t["bikes_available_at0"].astype("float16")


# %%
# データセット作成
main_df = status_df[
    [
        "date",
        "hour",
        "station_id",
        "bikes_available",
        "bikes_available_at0",
        "dayofweek",
        "predict",
    ]
]
main_df.head()


# %%
# 学習用のデータフレーム作成
train_dataset_df = main_df[main_df["date"] < "2014-09-01"]
# 評価用のデータフレーム作成（使用するモデルの関係上、前日のデータが必要なため2014-08-31から取得）
evaluation_dataset_df = main_df[main_df["date"] >= "2014-08-25"]

# %%
train_dataset_df.head()


# %%
train_dataset_df = train_dataset_df[train_dataset_df["date"] >= "2014-05-01"]

# %%

# 各ステーション毎に、欠損値を後の値で埋める
train_dataset_df_new = pd.DataFrame()
for station_id in train_dataset_df["station_id"].unique().tolist():
    temp_df = train_dataset_df[train_dataset_df["station_id"] == station_id]
    temp_df["bikes_available"] = temp_df["bikes_available"].astype("object").bfill().astype("float16")

    train_dataset_df_new = pd.concat([train_dataset_df_new, temp_df])

print(train_dataset_df_new.isnull().sum())

# %%
train_dataset_df_new2 = pd.DataFrame()
for station_id in train_dataset_df_new["station_id"].unique().tolist():
    temp_df2 = train_dataset_df_new[train_dataset_df_new["station_id"] == station_id]
    temp_df2["bikes_available_at0"] = temp_df2["bikes_available_at0"].astype("object").bfill().astype("float16")


    train_dataset_df_new2 = pd.concat([train_dataset_df_new2, temp_df2])

print(train_dataset_df_new2.isnull().sum())

train_dataset_df_new2.head()


# %%
# train_dpar_df = train_dataset_df_new2.drop("predict", axis=1)
# train_dpar_df.head()

train_dpar_df = train_dataset_df.drop("predict", axis=1)
train_dpar_df.head()

# %%
train_dpar_df.isna().sum()

# %%
# data = train_dpar_df[train_dpar_df["date"].dt.hour >= 1]

data = train_dpar_df.copy()
# %%
data = data.sort_values(by=["station_id", "date"])
# %%
# タイムスタンプに型作成
data["time_idx"] = (data["date"] - data["date"].min()).dt.total_seconds() // 3600

data["time_idx"] = data["time_idx"].astype("int")

# %%
data["station_id"] = data["station_id"].astype(str)
print(data.info())
# %%
data.info()
# %%
# data = data_pre00(data)

# %%
# %%
# create dataset and dataloaders
max_encoder_length = 168
max_prediction_length = 168
# 訓練データのカットオフ

training_cutoff = (
    data["time_idx"].max() - 168
)  # max_prediction_length 1週間をバリでにしたい168時間24*7

context_length = max_encoder_length
prediction_length = max_prediction_length
# %%

# データセットの設定
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",  
    target="bikes_available",  
    group_ids=["station_id"],
    categorical_encoders={"station_id": NaNLabelEncoder().fit(data.station_id)},
    time_varying_unknown_reals=["bikes_available"],
    static_categoricals=["station_id"],
    time_varying_known_reals=[
        "time_idx",
        "bikes_available_at0",
        "hour",
    ], 
    time_varying_known_categoricals=["dayofweek"], 
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
    # allow_missing_timesteps=True,
)


#%%
validation = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=training_cutoff + 1
)


# validation = TimeSeriesDataSet(
#     data[lambda x: x.time_idx >  training_cutoff -168],
#     time_idx="time_idx",  # 時間のインデックス
#     target="bikes_available",  # 予測する目的変数
#     group_ids=["station_id"],
#     categorical_encoders={"station_id": NaNLabelEncoder().fit(data.station_id)},
#     time_varying_unknown_reals=["bikes_available"],
#     static_categoricals=["station_id"],
#     time_varying_known_reals=[
#         "time_idx",
#         "bikes_available_at0",
#         "hour",
#     ], 
#     time_varying_known_categoricals=["dayofweek"], 
#     max_encoder_length=context_length,
#     max_prediction_length=prediction_length,
#     min_prediction_idx=training_cutoff + 1,
#     # allow_missing_timesteps=True,
# )







# %%
batch_size = 70

train_dataloader = training.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=0,
)
val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=batch_size,
    num_workers=0,
)

# %%
# calculate baseline absolute error
baseline_predictions = Baseline().predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)

RMSE()(baseline_predictions.output, baseline_predictions.y)

# %%
pl.seed_everything(123)

trainer = pl.Trainer(
    accelerator="cpu",
)  # gradient_clip_val=1e-1
net = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=NormalDistributionLoss(),
    optimizer="Adam",
)


# %%
# find optimal learning rate

res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
net.hparams.learning_rate = res.suggestion()

# %%
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
# トレーナーの設定
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    # gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)

# モデルのインスタンス化
net = DeepAR.from_dataset(
    training,
    learning_rate=3.548133892335755e-05,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=NormalDistributionLoss(),
)

# モデルのトレーニング
trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)
# %%
# best_model = net
predictions = best_model.predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)
RMSE()(predictions.output, predictions.y)
# %%
raw_predictions = net.predict(
    val_dataloader,
    mode="raw",
    return_x=True,
    n_samples=100,
    trainer_kwargs=dict(accelerator="cpu"),
)
# %%
station_id = validation.x_to_index(raw_predictions.x)["station_id"]
for idx in range(70):  # plot 10 examples
    best_model.plot_prediction(
        raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
    )
    plt.suptitle(f"station_id: {station_id.iloc[idx]}")



# %%


# %%


# %%
# 正義
# =============ここから=============
# %%
# data2 = {
#     "date": pd.date_range(start="2013-09-01", periods=38400, freq="h"),
#     "station_id": 1 * 38400,
#     "bikes_available": [10 + i % 5 for i in range(38400)],
# }


# data = pd.DataFrame(data2)
# # data["date"] = pd.to_datetime(data["date"])
# data["time_idx"] = range(len(data))
# data["bikes_available"] = data["bikes_available"].astype(float)
# data["station_id"] = data["station_id"].astype(str)
# # 訓練データのカットオフ
# # training_cutoff2 = data["date"].max() - pd.Timedelta(days=3)
# # %%
# data = data.astype(dict(station_id=str))
# data.info()
# # %%
# data.head()
# # %%
# # create dataset and dataloaders
# max_encoder_length = 72
# max_prediction_length = 24

# training_cutoff = data["time_idx"].max() - max_prediction_length

# context_length = max_encoder_length
# prediction_length = max_prediction_length

# training = TimeSeriesDataSet(
#     data[lambda x: x.time_idx <= training_cutoff],
#     time_idx="time_idx",
#     target="bikes_available",
#     categorical_encoders={"station_id": NaNLabelEncoder().fit(data.station_id)},
#     group_ids=["station_id"],
#     static_categoricals=[
#         "station_id"
#     ],  # as we plan to forecast correlations, it is important to use station_id characteristics (e.g. a station_id identifier)
#     time_varying_unknown_reals=["bikes_available"],
#     max_encoder_length=context_length,
#     max_prediction_length=prediction_length,
# )

# validation = TimeSeriesDataSet.from_dataset(
#     training, data, min_prediction_idx=training_cutoff + 1
# )
# batch_size = 64

# # synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
# train_dataloader = training.to_dataloader(
#     train=True,
#     batch_size=batch_size,
#     num_workers=0,  # batch_sampler="synchronized"
# )
# val_dataloader = validation.to_dataloader(
#     train=False,
#     batch_size=batch_size,
#     num_workers=0,  # batch_sampler="synchronized"
# )

# # %%
# # calculate baseline absolute error
# baseline_predictions = Baseline().predict(
#     val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
# )
# SMAPE()(baseline_predictions.output, baseline_predictions.y)

# # %%
# pl.seed_everything(123)


# trainer = pl.Trainer(
#     accelerator="cpu",
# )  # gradient_clip_val=1e-1
# net = DeepAR.from_dataset(
#     training,
#     learning_rate=3e-2,
#     hidden_size=30,
#     rnn_layers=2,
#     loss=NormalDistributionLoss(),
#     optimizer="Adam",
# )


# # %%
# # find optimal learning rate

# res = Tuner(trainer).lr_find(
#     net,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
#     min_lr=1e-5,
#     max_lr=1e0,
#     early_stop_threshold=100,
# )
# print(f"suggested learning rate: {res.suggestion()}")
# fig = res.plot(show=True, suggest=True)
# fig.show()
# net.hparams.learning_rate = res.suggestion()

# # %%
# early_stop_callback = EarlyStopping(
#     monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
# )
# trainer = pl.Trainer(
#     max_epochs=30,
#     accelerator="cpu",
#     enable_model_summary=True,
#     # gradient_clip_val=0.1,
#     callbacks=[early_stop_callback],
#     limit_train_batches=50,
#     enable_checkpointing=True,
# )


# net = DeepAR.from_dataset(
#     training,
#     learning_rate=0.03981071705534971,
#     log_interval=10,
#     log_val_interval=1,
#     hidden_size=30,
#     rnn_layers=2,
#     optimizer="Adam",
#     loss=NormalDistributionLoss(),
# )

# trainer.fit(
#     net,
#     train_dataloaders=train_dataloader,
#     val_dataloaders=val_dataloader,
# )
# # %%
# best_model_path = trainer.checkpoint_callback.best_model_path
# best_model = DeepAR.load_from_checkpoint(best_model_path)
# # %%
# # best_model = net
# predictions = best_model.predict(
#     val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
# )
# MAE()(predictions.output, predictions.y)
# # %%
# raw_predictions = net.predict(
#     val_dataloader,
#     mode="raw",
#     return_x=True,
#     n_samples=100,
#     trainer_kwargs=dict(accelerator="cpu"),
# )
# # %%
# station_id = validation.x_to_index(raw_predictions.x)["station_id"]
# for idx in range(1):  # plot 10 examples
#     best_model.plot_prediction(
#         raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
#     )
#     plt.suptitle(f"station_id: {station_id.iloc[idx]}")

# ===========ここまで======================


# %%
# バリデーション
length = len(train_dpar_df)
train_size = int(length * 0.8)
test_size = length - train_size
train_lstm, test_lstm = (
    train_dpar_df[0:train_size, :],
    train_dpar_df[train_size:length, :],
)
print(train_lstm.shape)
print(test_lstm.shape)


# %%


def create_dataset(dataset):
    dataX = []
    dataY = np.array([])
    # 1680で一つのデータセットであるためあまりの分は使わない
    extra_num = len(dataset) % 70
    max_len = len(dataset) - extra_num
    for i in range(1680, max_len, 70):
        xset = []
        for j in range(dataset.shape[1]):
            a = dataset[i - 1680 : i, j]
            xset.append(a)

        temp_array = np.array(dataset[i : i + 70, 0])
        dataY = np.concatenate([dataY, temp_array])
        dataX.append(xset)

    dataY = dataY.reshape(-1, 70)
    return np.array(dataX), dataY


# %%
trainX, trainY = create_dataset(train_lstm)
testX, testY = create_dataset(test_lstm)
# LSTMのモデルに入力用にデータの形を成型
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], trainX.shape[2]))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], testX.shape[2]))

# 入力データと正解データの形を確認
print(trainX.shape)
print(trainY.shape)

# %%

# LSTMの学習
model = Sequential()
model.add(LSTM(50, input_shape=(trainX.shape[1], 1680)))
model.add(Dense(70))
model.compile(loss="mean_squared_error", optimizer="adam")
hist = model.fit(trainX, trainY, epochs=20, batch_size=1, verbose=2)


# %%
# 予測精度の確認
# 学習済みモデルで予測
train_predict = model.predict(trainX)
test_predict = model.predict(testX)

# スケールを元に戻す
train_predict = scaler_for_inverse.inverse_transform(train_predict)
trainY = scaler_for_inverse.inverse_transform(trainY)
test_predict = scaler_for_inverse.inverse_transform(test_predict)
testY = scaler_for_inverse.inverse_transform(testY)

# 各ステーションのスコアの平均値を算出
train_score_list = []
test_score_list = []
for i in range(70):
    trainscore = math.sqrt(mean_squared_error(trainY[:, i], train_predict[:, i]))
    train_score_list.append(trainscore)
    testscore = math.sqrt(mean_squared_error(testY[:, i], test_predict[:, i]))
    test_score_list.append(testscore)

print("trainのRMSE平均:", mean(train_score_list))
print("testのRMSE平均:", mean(test_score_list))


# %%
# # 予測日とその前日を含むデータフレームを作成すると前日の日付データを返す関数
def make_sameday_thedaybefore_dataset(dataset, prediction_date):
    # 前日の日付をtimedeltaで取得
    before_date = prediction_date - timedelta(days=1)
    prediction_date = str(prediction_date).split(" ")[0]
    before_date = str(before_date).split(" ")[0]

    # 予測日とその前日を含むものだけを抽出
    temp_dataset = dataset[dataset["date"].isin([before_date, prediction_date])]

    return before_date, temp_dataset


# 評価用のデータセットを作成する関数
def make_evaluation_dataset(dataset):
    output_df = pd.DataFrame()
    prediction_date_list = dataset[dataset["predict"] == 1]["date"].tolist()
    for date in sorted(list(set(prediction_date_list))):
        before_date, temp_dataset = make_sameday_thedaybefore_dataset(dataset, date)
        # 前日のbikes_availableに欠損値が含まれるかどうかの判定
        if (
            temp_dataset[temp_dataset["date"] == before_date]["bikes_available"][1:]
            .isna()
            .any()
        ):
            # 各ステーションで予測日の0時で前日の1時以降のデータを置換
            # 予測日のbikes_availableの置換は、後程別途処理するので今回は無視
            temp_dataset = temp_dataset.sort_values(["station_id", "date", "hour"])
            temp_dataset["bikes_available"] = (
                temp_dataset["bikes_available"]
                .astype("object")
                .bfill()
                .astype("float16")
            )
            temp_dataset = temp_dataset.sort_values(
                ["date", "hour", "station_id"], ascending=True
            )
            # 予測には、前日の1時からのデータしか使用しないので、0時のデータは除く
            output_df = pd.concat([output_df, temp_dataset.iloc[70:, :]])
        else:
            output_df = pd.concat([output_df, temp_dataset.iloc[70:, :]])
    return output_df


# %%
# 評価用のデータセット
evaluation_df = make_evaluation_dataset(evaluation_dataset_df)
evaluation_df.head()
# %%
evaluation_df.info()


# %%
# LSTMの出力結果でデータを補完しながら、提出用データフレームを作成する関数
def predict_eva_dataset(eva_dataset):
    submission_df = pd.DataFrame()
    # 予測したbikes_availableを元のスケールに戻すための変数
    scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
    scale_y = scaler_for_inverse.fit_transform(eva_dataset[["bikes_available"]])
    prediction_date_list = eva_dataset[eva_dataset["predict"] == 1]["date"].tolist()
    for date in sorted(list(set(prediction_date_list))):
        _, temp_eva_dataset = make_sameday_thedaybefore_dataset(eva_dataset, date)
        for i in range(0, 1610, 70):
            # モデルに入れるためのデータセット(1680×columns)
            temp_eva_dataset_train = temp_eva_dataset.iloc[i : 1680 + i, :]
            # predictは特徴量に使わないため、ここで削除
            temp_eva_dataset_train = temp_eva_dataset_train.drop("predict", axis=1)
            # データを標準化する
            scaler = MinMaxScaler(feature_range=(0, 1))
            temp_eva_dataset_scale = scaler.fit_transform(
                temp_eva_dataset_train.iloc[:, 3:]
            )

            # モデルに入力する形にデータを整形
            train = []
            xset = []
            for j in range(temp_eva_dataset_scale.shape[1]):
                a = temp_eva_dataset_scale[:, j]
                xset.append(a)
            train.append(xset)
            train = np.array(train)
            train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2]))

            # 学習済みlstmモデルで予測
            predict_scale = model.predict(train)
            predict = scaler_for_inverse.inverse_transform(predict_scale)

            # 次に使うbikes_availableに出力結果を補完
            temp_eva_dataset.iloc[1680 + i : 1750 + i, 3] = predict[0]

        submission_df = pd.concat([submission_df, temp_eva_dataset.iloc[1610:, :]])

    return submission_df


# %%
evaluation_df
# %%
# 予測した結果を時系列で可視化して確認
submission_df = predict_eva_dataset(evaluation_df)
sns.lineplot(x="date", y="bikes_available", data=submission_df)

# %%
submission_df
# %%

# %%
lstm_submit_df = submission_df[submission_df["predict"] == 1].sort_values(
    ["station_id", "date"]
)[["bikes_available"]]
lstm_submit_df["bikes_available"] = lstm_submit_df["bikes_available"].map(
    lambda x: 0 if x < 0 else x
)
lstm_submit_df.index = status_df[status_df["predict"] == 1].index
# lstm_submit_df.to_csv("lstm_submission.csv",header=None)#
lstm_submit_df.head()
# %%
