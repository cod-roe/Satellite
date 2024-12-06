# %% [markdown]
## 特徴量生成！
# =================================================
# 特徴量選択   valid：直前1ヶ月を検証*12モデル


# %%
# ライブラリ読み込み
# =================================================
import datetime as dt
from datetime import timedelta

# import gc
# import json
import logging
import math

# import re
import os
from statistics import mean
import sys
import pickle
from IPython.display import display
import warnings
# import zipfile

import numpy as np
import pandas as pd

# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

# sckit-learn
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
import category_encoders as ce

# from sklearn.model_selection import StratifiedKFold, train_test_split , KFold
from sklearn.metrics import (
    mean_squared_error,
)  # ,accuracy_score, roc_auc_score ,confusion_matrix

# lightGBM
import lightgbm as lgb

# lightGBM精度測定
import shap

# パラメータチューニング
# import optuna

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam

# 次元圧縮
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# クラスタで選別
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 20  # スプレッドシートAの番号

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
# Dataset #
######################
target_columns = "bikes_available"
sub_index = "id"

######################
# ハイパーパラメータの設定
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


# 曜日を日本語に直す関数
def get_weekday_jp(dt):
    w_list = [
        "月曜日",
        "火曜日",
        "水曜日",
        "木曜日",
        "金曜日",
        "土曜日",
        "日曜日",
    ]
    return w_list[dt.weekday()]


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


# %% ファイルの読み込み
# Load Data
# =================================================
# status
status_df = load_data(2)
# %%
display(status_df.shape)
status_df.info()


# %%
# 日付と曜日カラムの作成
status_df["date"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    # + " "
    # + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date"] = pd.to_datetime(status_df["date"])  # .dt.strftime("%Y-%m-%d")


status_df["weekday"] = status_df["date"].apply(get_weekday_jp)


status_df.head(10)
# %%
status_df["date2"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    + " "
    + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date2"] = pd.to_datetime(status_df["date2"]).dt.strftime("%Y-%m-%d %H")
status_df.head()


# %%
# station読み込み
# =================================================
station_df = load_data(1)

# 日付に変更
station_df["installation_date"] = pd.to_datetime(station_df["installation_date"])
station_df["installation_date"].info()
station_df["installation_date"].head()


# %%
# weather読み込み
# =================================================
weather_df = load_data(4)
# %%
# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df.info()


# %%
# trip読み込み
# =================================================
trip_df = load_data(3)

# %%
# time型変換
trip_df["start_date"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d %H")
trip_df["end_date"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d %H")


# %%
# 貸出集計
trip_tbl_s = trip_df.groupby(["start_date", "start_station_id"])["trip_id"].count()

trip_tbl_s = trip_tbl_s.reset_index()
trip_tbl_s.columns = ["date2", "station_id", "count_start"]


# %%
# 返却集計
trip_tbl_e = trip_df.groupby(["end_date", "end_station_id"])["trip_id"].count()

trip_tbl_e = trip_tbl_e.reset_index()
trip_tbl_e.columns = ["date2", "station_id", "count_end"]


# %%
# statusにstationのstation_idをマージ
status_df = pd.merge(status_df, station_df, on="station_id", how="left")
display(status_df.head())

# weatherのprecipitationをマージ
status_df = pd.merge(status_df, weather_df, on="date", how="left")
# %%
# trip集計をマージ
status_df = pd.merge(status_df, trip_tbl_s, on=["date2", "station_id"], how="left")

status_df = pd.merge(status_df, trip_tbl_e, on=["date2", "station_id"], how="left")


# %%
# count_start,endをdock_countで割る
status_df["count_start/dc"] = status_df["count_start"] / status_df["dock_count"]

status_df["count_end/dc"] = status_df["count_end"] / status_df["dock_count"]

status_df.head()
# %%
# 0時の特徴量
t = status_df.groupby(["station_id", "date"]).first()["bikes_available"].reset_index()

t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
t.columns = ["staion_id", "date", "bikes_available_at0"]
t.head(25)

status_df["bikes_available_at0"] = t["bikes_available_at0"].astype("float16")

# %%
# city列をcategorical encoderを用いて数値化
cols = ["city"]

encoder = ce.OneHotEncoder(cols=cols, handle_unknown="impute")
temp_ = encoder.fit_transform(status_df[cols]).add_prefix("CE_")

status_df = pd.concat([status_df, temp_], axis=1)
# %%
# 9~16時の識別
status_df["daytime"] = status_df["hour"].apply(lambda x: 1 if x >= 9 and x <= 16 else 0)

status_df.head()

# %%
# データを分けるカラム作成
status_df["yearmonth"] = status_df["date"].astype(str).apply(lambda x: x[:7])
status_df["yearmonth"].head()

# %%
# 祝日 アメリカ
holiday_list = [
    "2013-01-01",
    "2013-01-21",
    "2013-02-18",
    "2013-05-27",
    "2013-06-19",
    "2013-07-04",
    "2013-09-02",
    "2013-10-14",
    "2013-11-11",
    "2013-11-28",
    "2013-12-25",
    "2014-01-01",
    "2014-01-20",
    "2014-02-17",
    "2014-05-26",
    "2014-06-19",
    "2014-07-04",
    "2014-09-01",
    "2014-10-13",
    "2014-11-11",
    "2014-11-27",
    "2014-12-25",
    "2015-01-01",
    "2015-01-19",
    "2015-02-16",
    "2015-05-25",
    "2015-06-19",
    "2015-07-03",
    "2015-09-07",
    "2015-10-12",
    "2015-11-11",
    "2015-11-26",
    "2015-12-25",
]


status_df["holiday"] = np.zeros(len(status_df))

for i in holiday_list:
    status_df["holiday"][status_df["date"] == i] = 1

# 曜日をワンホットエンコーダー
# df_ohe = pd.get_dummies(
#     status_df[["weekday", "events"]], drop_first=False, dtype="uint8"
# )  # dtype指定しないとTrue,Falseになる
# status_df = pd.concat([status_df, df_ohe], axis=1)


# %%
# カテゴリ変数
status_df = data_pre00(status_df)

# %%
# ラグ
# 前日の同じ時刻の台数
status_df["available_shift24"] = status_df.groupby("station_id")[
    "bikes_available"
].shift(24)

# 1週間前（同じ曜日）の同じ時刻の台数
status_df["available_shift168"] = status_df.groupby("station_id")[
    "bikes_available"
].shift(168)


# %%
# status_df.query("station_id == 0 and weekday.str.contains('土曜日') ")[
#     ["date2", "hour", "bikes_available"]
# ]


# %% 移動平均
# # 1month
status_df["weekday_hour_1m_mean"] = (
    status_df.groupby("station_id")["bikes_available"].shift(168)
    + status_df.groupby("station_id")["bikes_available"].shift(336)
    + status_df.groupby("station_id")["bikes_available"].shift(504)
    + status_df.groupby("station_id")["bikes_available"].shift(672)
) / 4.0

status_df["weekday_hour_1m_std"] = (
    status_df.groupby(["station_id", "weekday", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).std())
    .interpolate(method="bfill")
)


status_df["weekday_hour_1m_max"] = (
    status_df.groupby(["station_id", "weekday", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=4, min_periods=1).max())
    .interpolate(method="bfill")
)

# 3month
status_df["weekday_hour_3m_mean"] = (
    status_df.groupby(["station_id", "weekday", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=13, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 6month
status_df["weekday_hour_6m_mean"] = (
    status_df.groupby(["station_id", "weekday", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=25, min_periods=1).mean())
    .interpolate(method="bfill")
)


status_df["weekday_hour_6m_min"] = (
    status_df.groupby(["station_id", "weekday", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=25, min_periods=1).min())
    .interpolate(method="bfill")
)


status_df["week_hour_1/3"] = (
    status_df["weekday_hour_1m_mean"] / status_df["weekday_hour_3m_mean"]
)

status_df["week_hour_1/6"] = (
    status_df["weekday_hour_1m_mean"] / status_df["weekday_hour_6m_mean"]
)


# 7
status_df["hour_1w_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    .interpolate(method="bfill")
)

status_df["hour_1w_std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).std())
    .interpolate(method="bfill")
)

status_df["hour_1w_min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).min())
    .interpolate(method="bfill")
)


# 30
status_df["hour_1m_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    .interpolate(method="bfill")
)


status_df["hour_1m_min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).min())
    .interpolate(method="bfill")
)


# 3month
status_df["hour_3m_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=90, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 3d
status_df["3_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=72, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 7d
status_df["7_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=168, min_periods=1).mean())
    .interpolate(method="bfill")
)


status_df["30_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=720, min_periods=1).mean())
    .interpolate(method="bfill")
)

# 90d
status_df["90_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=2160, min_periods=1).mean())
    .interpolate(method="bfill")
)
# 180d
status_df["180_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.shift(1).rolling(window=43420, min_periods=1).mean())
    .interpolate(method="bfill")
)

# 7/180
status_df["7/180"] = status_df["7_mean"] / status_df["180_mean"]

# 7/90
status_df["7/90"] = status_df["7_mean"] / status_df["90_mean"]

# 7/30
status_df["7/30"] = status_df["7_mean"] / status_df["30_mean"]


# 3/90
status_df["3/90"] = status_df["3_mean"] / status_df["90_mean"]

# 3/30
status_df["3/30"] = status_df["3_mean"] / status_df["30_mean"]

# 3/7
status_df["3/7"] = status_df["3_mean"] / status_df["7_mean"]

# %%
status_df.tail(5)


# %%
# 相関係数の高いペアを片方削除
def drop_features(train, target):
    # 相関係数の計算
    train_ = train.drop([target], axis=1)
    corr_matrix_ = train_.corr().abs()
    corr_matrix = train.corr().abs()

    # 相関係数が0.95以上の変数ペアを抽出
    high_corr_vars = np.where(np.triu(corr_matrix_, k=1) > 0.95)
    high_corr_pairs = [
        (train_.columns[x], train_.columns[y]) for x, y in zip(*high_corr_vars)
    ]
    display(high_corr_pairs)

    # 目的変数との相関係数が小さい方の変数を削除
    for pair in high_corr_pairs:
        var1_corr = corr_matrix.loc[target, pair[0]]
        var2_corr = corr_matrix.loc[target, pair[1]]

        try:  # 既に消した変数が入ってきたとき用
            if var1_corr < var2_corr:
                train = train.drop(pair[0], axis=1)
            else:
                train = train.drop(pair[1], axis=1)
        except Exception as e:
            print(f"How exceptional! {e}")
            pass
    return train


# %%
def remove_collinear_features(train, target, threshold=1.0, s=0):
    X = train.drop(target, axis=1)
    y = train[target]
    cols = X.columns
    # 特徴量間の非類似性距離行列を計算
    std = StandardScaler().fit_transform(X)
    X_ = pd.DataFrame(std, columns=X.columns)  # 標準化
    distances = np.zeros((X_.shape[1], X_.shape[1]))
    for i in range(X_.shape[1]):
        for j in range(i + 1, X_.shape[1]):
            corr, _ = spearmanr(X_.iloc[:, i], X_.iloc[:, j])
            distances[i, j] = distances[j, i] = 1 - abs(corr)
    np.fill_diagonal(distances, 0)  # 対角成分をゼロに設定
    distances = squareform(distances)

    # Ward の最小分散基準で階層的クラスター分析
    clusters = linkage(distances, method="ward")
    cluster_labels = fcluster(clusters, threshold, criterion="distance")
    # クラスター内で1つの特徴量のみ残す
    unique_cluster_labels = np.unique(cluster_labels)
    unique_features = []
    for label in unique_cluster_labels:
        features = X.columns[cluster_labels == label]
        print(f"同じクラスタの特徴量は{features}です。")
        if len(features) > 1:
            print(f"選ばれたのは{features[s]}でした。")
            unique_features.append(features[s])
        else:
            print(f"選ばれたのは{features}でした。")
            unique_features.extend(features)

    df = X[unique_features]
    df[target] = y

    return df, clusters, cols


# %%
# train_del = status_df.drop(["date","weekday", "date2","installation_date", "city", "events", "yearmonth","installation_date"], axis=1)

# # %%
# # 相関高いもの消す
# train_2 = drop_features(train_del, "bikes_available")
# print(train_2.shape)


# #%%
# # %%
# # クラス分け一つ目
# # remove_collinear_features1 前から1つ
# train_3_1, clusters, columns = remove_collinear_features(
#     train_2, "bikes_available", threshold=1.0, s=0
# )
# print(train_3_1.shape)
# print(train_3_1.columns)


# %%
# データセット作成
# 2014/09/01以前をtrain、以後をtestに分割(testはpredict = 1のものに絞る)
# =================================================


train_lgbm = status_df  # [status_df["date"] < "2014-09-01"]

valid_lgbm = status_df[
    (status_df["date"] >= "2014-09-01") & (status_df["predict"] == 0)
]


test_lgbm = status_df[(status_df["date"] >= "2014-09-01") & (status_df["predict"] == 1)]


# %%
# train_lgbmはbikes_availableがnanではないものに絞る
train_lgbm = train_lgbm[train_lgbm["bikes_available"].notna()]
valid_lgbm = valid_lgbm[valid_lgbm["bikes_available"].notna()]

# %%
train_lgbm.columns
# %%


# %%
# 使用カラム

train_lgbm_df = train_lgbm[
    [
        # "year",
        "month",
        "day",
        "hour",
        "station_id",
        "weekday",
        "dock_count",
        # "lat",
        # "long",
        "events",
        # "precipitation",
        "count_start/dc",
        "count_end/dc",
        # "CE_city_1",
        # "CE_city_2",
        # "CE_city_3",
        # "CE_city_4",
        # "CE_city_5",
        "daytime",
        "holiday",
        "available_shift24",
        "available_shift168",
        "weekday_hour_1m_mean",
        "weekday_hour_1m_std",
        "weekday_hour_1m_max",
        "weekday_hour_3m_mean",
        "weekday_hour_6m_mean",
        "weekday_hour_6m_min",
        # "weekday_hour_6m_max",
        "week_hour_1/3",
        "week_hour_1/6",
        "hour_1w_mean",
        "hour_1w_std",
        # "hour_1w_min",
        # "hour_1w_max",
        "hour_1m_mean",
        # "hour_1m_std",
        "hour_1m_min",
        # "hour_1m_max",
        "hour_3m_mean",
        # "hour_3m_std",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        # "180_mean",
        "7/180",
        "7/90",
        "7/30",
        # "3/180",
        "3/90",
        "3/30",
        "3/7",
        # "bikes_available_at0",
        "bikes_available",
    ]
]

valid_lgbm_df = valid_lgbm[
    [
        # "year",
        "month",
        "day",
        "hour",
        "station_id",
        "weekday",
        "dock_count",
        # "lat",
        # "long",
        "events",
        # "precipitation",
        "count_start/dc",
        "count_end/dc",
        # "CE_city_1",
        # "CE_city_2",
        # "CE_city_3",
        # "CE_city_4",
        # "CE_city_5",
        "daytime",
        "holiday",
        "available_shift24",
        "available_shift168",
        "weekday_hour_1m_mean",
        "weekday_hour_1m_std",
        "weekday_hour_1m_max",
        "weekday_hour_3m_mean",
        "weekday_hour_6m_mean",
        "weekday_hour_6m_min",
        # "weekday_hour_6m_max",
        "week_hour_1/3",
        "week_hour_1/6",
        "hour_1w_mean",
        "hour_1w_std",
        # "hour_1w_min",
        # "hour_1w_max",
        "hour_1m_mean",
        # "hour_1m_std",
        "hour_1m_min",
        # "hour_1m_max",
        "hour_3m_mean",
        # "hour_3m_std",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        # "180_mean",
        "7/180",
        "7/90",
        "7/30",
        # "3/180",
        "3/90",
        "3/30",
        "3/7",
        # "bikes_available_at0",
        "bikes_available",
    ]
]


id_train_lgbm = train_lgbm[["yearmonth"]]


test_lgbm_df = test_lgbm[
    [
        # "year",
        "month",
        "day",
        "hour",
        "station_id",
        "weekday",
        "dock_count",
        # "lat",
        # "long",
        "events",
        # "precipitation",
        "count_start/dc",
        "count_end/dc",
        # "CE_city_1",
        # "CE_city_2",
        # "CE_city_3",
        # "CE_city_4",
        # "CE_city_5",
        "daytime",
        "holiday",
        "available_shift24",
        "available_shift168",
        "weekday_hour_1m_mean",
        "weekday_hour_1m_std",
        "weekday_hour_1m_max",
        "weekday_hour_3m_mean",
        "weekday_hour_6m_mean",
        "weekday_hour_6m_min",
        # "weekday_hour_6m_max",
        "week_hour_1/3",
        "week_hour_1/6",
        "hour_1w_mean",
        "hour_1w_std",
        # "hour_1w_min",
        # "hour_1w_max",
        "hour_1m_mean",
        # "hour_1m_std",
        "hour_1m_min",
        # "hour_1m_max",
        "hour_3m_mean",
        # "hour_3m_std",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        # "180_mean",
        "7/180",
        "7/90",
        "7/30",
        # "3/180",
        "3/90",
        "3/30",
        "3/7",
        # "bikes_available_at0",
    ]
]
print(train_lgbm_df.shape, valid_lgbm_df.shape, test_lgbm_df.shape)

# %%
# データセット
x_train_lgbm = train_lgbm_df.drop("bikes_available", axis=1)
y_train_lgbm = train_lgbm["bikes_available"]

# %%
# バリデーション

list_cv_month = [
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
        ],
        ["2014-08"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
        ],
        ["2014-09"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
        ],
        ["2014-10"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
        ],
        ["2014-11"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
        ],
        ["2014-12"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
        ],
        ["2015-01"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
        ],
        ["2015-02"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
            "2015-02",
        ],
        ["2015-03"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
            "2015-02",
            "2015-03",
        ],
        ["2015-04"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
            "2015-02",
            "2015-03",
            "2015-04",
        ],
        ["2015-05"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
            "2015-02",
            "2015-03",
            "2015-04",
            "2015-05",
        ],
        ["2015-06"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
            "2014-01",
            "2014-02",
            "2014-03",
            "2014-04",
            "2014-05",
            "2014-06",
            "2014-07",
            "2014-08",
            "2014-09",
            "2014-10",
            "2014-11",
            "2014-12",
            "2015-01",
            "2015-02",
            "2015-03",
            "2015-04",
            "2015-05",
            "2015-06",
        ],
        ["2015-07"],
    ],
]


# %%
# 学習関数の定義
# =================================================
def train_lgb(
    input_x,
    input_y,
    input_id,
    params,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    metrics = []
    imp = pd.DataFrame()
    # train_oof = np.zeros(len(input_x))
    # shap_v = pd.DataFrame()

    # cross-validation
    # validation
    cv = []
    for month_tr, month_va in list_cv_month:
        cv.append(
            [
                input_id.index[input_id["yearmonth"].isin(month_tr)],
                input_id.index[input_id["yearmonth"].isin(month_va)],
            ]
        )

    # 1.学習データと検証データに分離
    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)
        print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

        x_tr, y_tr = (
            input_x.loc[idx_tr, :],
            input_y[idx_tr],
        )
        x_va, y_va = (
            input_x.loc[idx_va, :],
            input_y[idx_va],
        )

        print(x_tr.shape, x_va.shape)

        # モデルの保存先名
        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")

        if not os.path.isfile(fname_lgb):  # if trained model, no training
            # train
            print("-------training start-------")
            model = lgb.LGBMRegressor(**params)
            model.fit(
                x_tr,
                y_tr,
                eval_set=[(x_tr, y_tr), (x_va, y_va)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=100, verbose=True),
                    lgb.log_evaluation(100),
                ],
            )

            # モデルの保存
            # with open(fname_lgb, "wb") as f:
            #     pickle.dump(model, f, protocol=4)

        else:
            print("すでに学習済みのためモデルを読み込みます")
            with open(fname_lgb, "rb") as f:
                model = pickle.load(f)

        # evaluate
        y_tr_pred = model.predict(x_tr)
        y_va_pred = model.predict(x_va)

        metric_tr = mean_squared_error(y_tr, y_tr_pred, squared=False)
        metric_va = mean_squared_error(y_va, y_va_pred, squared=False)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[rmse] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # oof
        # train_oof[idx_va] = y_va_pred

        # shap_v  & 各特徴量のSHAP値の平均絶対値で重要度を算出
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(input_x)

        # _shap_importance = np.abs(shap_values).mean(axis=0)
        # _shap = pd.DataFrame(
        #     {"col": input_x.columns, "shap": _shap_importance, "nfold": nfold}
        # )
        # shap_v = pd.concat([shap_v, _shap])

        # imp
        _imp = pd.DataFrame(
            {"col": input_x.columns, "imp": model.feature_importances_, "nfold": nfold}
        )
        imp = pd.concat([imp, _imp])

    print("-" * 20, "result", "-" * 20)

    # metric
    metrics = np.array(metrics)
    print(metrics)
    print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
        va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

    # print(f"[oof]{mean_squared_error(input_y, train_oof):.4f}")

    # oof
    # train_oof = pd.concat(
    #     [
    #         input_id,
    #         pd.DataFrame({"pred": train_oof}),
    #     ],
    #     axis=1,
    # )

    # importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    # shap値
    # shap_v = shap_v.groupby("col")["shap"].agg(["mean", "std"]).reset_index(drop=False)
    # shap_v.columns = ["col", "shap", "shap_std"]

    # stdout と stderr を一時的にリダイレクト
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    sys_stdout_backup = sys.stdout
    sys_stderr_backup = sys.stderr

    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
    print("-" * 20, "result", "-" * 20)
    print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))
    print(name)
    print(x_tr.shape, x_va.shape)
    print(metrics)
    print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
        va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

    print("-" * 20, "importance", "-" * 20)
    print(imp.sort_values("imp", ascending=False)[:10])

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return imp, metrics # , shap_v


# %%
# train
imp, metrics , shap_v = train_lgb(
    x_train_lgbm,
    y_train_lgbm,
    id_train_lgbm,
    params,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
# imp
imp_sort = imp.sort_values("imp", ascending=False)
display(imp_sort)
# 重要度をcsvに出力
# imp_sort.to_csv(os.path.join(OUTPUT_EXP, f"imp_{name}.csv"), index=False, header=False)
# %%
shap_sort = shap_v.sort_values("shap", ascending=False)
display(shap_sort)

shap_value = shap_sort.copy()
imp_value = imp_sort.copy()
# %%
# %%
# 一個ずつ加えて精度確認
select_list = []
scores = []
for i in shap_value["col"]:  # [:20]:
    select_list.append(i)
    print(select_list)
    x_trains = x_train_lgbm[select_list]
    print(x_trains.shape)
    imp, metrics= train_lgb(
        x_trains,
        y_train_lgbm,
        id_train_lgbm,
        params,
        list_nfold=[9,10,11],
    )
    scores.append([len(select_list), metrics[:, 2].mean()])


# %%
# プロット作成
scores = pd.DataFrame(scores)
scores.head()
"""
精度が改善されなくなる場所がどこが確認する
"""
sns.lineplot(data=scores, x=0, y=1)


# %%
scores.head(50)
# %%
display(shap_sort[:26].reset_index())
print(shap_sort[:26].shape)
list(shap_sort["col"][:26].values)

#%%
#削除
list(shap_sort["col"][26:].values)

# %%
list_valid_month = [
    "2014-09",
    "2014-10",
    "2014-11",
    "2014-12",
    "2015-01",
    "2015-02",
    "2015-03",
    "2015-04",
    "2015-05",
    "2015-06",
    "2015-07",
    "2015-08",
]
# %%
# 推論関数の定義 検証用 =================================================


def predict_lgb(
    input_x,
    input_y,
    input_id,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    # pred = np.zeros((len(input_x), len(list_nfold)))
    valid_oof = pd.DataFrame(index=input_id.index)
    valid_oof["pred"] = np.zeros(len(input_id))

    cv = []
    for month_va in list_valid_month:
        cv.append(
            input_id.index[input_id["yearmonth"].isin([month_va])],
        )

    for nfold in list_nfold:
        print("-" * 20, nfold, f"{list_valid_month[nfold]}", "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        idx_va = cv[nfold]
        x_va, y_va = input_x.loc[idx_va, :], input_y[idx_va]

        y_va_pred = model.predict(x_va)
        metric_va = mean_squared_error(y_va, y_va_pred, squared=False)
        print(f"{list_valid_month[nfold]} [rmse] va:{metric_va:.4f}")

        # oof
        valid_oof.loc[idx_va, "pred"] = y_va_pred

    # 平均値算出
    # pred = pd.concat(
    #     [
    #         input_id["id"],
    #         pd.DataFrame(pred.mean(axis=1)),
    #     ],
    #     axis=1,
    # )

    print("Done.")
    metric_valid_ = mean_squared_error(input_y, valid_oof.loc[:, "pred"], squared=False)
    print(f"[rmse] valid:{metric_valid_:.4f}")
    return valid_oof


# %%
# 検証処理
# =================================================
valid_lgbm_x = valid_lgbm_df.drop("bikes_available", axis=1)
valid_lgbm_y = valid_lgbm_df["bikes_available"]

id_valid_lgbm = valid_lgbm[["yearmonth"]]


# %%
valid_lgbm_oof = predict_lgb(
    valid_lgbm_x,
    valid_lgbm_y,
    id_valid_lgbm,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
valid_lgbm_oof.head()

# %%


# %%
# ここまで
# =================================================

# %%
# 推論処理
# =================================================

id_test_lgbm = test_lgbm[["yearmonth"]]

# %%
# 提出用推論関数の定義 =================================================


def predict_sub_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
):
    # pred = np.zeros((len(input_x), len(list_nfold)))
    valid_oof = pd.DataFrame(index=input_id.index)
    valid_oof["pred"] = np.zeros(len(input_id))

    cv = []
    for month_va in list_valid_month:
        cv.append(
            input_id.index[input_id["yearmonth"].isin([month_va])],
        )

    for nfold in list_nfold:
        print("-" * 20, nfold, f"{list_valid_month[nfold]}", "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        idx_va = cv[nfold]
        x_va = input_x.loc[idx_va, :]

        y_va_pred = model.predict(x_va)

        # oof
        valid_oof.loc[idx_va, "pred"] = y_va_pred

    # 平均値算出
    # pred = pd.concat(
    #     [
    #         input_id["id"],
    #         pd.DataFrame(pred.mean(axis=1)),
    #     ],
    #     axis=1,
    # )

    print("Done.")

    return valid_oof


# %%
test_lgbm_pred = predict_sub_lgb(
    test_lgbm_df,
    id_test_lgbm,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
test_lgbm_pred.head()


# %%
# submitファイルの出力
# =================================================
# test_lgbm_pred.to_csv(
#     os.path.join(OUTPUT_EXP, f"submission_{name}.csv"),
#     header=False,  # ,index=False
# )


#
# %% [markdown]
## LSTM 分析start!
# ==========================================================


# %%
# データセット作成
main_df = status_df[
    ["date", "hour", "station_id", "bikes_available", "weekday", "predict"]
]
main_df.head()

# %%
main_df = pd.get_dummies(main_df, dtype="uint8")
print(main_df.columns)
print(main_df.shape)

# %%
main_df.head()
# %%
# 学習用のデータフレーム作成
train_dataset_df = main_df[main_df["date"] < "2014-09-01"]
# 評価用のデータフレーム作成（使用するモデルの関係上、前日のデータが必要なため2014-08-31から取得）
evaluation_dataset_df = main_df[main_df["date"] >= "2014-08-31"]

# %%

print(train_dataset_df.isna().sum())
# %%
train_dataset_df[train_dataset_df["bikes_available"].isna()].describe().T
# %%
train_dataset_df[train_dataset_df["bikes_available"].isna()].head()
# %%

train_dataset_df[train_dataset_df["bikes_available"].isna()].tail()
# %%

# 各ステーション毎に、欠損値を後の値で埋める
train_dataset_df_new = pd.DataFrame()
for station_id in train_dataset_df["station_id"].unique().tolist():
    temp_df = train_dataset_df[train_dataset_df["station_id"] == station_id]
    temp_df["bikes_available"] = (
        temp_df["bikes_available"].astype("object").bfill().astype("float16")
    )

    train_dataset_df_new = pd.concat([train_dataset_df_new, temp_df])

print(train_dataset_df_new.isnull().sum())

# %%
train_lstm_df = train_dataset_df_new.sort_values(
    ["date", "hour", "station_id"], ascending=True
).reset_index(drop=True)
evaluation_dataset_df = evaluation_dataset_df.sort_values(
    ["date", "hour", "station_id"], ascending=True
).reset_index(drop=True)

train_lstm_df.head()
# %%
train_lstm_df = train_lstm_df.drop("predict", axis=1)
train_lstm_df.head()
# %%
# データの標準化
# 特徴量を標準化するための変数
scaler = MinMaxScaler(feature_range=(0, 1))

# 標準化された出力をもとにスケールに変換(inverse)するために必要な変数
scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
train_lstm_df_scale = scaler.fit_transform(train_lstm_df.iloc[:, 3:])


bikes_available_scale = scaler_for_inverse.fit_transform(
    train_lstm_df[["bikes_available"]]
)
print(train_lstm_df_scale.shape)


# %%
# バリデーション
length = len(train_lstm_df_scale)
train_size = int(length * 0.8)
test_size = length - train_size
train_lstm, test_lstm = (
    train_lstm_df_scale[0:train_size, :],
    train_lstm_df_scale[train_size:length, :],
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
    scale_y = scaler_for_inverse.fit_transform(eva_dataset[["bikes_available"]])  # noqa: F841
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


# %% [markdown]
## submitファイルの作成!
# ==========================================================
display(lstm_submit_df.head(10))
display(test_lgbm_pred.head(10))

# %%
# 単純平均
test_pred2 = pd.DataFrame()
test_pred2["pred"] = (lstm_submit_df["bikes_available"] + test_lgbm_pred["pred"]) / 2
test_pred2.index = status_df[status_df["predict"] == 1].index
# %%
test_pred2.head()

# %%
# submitファイルの出力
# =================================================
# test_pred2.to_csv(
#     os.path.join(OUTPUT_EXP, f"submission_{name}.csv"),
#     header=False,  # ,index=False
# )
