# %% [markdown]
## 特徴量生成！
# =================================================
# アンサンブル valid003 LSTMで予測してから平均だす


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
import optuna

# tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
)

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
serial_number = 32  # スプレッドシートAの番号

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
    "learning_rate": 0.02,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}


params2 = {
    "num_leaves": 33,
    "min_child_samples": 167,
    "min_sum_hessian_in_leaf": 0.0013701073829839475,
    "feature_fraction": 0.9007334262963621,
    "bagging_fraction": 0.8100637274542337,
    "lambda_l1": 0.6012343153268627,
    "lambda_l2": 92.98811650570019,
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "learning_rate": 0.05,
    "n_estimators": 100000,
    "bagging_freq": 1,
    "random_state": 123,
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


# %%
# 相関係数の高いペアを片方削除
# def drop_features(train, target):
#     # 相関係数の計算
#     train_ = train.drop([target], axis=1)
#     corr_matrix_ = train_.corr().abs()
#     corr_matrix = train.corr().abs()

#     # 相関係数が0.95以上の変数ペアを抽出
#     high_corr_vars = np.where(np.triu(corr_matrix_, k=1) > 0.95)
#     high_corr_pairs = [
#         (train_.columns[x], train_.columns[y]) for x, y in zip(*high_corr_vars)
#     ]
#     display(high_corr_pairs)

#     # 目的変数との相関係数が小さい方の変数を削除
#     for pair in high_corr_pairs:
#         var1_corr = corr_matrix.loc[target, pair[0]]
#         var2_corr = corr_matrix.loc[target, pair[1]]

#         try:  # 既に消した変数が入ってきたとき用
#             if var1_corr < var2_corr:
#                 train = train.drop(pair[0], axis=1)
#             else:
#                 train = train.drop(pair[1], axis=1)
#         except Exception as e:
#             print(f"How exceptional! {e}")
#             pass
#     return train


# %%
# クラス分けして似たものを削除
# def remove_collinear_features(train, target, threshold=1.0, s=0):
#     X = train.drop(target, axis=1)
#     y = train[target]
#     cols = X.columns
#     # 特徴量間の非類似性距離行列を計算
#     std = StandardScaler().fit_transform(X)
#     X_ = pd.DataFrame(std, columns=X.columns)  # 標準化
#     distances = np.zeros((X_.shape[1], X_.shape[1]))
#     for i in range(X_.shape[1]):
#         for j in range(i + 1, X_.shape[1]):
#             corr, _ = spearmanr(X_.iloc[:, i], X_.iloc[:, j])
#             distances[i, j] = distances[j, i] = 1 - abs(corr)
#     np.fill_diagonal(distances, 0)  # 対角成分をゼロに設定
#     distances = squareform(distances)

#     # Ward の最小分散基準で階層的クラスター分析
#     clusters = linkage(distances, method="ward")
#     cluster_labels = fcluster(clusters, threshold, criterion="distance")
#     # クラスター内で1つの特徴量のみ残す
#     unique_cluster_labels = np.unique(cluster_labels)
#     unique_features = []
#     for label in unique_cluster_labels:
#         features = X.columns[cluster_labels == label]
#         print(f"同じクラスタの特徴量は{features}です。")
#         if len(features) > 1:
#             print(f"選ばれたのは{features[s]}でした。")
#             unique_features.append(features[s])
#         else:
#             print(f"選ばれたのは{features}でした。")
#             unique_features.extend(features)

#     df = X[unique_features]
#     df[target] = y

#     return df, clusters, cols


# %%
# 祝日 アメリカ
# 祝日リスト アメリカ
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
    "2013-12-24",
    "2013-12-25",
    "2013-12-26",
    "2013-12-27",
    "2013-12-28",
    "2013-12-29",
    "2013-12-30",
    "2013-12-31",
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
    "2014-12-24",
    "2014-12-25",
    "2014-12-26",
    "2014-12-27",
    "2014-12-28",
    "2014-12-29",
    "2014-12-30",
    "2014-12-31",
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
    "2015-12-24",
    "2015-12-25",
    "2015-12-26",
    "2015-12-27",
    "2015-12-28",
    "2015-12-29",
    "2015-12-30",
    "2015-12-31",
]


# %%
# 季節
def month_to_season2(month):
    if month in [1, 2, 3]:
        return 4  # "Winter"
    elif month in [4, 5, 6]:
        return 1  # "Spring"
    elif month in [7, 8, 9]:
        return 2  # "Summer"
    elif month in [10, 11, 12]:
        return 3  # "Fall"
    else:
        return "Invalid month"


# 祝日1
def holiday(df):
    # date yyyy-mm-dd
    df["holiday"] = np.zeros(len(df))
    for i in holiday_list:
        df["holiday"][df["date"] == i] = 1  # "祝日"


# 平日1,祝日0
# workingday
def workingday(df):
    if df["dayofweek"] in ["Sat", "Sun"] or df["holiday"] == 1:
        return 0  # "土日祝"
    # elif df["dayofweek"] in ["Sat", "Sun"]:
    #     return 0  # "土日祝"
    else:
        return 1  # "平日"


# 通勤タイム
def commute(df):
    if df["workingday"] in [1] and df["hour"] in [7, 8, 17, 18]:
        return 1  # "通勤"
    else:
        return 0  # "その他"


# leisure()
def leisure(df):
    if df["workingday"] in [0] and df["hour"] in [11, 12, 13, 14, 15, 16, 17]:
        return 1  # "レジャー"
    else:
        return 0  # "その他"


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
# pd.set_option("display.max_rows", None)
# pd.set_option("display.max_columns", None)


# %% ファイルの読み込み
# Load Data
# =================================================
# status
status_df = load_data(2)

# %%
# 日付作成
# date yyyy-mm-dd
status_df["date"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    # + " "
    # + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date"] = pd.to_datetime(status_df["date"])  # .dt.strftime("%Y-%m-%d")

# date_h yyyy-mm-dd hh
status_df["date_h"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    + " "
    + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date_h"] = pd.to_datetime(status_df["date_h"]).dt.strftime("%Y-%m-%d %H")


# dayofweek 曜日カラムの作成 Sun
status_df["dayofweek"] = status_df["date"].dt.strftime("%a")


# date_h_a yyyy-mm-dd hh(a)
status_df["date_h_a"] = (
    status_df["date_h"].astype(str) + " " + " (" + status_df["dayofweek"] + ")"
)

# yearmonth yyyy-mm データを分ける用
status_df["yearmonth"] = status_df["date"].astype(str).apply(lambda x: x[:7])


# %%
# 特徴量作成
# 季節   1 "Spring"
status_df["season"] = status_df["month"].apply(month_to_season2)

# 祝日1
holiday(status_df)

# 平日1、土日祝0
status_df["workingday"] = status_df.apply(workingday, axis=1)

# 通勤タイム
# status_df["commute"] = status_df.apply(commute, axis=1)

# leisure
status_df["leisure"] = status_df.apply(leisure, axis=1)


# %%
# station読み込み
# =================================================
station_df = load_data(1)


# %%
# weather読み込み
# =================================================
weather_df = load_data(4)

# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])

# dayofweek 曜日カラムの作成 Sun
weather_df["dayofweek"] = weather_df["date"].dt.strftime("%a")
# %
holiday(weather_df)

# 平日1、土日祝0
weather_df["workingday"] = weather_df.apply(workingday, axis=1)


# %%
# よく利用されている時の特徴量を作成
# wether merge後のstatusでやる
def comforty(df):
    df["comfort"] = np.zeros(len(df))
    if (
        (55.3 <= df["mean_temperature"] <= 70.7)
        and (35 < df["mean_humidity"] < 75)
        and (df["max_wind_Speed"] < 19)
        and (df["workingday"] == 0)
    ):
        return 1  # いい天気
    else:
        return 0


# weather_df["comfort"] = weather_df.apply(comforty, axis=1)
# display(weather_df["comfort"].value_counts())


# %%
# 天気特徴量
# eventsをラベルエンコーディング
le_ = LabelEncoder()
weather_df["events"] = le_.fit_transform(weather_df["events"])
# %%
# 型変換
weather_df["events"] = weather_df["events"].astype(int)
weather_df["events"].value_counts()

# 3：なし、2：雨、１、きりと雨、０きり


# %%
# 天気が悪いかどうか
def events_to_events2(eve_number):
    if eve_number < 3:
        return 1
    else:
        return 0


weather_df["events2"] = weather_df["events"].apply(events_to_events2)

weather_df["events2"].value_counts()
# %%
# 0.3以上をバッドウェザーとする
weather_df["bad_weather"] = np.zeros(len(weather_df))


def bad_wether(amount):
    if amount > 0.3:
        return 1
    else:
        return 0


weather_df["bad_weather"] = weather_df["precipitation"].apply(bad_wether)
weather_df["bad_weather"].value_counts()


# %%
# trip読み込み
# =================================================
trip_df = load_data(3)

# %%
# time型変換
# time型変換
trip_df["start_date"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d %H")
trip_df["end_date"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d %H")


trip_df["start_date2"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d")
trip_df["end_date2"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d")

trip_df["start_hour"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%H")
trip_df["end_hour"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%H")

trip_df["dayofweek_st"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%a")
trip_df["dayofweek_end"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%a")


# %%
def shift_cumsum1(p_df, p_id, p_hour):
    # 時間、曜日、idを渡してそれぞれ毎の平均のラグと累積を計算したデータフレームを返す
    melt = pd.DataFrame()
    for i in np.sort(p_df[p_id].unique()):
        print(f"id[{i}]を処理")
        temp_a_df = p_df[p_df[p_id] == i]

        temp_b_df = (
            temp_a_df.pivot_table(
                index=p_hour,
                columns="dayofweek",
                values="count",
                aggfunc="sum",
            )
            / 104
        )
        temp_b_df.fillna(0, inplace=True)

        for j in p_df["dayofweek"].unique():
            print(f"{j}曜日の処理始めまーす")
            print(f"処理前：{melt.shape}")
            dow_ = pd.DataFrame()
            dow_ = temp_b_df[[j]]

            dow_["shift1"] = dow_.shift(1).interpolate(method="bfill")

            dow_["cumsum"] = dow_[j].cumsum()

            melt_ = pd.melt(
                dow_.reset_index(),
                id_vars=[p_hour, "shift1", "cumsum"],
                value_vars=j,
                var_name="dayofweek",
            )
            melt_["station_id"] = np.full((len(melt_)), i)
            melt = pd.concat([melt, melt_], axis=0)
            print(f"処理後：{melt.shape}")

    print("Done!")
    return melt


# %%
# カウント用
trip_df["count"] = np.ones(len(trip_df))

# %%
trip_df.rename(columns={"dayofweek_st": "dayofweek"}, inplace=True)
# %%
# 貸出集計
sum_cumsum_start = shift_cumsum1(trip_df, "start_station_id", "start_hour")

sum_cumsum_start.head()
# %%
# 返却集計
sum_cumsum_end = shift_cumsum1(trip_df, "end_station_id", "end_hour")
sum_cumsum_end.head()

# %%
# 全く利用されていない時間帯があるのでmerge後fillna(0)忘れないこと
sum_cumsum_start["station_id"].value_counts()

# %%
# 並び変えとカラム名変更
sum_cumsum_start = sum_cumsum_start[
    ["station_id", "dayofweek", "start_hour", "shift1", "cumsum"]
]

sum_cumsum_start.columns = [
    "station_id",
    "dayofweek",
    "hour",
    "shift1_start",
    "cumsum_start",
]

sum_cumsum_start["hour"] = sum_cumsum_start["hour"].astype(int)

# %%
# 並び変えとカラム名変更
sum_cumsum_end = sum_cumsum_end[
    ["station_id", "dayofweek", "end_hour", "shift1", "cumsum"]
]

sum_cumsum_end.columns = ["station_id", "dayofweek", "hour", "shift1_end", "cumsum_end"]

sum_cumsum_end["hour"] = sum_cumsum_end["hour"].astype(int)


# %%
# statusにstationのstation_idをマージ
# ===========================================
status_df = pd.merge(status_df, station_df, on="station_id", how="left")
# %%
# weatherをマージ
status_df = pd.merge(
    status_df,
    weather_df[
        [
            "date",
            "events2",
            "bad_weather",
        ]
    ],
    on="date",
    how="left",
)

# tripをマージ なし

# %%
# 貸出集計マージ
status_df = pd.merge(
    status_df, sum_cumsum_start, on=["station_id", "dayofweek", "hour"], how="left"
)

# 返却集計マージ
status_df = pd.merge(
    status_df, sum_cumsum_end, on=["station_id", "dayofweek", "hour"], how="left"
)


# %%
status_df["shift1_minus"] = status_df["shift1_end"] - status_df["shift1_start"]

status_df["cumsum_minus"] = status_df["cumsum_end"] - status_df["cumsum_start"]

status_df.head()

# %%
# 集計の欠損値を後の値で埋める
status_df[
    [
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
    ]
] = status_df[
    [
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
    ]
].fillna(0)
status_df.head()


# %%
# 0時の特徴量
t = status_df.groupby(["station_id", "date"]).first()["bikes_available"].reset_index()

t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
t.columns = ["staion_id", "date", "bikes_available_at0"]
t.head(25)

status_df["bikes_available_at0"] = t["bikes_available_at0"].astype("float16")

# %%
status_df["0_cumsum"] = status_df["cumsum_minus"] + status_df["bikes_available_at0"]

# %%
#
# %% [markdown]
## LSTM 分析start!
# ==========================================================

# %%
# データセット作成
main_df = status_df[
    [
        "date",
        "hour",
        "station_id",
        "bikes_available",
        "dayofweek",
        "predict",
    ]
]
main_df.head()

# %%
main_df = pd.get_dummies(main_df, dtype="uint8")
print(main_df.columns)
print(main_df.shape)

# %%
main_df.head()
# %%
print(main_df.isna().sum())


# %%
# 学習用のデータフレーム作成
train_dataset_df = main_df[main_df["date"] < "2014-09-01"]
# 評価用のデータフレーム作成（使用するモデルの関係上、前日のデータが必要なため2014-08-31から取得）
evaluation_dataset_df = main_df[main_df["date"] >= "2014-08-31"]

# %%

print(train_dataset_df.isna().sum())


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
def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_pred - y_true))


# LSTMの学習
model = Sequential()
# 1層目のLSTM (Dropout 30%)
model.add(LSTM(100, return_sequences=True, input_shape=(trainX.shape[1], 1680)))
model.add(BatchNormalization())
model.add(Dropout(0.3))
# 2層目のLSTM (Dropout 30%)
model.add(LSTM(100, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# 3層目のLSTM (Dropout 30%)
model.add(LSTM(100, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.2))

# 4層目のLSTM (Dropout 20%)
model.add(LSTM(50, return_sequences=False))  # 最終層なので return_sequences=False
model.add(BatchNormalization())
model.add(Dropout(0.1))
# 出力層
model.add(Dense(70))
model.compile(loss="mse", optimizer="adam", metrics=[rmse])
hist = model.fit(
    trainX,
    trainY,
    epochs=20,
    batch_size=1,
    callbacks=[
        ModelCheckpoint(
            filepath="model_keras.weights.h5",
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
            patience=5,
            verbose=1,
            restore_best_weights=True,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", mode="min", factor=0.1, patience=5, verbose=1
        ),
    ],
    verbose=2,
)




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
display(submission_df)
submission_df.to_csv(
    os.path.join(OUTPUT_EXP, "pre_lstm_submission.csv")
)  ##,header=None

# %%


# %%
lstm_submit_df = submission_df[submission_df["predict"] == 1].sort_values(
    ["station_id", "date"]
)[["bikes_available"]]
lstm_submit_df["bikes_available"] = lstm_submit_df["bikes_available"].map(
    lambda x: 0 if x < 0 else x
)
lstm_submit_df.index = status_df[status_df["predict"] == 1].index
lstm_submit_df.to_csv(os.path.join(OUTPUT_EXP, "lstm_submission.csv"), header=None)  #
lstm_submit_df.head()


# %%
# 読み込み

# %%
submission_df = pd.read_csv(
    "/opt/src/output/exp111_em_1014/pre_lstm_submission.csv", index_col=0
)

lstm_submit_df = pd.read_csv(
    "/opt/src/output/exp111_em_1014/lstm_submission.csv", header=None, index_col=0
)
# %%
display(submission_df.head())
display(lstm_submit_df.head())
# %%
lstm_submit_df.columns = ["pred"]

# %%
submission_df["date"] = pd.to_datetime(submission_df["date"])

# %%
submission_df["lstm_pred"] = submission_df["bikes_available"].map(
    lambda x: 0 if x < 0 else x
)

status_df = pd.merge(
    status_df,
    submission_df[["date", "hour", "station_id", "lstm_pred"]],
    on=["date", "hour", "station_id"],
    how="left",
)

status_df.head()
# %%
status_df["bikes_available2"] = status_df["bikes_available"].copy()

status_df["bikes_available2"] = status_df["bikes_available2"].fillna(
    status_df["lstm_pred"]
)

status_df.isna().sum()


# %%


# %%
# lgbm
# ================================

# %%
# ラグ


# 1週間前（同じ曜日）の同じ時刻の台数
status_df["available_shift168"] = status_df.groupby("station_id")[
    "bikes_available2"
].shift(168)


# %% 移動平均
# # 1month
status_df["weekday_hour_1m_mean"] = (
    status_df.groupby("station_id")["bikes_available2"].shift(168)
    + status_df.groupby("station_id")["bikes_available2"].shift(336)
    + status_df.groupby("station_id")["bikes_available2"].shift(504)
    + status_df.groupby("station_id")["bikes_available2"].shift(672)
) / 4.0


# 7
status_df["hour_1w_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    .interpolate(method="bfill")
)

status_df["hour_1w_std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).std())
    .interpolate(method="bfill")
)


# 30
status_df["hour_1m_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 3month
status_df["hour_3m_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=90, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 3d
status_df["3_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=72, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 7d
status_df["7_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=168, min_periods=1).mean())
    .interpolate(method="bfill")
)


status_df["30_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=720, min_periods=1).mean())
    .interpolate(method="bfill")
)

# 90d
status_df["90_mean"] = (
    status_df.groupby(
        [
            "station_id",
        ]
    )["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=2160, min_periods=1).mean())
    .interpolate(method="bfill")
)
# 180d
status_df["180_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=43420, min_periods=1).mean())
    .interpolate(method="bfill")
)


# %%
# ゼロフラグ
# tmp_ = pd.DataFrame()
def flg_0(count):
    if count == 0:
        return 1
    else:
        return 0


status_df["flg_0"] = status_df["bikes_available2"].apply(flg_0)
display(status_df["flg_0"].value_counts())

# %%
near_df2_1019 = pd.read_csv("/opt/src/output/exp202_minus0_1016/near_df.csv")
near_df2_1019.head()
# %%
status_df = pd.merge(status_df, near_df2_1019, on=["date_h", "station_id"], how="left")


# %%
# カテゴリ変数
status_df = data_pre00(status_df)

# %%
# データセット作成
# 2014/09/01以前をtrain、以後をtestに分割(testはpredict = 1のものに絞る)
# =================================================


train_lgbm = status_df[status_df["date"] < "2014-09-01"]

valid_lgbm = status_df[
    (status_df["date"] >= "2014-09-01") & (status_df["predict"] == 0)
]


test_lgbm = status_df[(status_df["date"] >= "2014-09-01") & (status_df["predict"] == 1)]


# %%
# train_lgbmはbikes_available2がnanではないものに絞る
train_lgbm = train_lgbm[train_lgbm["bikes_available"].notna()]
valid_lgbm = valid_lgbm[valid_lgbm["bikes_available"].notna()]

# %%
train_lgbm.columns


# %%
# 使用カラム

train_lgbm_df = train_lgbm[
    [
        # "id",
        # "year",
        # "month",
        # "day",
        "hour",
        "station_id",
        # "predict",
        # "date",
        # "date_h",
        "dayofweek",
        # "date_h_a",
        # "yearmonth",
        "season",
        "holiday",
        "workingday",
        # "commute",
        "leisure",
        # "lat",
        # "long",
        "dock_count",
        # "city",
        # "installation_date",
        # "mean_temperature",
        # "mean_humidity",
        # "max_wind_Speed",
        "events2",
        "bad_weather",
        # "comfort",
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
        "available_shift168",
        "weekday_hour_1m_mean",
        "hour_1w_mean",
        "hour_1w_std",
        "hour_1m_mean",
        "hour_3m_mean",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        "180_mean",
        "flg_0",
        # "near_flg_0",
        # "near_flg_max",
        "near_available",
        "bikes_available",
        "bikes_available_at0",
        # "0_cumsum"
    ]
]

valid_lgbm_df = valid_lgbm[
    [
        # "id",
        # "year",
        # "month",
        # "day",
        "hour",
        "station_id",
        # "predict",
        # "date",
        # "date_h",
        "dayofweek",
        # "date_h_a",
        # "yearmonth",
        "season",
        "holiday",
        "workingday",
        # "commute",
        "leisure",
        # "lat",
        # "long",
        "dock_count",
        # "city",
        # "installation_date",
        # "mean_temperature",
        # "mean_humidity",
        # "max_wind_Speed",
        "events2",
        "bad_weather",
        # "comfort",
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
        "available_shift168",
        "weekday_hour_1m_mean",
        "hour_1w_mean",
        "hour_1w_std",
        "hour_1m_mean",
        "hour_3m_mean",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        "180_mean",
        "flg_0",
        # "near_flg_0",
        # "near_flg_max",
        "near_available",
        "bikes_available",
        "bikes_available_at0",
        # "0_cumsum"
    ]
]


id_train_lgbm = train_lgbm[["yearmonth"]]


test_lgbm_df = test_lgbm[
    [
        # "id",
        # "year",
        # "month",
        # "day",
        "hour",
        "station_id",
        # "predict",
        # "date",
        # "date_h",
        "dayofweek",
        # "date_h_a",
        # "yearmonth",
        "season",
        "holiday",
        "workingday",
        # "commute",
        "leisure",
        # "lat",
        # "long",
        "dock_count",
        # "city",
        # "installation_date",
        # "mean_temperature",
        # "mean_humidity",
        # "max_wind_Speed",
        "events2",
        "bad_weather",
        # "comfort",
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
        "available_shift168",
        "weekday_hour_1m_mean",
        "hour_1w_mean",
        "hour_1w_std",
        "hour_1m_mean",
        "hour_3m_mean",
        "3_mean",
        "7_mean",
        "30_mean",
        "90_mean",
        "180_mean",
        "flg_0",
        # "near_flg_0",
        # "near_flg_max",
        "near_available",
        # "bikes_available",
        "bikes_available_at0",
        # "0_cumsum"
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
        ],
        ["2013-12"],
    ],
    [
        [
            "2013-10",
            "2013-11",
            "2013-12",
        ],
        ["2014-01"],
    ],
    [
        [
            "2013-11",
            "2013-12",
            "2014-01",
        ],
        ["2014-02"],
    ],
    [
        [
            "2013-12",
            "2014-01",
            "2014-02",
        ],
        ["2014-03"],
    ],
    [
        [
            "2014-01",
            "2014-02",
            "2014-03",
        ],
        ["2014-04"],
    ],
    [
        [
            "2014-02",
            "2014-03",
            "2014-04",
        ],
        ["2014-05"],
    ],
    [
        [
            "2014-03",
            "2014-04",
            "2014-05",
        ],
        ["2014-06"],
    ],
    [
        [
            "2014-04",
            "2014-05",
            "2014-06",
        ],
        ["2014-07"],
    ],
    [
        [
            "2014-05",
            "2014-06",
            "2014-07",
        ],
        ["2014-08"],
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
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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
        va:{metrics[:,2].mean():.4f}+-{metrics[:,2].std():.4f}")

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

    return imp, metrics  # , shap_v


# %%
# train #, shap_v
imp, metrics = train_lgb(
    x_train_lgbm,
    y_train_lgbm,
    id_train_lgbm,
    params,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)


# %%
# imp
imp_sort = imp.sort_values("imp", ascending=False)
display(imp_sort)
# 重要度をcsvに出力
# imp_sort.to_csv(os.path.join(OUTPUT_EXP, f"imp_{name}.csv"), index=False, header=False)
# %%
# shap_sort = shap_v.sort_values("shap", ascending=False)
# display(shap_sort)

# shap_value = shap_sort.copy()
# imp_value = imp_sort.copy()

# %%
# 一個ずつ加えて精度確認
# select_list = []
# scores = []
# for i in imp_value["col"]:  # [:20]:
#     select_list.append(i)
#     print(select_list)
#     x_trains = x_train_lgbm[select_list]
#     print(x_trains.shape)
#     imp, metrics = train_lgb(
#         x_trains,
#         y_train_lgbm,
#         id_train_lgbm,
#         params,
#         list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
#     )
#     scores.append([len(select_list), metrics[:, 2].mean()])


# %%
# プロット作成
# scores = pd.DataFrame(scores)
# scores.head()
# """
# 精度が改善されなくなる場所がどこが確認する
# """
# sns.lineplot(data=scores, x=0, y=1)


# %%
# scores.head(50)
# # %%
# display(shap_sort[:26].reset_index())
# print(shap_sort[:26].shape)
# list(shap_sort["col"][:26].values)

# # %%
# # 削除
# list(shap_sort["col"][26:].values)

# %%
# チューニング


# %%
# 探索しないハイパーパラメータ
# params_base = {
#     "boosting_type": "gbdt",
#     "objective": "regression",
#     "metric": "rmse",
#     "verbosity": -1,
#     "learning_rate": 0.05,
#     "n_estimators": 100000,
#     "bagging_freq": 1,
#     "random_state": 123,
# }


# # 目的関数の定義
# def objective(trial):
#     # 探索するハイパーパラメータ
#     params_tuning = {
#         "num_leaves": trial.suggest_int("num_leaves", 8, 256),
#         "min_child_samples": trial.suggest_int("min_child_samples", 5, 200),
#         "min_sum_hessian_in_leaf": trial.suggest_float(
#             "min_sum_hessian_in_leaf", 1e-5, 1e-2, log=True
#         ),
#         "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
#         "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
#         "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1e2, log=True),
#         "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1e2, log=True),
#     }
#     params_tuning.update(params_base)

#     # モデル学習・評価
#     list_metrics = []
#     cv = []
#     for month_tr, month_va in list_cv_month:
#         cv.append(
#             [
#                 id_train_lgbm.index[id_train_lgbm["yearmonth"].isin(month_tr)],
#                 id_train_lgbm.index[id_train_lgbm["yearmonth"].isin(month_va)],
#             ]
#         )

#     list_fold = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # 処理高速化のため1つ目のfoldのみとする
#     for nfold in list_fold:
#         idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

#         x_tr, y_tr = x_train_lgbm.loc[idx_tr, :], y_train_lgbm.loc[idx_tr]

#         x_va, y_va = x_train_lgbm.loc[idx_va, :], y_train_lgbm.loc[idx_va]

#         model = lgb.LGBMRegressor(**params_tuning)
#         model.fit(
#             x_tr,
#             y_tr,
#             eval_set=[(x_tr, y_tr), (x_va, y_va)],
#             callbacks=[
#                 lgb.early_stopping(stopping_rounds=100, verbose=True),
#                 lgb.log_evaluation(0),
#             ],
#         )

#         y_va_pred = model.predict(x_va)
#         metric_va = mean_squared_error(y_va, y_va_pred, squared=False)  # 評価指標RMSE
#         list_metrics.append(metric_va)

#     # 評価指標の算出
#     metrics = np.mean(list_metrics)
#     return metrics


# %%
# 7-50:最適化処理（探索の実行）
# sampler = optuna.samplers.TPESampler(seed=123)
# study = optuna.create_study(sampler=sampler, direction="minimize")
# study.optimize(objective, n_trials=50, n_jobs=5)

# %%
# trial = study.best_trial
# print(f"RMSE(best)={trial.value:.4f}")
# display(trial.params)
# # %%
# params_beat = trial.params
# params_beat.update(params_base)
# display(params_beat)


# %%

# %%
# シード数変更によるアンサンブル学習
seed_values = [1, 3, 11, 13, 42, 123, 456, 789, 1988, 3407]
n_models = len(seed_values)
print(f"モデルの数：{len(seed_values)}")

metrics_mix = np.zeros((len(x_train_lgbm), n_models))


for i, seed_value in enumerate(seed_values):
    print(f"Training model with seed {seed_value}")
    params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.05,
        "num_leaves": 32,
        "n_estimators": 10000,
        "random_state": seed_value,
        "importance_type": "gain",
    }
    # params["random_state"] = seed_value  # シード値を変更
    imp, metrics = train_lgb(
        x_train_lgbm,
        y_train_lgbm,
        id_train_lgbm,
        params,
        list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )

    metrics_mix[:, i] = metrics[:, 2].mean()


print(f"[oof_mean]{metrics_mix.mean(axis=1):.4f}")


# %%
# 推論関数の定義 検証用 =================================================


def predict_lgb(
    input_x,
    input_y,
    input_id,
    list_nfold=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict(input_x)

    # 平均値算出
    pred = pd.concat(
        [
            input_id,
            pd.DataFrame(pred.mean(axis=1)),
        ],
        axis=1,
    )
    print("Done.")

    metric_valid = mean_squared_error(input_y, pred.iloc[:, 1], squared=False)
    print(f"[rmse] valid:{metric_valid:.4f}")
    return pred


# %%
# 検証処理
# =================================================
valid_lgbm_x = valid_lgbm_df.drop("bikes_available", axis=1)
valid_lgbm_y = valid_lgbm_df["bikes_available"]

id_valid_lgbm = pd.DataFrame(valid_lgbm.index)

# %%
valid_lgbm_oof = predict_lgb(
    valid_lgbm_x,
    valid_lgbm_y,
    id_valid_lgbm,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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

id_test_lgbm = pd.DataFrame(test_lgbm.index)

# %%
# 提出用推論関数の定義 =================================================


def predict_sub_lgb(
    input_x,
    input_id,
    list_nfold=[
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
    ],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict(input_x)

    # 平均値算出
    pred = pd.concat(
        [
            input_id,
            pd.DataFrame(pred.mean(axis=1)),
        ],
        axis=1,
    )
    print("Done.")

    return pred


# %%
test_pred = np.zeros((len(test_lgbm_df), n_models))

for i, seed_value in enumerate(seed_values):
    print(f"predict with seed {seed_value}")

    test_pred[:, i] = predict_sub_lgb(
        test_lgbm_df,
        id_test_lgbm,
        list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    )

final_pred = test_pred.mean(axis=1)


# %%
# 元
# test_lgbm_pred = predict_sub_lgb(
#     test_lgbm_df,
#     id_test_lgbm,
#     list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
# )


# %%
final_pred.columns = ["id", "pred"]
final_pred.head()
# %%
final_pred.set_index("id", inplace=True)

# %%
# submitファイルの出力
# =================================================
# test_lgbm_pred.to_csv(
#     os.path.join(OUTPUT_EXP, f"submission_{name}.csv"),
#     header=False,index=False
# )


# %%


# %% [markdown]
## submitファイルの作成!
# ==========================================================
display(lstm_submit_df.head(10))
display(final_pred.head(10))


# %%
# 単純平均
test_pred2 = pd.DataFrame()

test_pred2["pred"] = (lstm_submit_df["bikes_available"] + final_pred["pred"]) / 2

test_pred2.index = status_df[status_df["predict"] == 1].index
# %%
test_pred2.head()

# %%
# submitファイルの出力
# =================================================
test_pred2.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.csv"),
    header=False,  # ,index=False
)

# %%
