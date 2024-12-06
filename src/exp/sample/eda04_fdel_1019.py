# %% [markdown]
## 特徴量生成！
# =================================================
# city2探る valid003


# %%
# ライブラリ読み込み
# =================================================
import datetime as dt
# from datetime import timedelta

# import gc
# import json
import logging
# import math

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
# import japanize_matplotlib

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
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.optimizers import Adam

# 次元圧縮
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# クラスタで選別
# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr

from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import requests

from sklearn.cluster import KMeans


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 27  # スプレッドシートAの番号

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
# target_columns = "bikes_available"
# sub_index = "id"

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


# %%
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
# 季節 使わない
# def month_to_season2(month):
#     if month in [1, 2, 3]:
#         return 4  # "Winter"
#     elif month in [4, 5, 6]:
#         return 1  # "Spring"
#     elif month in [7, 8, 9]:
#         return 2  # "Summer"
#     elif month in [10, 11, 12]:
#         return 3  # "Fall"
#     else:
#         return "Invalid month"


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
# def commute(df):
#     if df["workingday"] in [1] and df["hour"] in [7, 8, 17, 18]:
#         return 1  # "通勤"
#     else:
#         return 0  # "その他"


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
sns.set()  # font="IPAexGothic"
####!%matplotlib inline
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
# status_df["date_h_a"] = (
#     status_df["date_h"].astype(str) + " " + " (" + status_df["dayofweek"] + ")"
# )

# yearmonth yyyy-mm データを分ける用
status_df["yearmonth"] = status_df["date"].astype(str).apply(lambda x: x[:7])


# %%
# 特徴量作成

# 祝日1
holiday(status_df)

# 平日1、土日祝0
status_df["workingday"] = status_df.apply(workingday, axis=1)


# leisure
status_df["leisure"] = status_df.apply(leisure, axis=1)


# %%
# station読み込み
# =================================================
# station_df = load_data(1)
station_df = pd.read_csv(file_list["ファイルパス"][1])
station_df.head()
# 日付に変更
station_df["installation_date"] = pd.to_datetime(station_df["installation_date"])

# %%
# Overpass APIのURL
OVERPASS_URL = "http://overpass-api.de/api/interpreter"


# Overpass APIで駅を検索するクエリ (経度・緯度の周辺半径100m)
def get_nearby_stations(lat, lon, radius=100):
    # Overpass QLクエリを作成
    overpass_query = f"""
    [out:json];
    (
        node["railway"="station"](around:{radius},{lat},{lon});
    );
    out body;
    """

    # APIリクエストを送信
    response = requests.get(OVERPASS_URL, params={"data": overpass_query})

    if response.status_code == 200:
        return response.json()["elements"]
    else:
        raise Exception(f"Error: {response.status_code} - {response.text}")


# 距離が近いかどうかを確認する関数
def is_near_station(lat, lon, station_lat, station_lon, max_distance=0.5):
    # 2地点間の距離を計算 (キロメートル単位)
    distance = geodesic((lat, lon), (station_lat, station_lon)).km
    return distance <= max_distance, distance


# %%
near_station = []
for i in range(station_df.shape[0]):
    lat_ = station_df.loc[i, "lat"]
    long_ = station_df.loc[i, "long"]
    print(f"station_id[{i}] lat:{lat_} ,long{long_}")
    stations = get_nearby_stations(lat_, long_)

    if stations:
        for station in stations:
            station_lat = station["lat"]
            station_lon = station["lon"]
            near, distance = is_near_station(lat_, long_, station_lat, station_lon)

            if near:
                print(
                    f"駅前です: {station['tags'].get('name', '駅名不明')}, 距離: {distance*1000:.2f}m"
                )
                n_tmp = [i, distance * 1000]
            else:
                print(
                    f"駅: {station['tags'].get('name', '駅名不明')} は {distance*1000:.2f}m 離れています"
                )
    else:
        print("近くに駅が見つかりません")
    near_station.append(n_tmp)

# %%
# 近くに駅があるかどうかのフラグ
near_station.columns = ["station_id", "distance"]
near_station = pd.DataFrame(near_station)
near_station = near_station.groupby("station_id")[["distance"]].min()


# %%
near_station["near_station_flg"] = np.ones(len(near_station))

station_df = pd.merge(station_df, near_station, on="station_id", how="left")

# %%
# weather読み込み
# =================================================
weather_df = load_data(4)

# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])

# dayofweek 曜日カラムの作成 Sun
weather_df["dayofweek"] = weather_df["date"].dt.strftime("%a")

holiday(weather_df)

# 平日1、土日祝0
weather_df["workingday"] = weather_df.apply(workingday, axis=1)


# %%
# 天気特徴量
# eventsをラベルエンコーディング
le_ = LabelEncoder()
weather_df["events"] = le_.fit_transform(weather_df["events"])

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
                # margins=True,
                # margins_name="Total",
            )
            / 52
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
    ["station_id", "dayofweek", "start_hour", "shift1", "cumsum", "value"]
]

sum_cumsum_start.columns = [
    "station_id",
    "dayofweek",
    "hour",
    "shift1_start",
    "cumsum_start",
    "value_start",
]

sum_cumsum_start["hour"] = sum_cumsum_start["hour"].astype(int)

# %%
# 並び変えとカラム名変更
sum_cumsum_end = sum_cumsum_end[
    ["station_id", "dayofweek", "end_hour", "shift1", "cumsum", "value"]
]

sum_cumsum_end.columns = [
    "station_id",
    "dayofweek",
    "hour",
    "shift1_end",
    "cumsum_end",
    "value_end",
]

sum_cumsum_end["hour"] = sum_cumsum_end["hour"].astype(int)


# %%
# 全体に対してのステーション毎の利用割合
temp_id = (
    (trip_df["start_station_id"].value_counts().sort_index() / len(trip_df))
    + (trip_df["end_station_id"].value_counts().sort_index() / len(trip_df))
) / 2

temp_id = pd.DataFrame(temp_id).reset_index()

temp_id.columns = ["station_id", "utili_rate"]
temp_id.head()


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
            "mean_temperature",
            "mean_humidity",
            "max_wind_Speed",
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
# ステーション毎の利用割合マージ
status_df = pd.merge(status_df, temp_id, on="station_id", how="left")
status_df.head()
# %%
# %%
# 返却ー貸出
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
        "value_start",
        "value_end",
    ]
] = status_df[
    [
        "shift1_start",
        "cumsum_start",
        "shift1_end",
        "cumsum_end",
        "shift1_minus",
        "cumsum_minus",
        "value_start",
        "value_end",
    ]
].fillna(0)
status_df.head()


# %%
# 0時の特徴量
t = status_df.groupby(["station_id", "date"]).first()["bikes_available"].reset_index()

t = pd.DataFrame(np.repeat(t.values, 24, axis=0))
t.columns = ["staion_id", "date", "bikes_available_at0"]

status_df["bikes_available_at0"] = t["bikes_available_at0"].astype("float16")



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
# lgbm
# ================================

# %% 移動平均

# 7
status_df["hour_1w_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=7, min_periods=1).mean())
    .interpolate(method="bfill")
)


# 30
status_df["hour_1m_mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available2"]
    .transform(lambda x: x.shift(1).rolling(window=30, min_periods=1).mean())
    .interpolate(method="bfill")
)



# %%
# カテゴリ変数
status_df = data_pre00(status_df)

# %%
# city列をcategorical encoderを用いて数値化
cols = ["city"]

# OneHotEncodeしたい列を指定
encoder = ce.OneHotEncoder(cols=cols, handle_unknown="impute")
temp_ = encoder.fit_transform(status_df[cols]).add_prefix("CE_")

status_df = pd.concat([status_df, temp_], axis=1)
# %%
#EDA
status_df_1 = status_df[status_df["CE_city_1"] == 1]
status_df_2 = status_df[status_df["CE_city_2"] == 1]
status_df_3 = status_df[status_df["CE_city_3"] == 1]
status_df_4 = status_df[status_df["CE_city_4"] == 1]
status_df_5 = status_df[status_df["CE_city_5"] == 1]

# %%
display(status_df_1["station_id"].unique())
display(status_df_2["station_id"].unique())
display(status_df_3["station_id"].unique())
display(status_df_4["station_id"].unique())
display(status_df_5["station_id"].unique())


# %%
# 利用可能台数のヒストグラム
sns.histplot(status_df_1["bikes_available"])
# %%
sns.histplot(status_df_2["bikes_available"])
# %%
sns.histplot(status_df_3["bikes_available"])
# %%
sns.histplot(status_df_4["bikes_available"])
# %%
sns.histplot(status_df_5["bikes_available"])


# %%
# 欠品がどのくらい出ているかステーション毎に表示
sns.histplot(status_df[status_df["bikes_available"] == 0]["station_id"])

# %%
# city2の分布
sns.scatterplot(data=status_df_5, x="lat", y="long")


# %%
# 欠品がどの程度起こっているか
display(status_df_1[(status_df_1["bikes_available"] == 0)].shape)
display(status_df_2[(status_df_2["bikes_available"] == 0)].shape)
display(status_df_3[(status_df_3["bikes_available"] == 0)].shape)
display(status_df_4[(status_df_4["bikes_available"] == 0)].shape)
display(status_df_5[(status_df_5["bikes_available"] == 0)].shape)

# %%
display(status_df_1[status_df_1["bikes_available"] == status_df_1["dock_count"]].shape)

display(status_df_2[status_df_2["bikes_available"] == status_df_2["dock_count"]].shape)

display(status_df_3[status_df_3["bikes_available"] == status_df_3["dock_count"]].shape)

display(status_df_4[status_df_4["bikes_available"] == status_df_4["dock_count"]].shape)

display(status_df_5[status_df_5["bikes_available"] == status_df_5["dock_count"]].shape)

# %%
# 特徴量作成
# 0時時点の台数/最大収容数
status_df_5["0/dock"] = status_df_5["bikes_available_at0"] / status_df_5["dock_count"]


# 現在時点の収容率とラグ
# これは使わない
status_df_5["target/dock"] = status_df_5["bikes_available"] / status_df_5["dock_count"]

status_df_5["target/dock_shift1"] = (
    status_df_5.groupby("station_id")["target/dock"]
    .shift(1)
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


status_df_5["flg_0"] = status_df_5["bikes_available"].apply(flg_0)
display(status_df_5["flg_0"].value_counts())


# %%
# 500m以内にあるステーションを探す
# status_df_5_n = station_df[["station_id", "lat", "long"]]

# nearby_stations = []
# for id in range(len(status_df_5_n)):
#     target_station = (status_df_5_n.iloc[id, 1], status_df_5_n.iloc[id, 2])
#     for index in range(len(status_df_5_n)):
#         station_latlon = (status_df_5_n.iloc[index, 1], status_df_5_n.iloc[index, 2])
#         if target_station != station_latlon:
#             distance = geodesic(target_station, station_latlon).kilometers
#             if distance <= 0.5:  # 指定した距離以内ならリストに追加
#                 nearby_stations.append(
#                     {
#                         "中心station": status_df_5_n.iloc[id, 0],
#                         "station_id": status_df_5_n.iloc[index, 0],
#                         "distance_km": distance,
#                     }
#                 )

# %%
# リスト化処理
# nearby_stations = pd.DataFrame(nearby_stations)

# abc = []
# for i in status_df_5["station_id"].unique():
#     a = nearby_stations[nearby_stations["中心station"] == i]["station_id"].tolist()
#     abc.append([i, a])
# abc
# %%
# abcd = dict(abc)


# 近くのステーションで自転車がゼロのステーション数をカウント
# def near_zero(df):
#     tmp_near = []
#     for time in df["date_h"].unique():
#         for key, value in abcd.items():
#             a, b = 0, 0
#             a = df[
#                 (df["date_h"] == time)
#                 & (df["station_id"].isin(value))
#                 & (df["flg_0"] == 1)
#             ].shape[0]
#             b = df[
#                 (df["date_h"] == time)
#                 & (df["station_id"].isin(value))
#                 & (df["flg_max"] == 1)
#             ].shape[0]
#             tmp_near.append([time, key, a, b])
#             c = (
#                 df[(df["date_h"] == time) & (df["station_id"].isin(value))][
#                     "bikes_available"
#                 ].sum()
#             ) / len(value)
#             tmp_near.append([time, key, a, b, c])

#     return tmp_near

# %%
# tmp_near = near_zero(status_df_5)
# display(tmp_near)

# near_df = pd.DataFrame(tmp_near)
# near_df.head()

# near_df.columns = [
#     "date_h",
#     "station_id",
#     "near_flg_0",
#     "near_flg_max",
#     "near_available",
# ]


# near_df2 = (
#     near_df.groupby(["date_h", "station_id"])[
#         ["near_flg_0", "near_flg_max", "near_available"]
#     ]
#     .mean()
#     .reset_index()
# )


# %%
# near_df2.to_csv(os.path.join(OUTPUT_EXP,"near_df.csv"),index=False)
near_df2 = pd.read_csv("/opt/src/output/exp202_minus0_1016/near_df.csv")
near_df2.head()
# %%
status_df_5 = pd.merge(status_df_5, near_df2, on=["date_h", "station_id"], how="left")

# %%

status_df_5["flg_0_shift1"] = (
    status_df_5.groupby("station_id")["flg_0"].shift(1).interpolate(method="bfill")
)


status_df_5["near_flg_0_shift1"] = (
    status_df_5.groupby("station_id")["near_flg_0"].shift(1).interpolate(method="bfill")
)


status_df_5["near_flg_max_shift1"] = (
    status_df_5.groupby("station_id")["near_flg_max"]
    .shift(1)
    .interpolate(method="bfill")
)

status_df_5["near_available_shift1"] = (
    status_df_5.groupby("station_id")["near_available"]
    .shift(1)
    .interpolate(method="bfill")
)

# %%
station_df_5 = station_df[station_df["city"] == "city2"]
station_df_5.head()
# %%
station_df_5["near_station_flg"].fillna(0, inplace=True)
# %%
station_df_5_set = station_df_5[["lat", "long", "dock_count", "near_station_flg"]]
station_df_5_set.head()
# %%
sc = StandardScaler()
station_df_5_set_sc = sc.fit_transform(station_df_5_set)
# %%
kmeans = KMeans(n_clusters=5, random_state=123)
clusters = kmeans.fit(station_df_5_set_sc)

station_df_5_set = station_df_5_set.assign(cluster=clusters.labels_)


# %%
station_df_5 = pd.concat([station_df_5, station_df_5_set["cluster"]], axis=1)
station_df_5.head()
# %%
status_df_5 = pd.merge(
    status_df_5, station_df_5[["station_id", "cluster"]], on="station_id", how="left"
)

