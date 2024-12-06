# %% [markdown]
## 特徴量生成！
# =================================================
#  移動平均 元：直前1ヶ月を検証*12モデル


# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
# import json
import logging

# import re
import os
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


# lightGBM
import lightgbm as lgb

# sckit-learn
# from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, train_test_split  # , KFold
from sklearn.metrics import (
    mean_squared_error,
)  # ,accuracy_score, roc_auc_score ,confusion_matrix


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 17  # スプレッドシートAの番号

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
status_df.isna().sum()
# %%
status_df.describe().T
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

status_df["week_num"] = status_df["date"].dt.weekday
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
status_df[status_df["predict"] == 1].head()

# %%
status_df[status_df["predict"] == 1].tail()
# %%
status_df[status_df["predict"] == 1].nunique()


# %%
# station読み込み
station_df = load_data(1)

# 日付に変更
station_df["installation_date"] = pd.to_datetime(station_df["installation_date"])
station_df["installation_date"].info()
station_df["installation_date"].head()
# %%

print(station_df["installation_date"].min())
print(station_df["installation_date"].max())


# %%
display(station_df["city"].value_counts())
# %%


# %%
# weather読み込み
weather_df = load_data(4)
# %%
# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df.info()


# %%
# trip読み込み
trip_df = load_data(3)

# %%
trip_df.info()


# %%
trip_df["start_date"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d %H")
trip_df["end_date"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d %H")

trip_df.head()

# %%
trip_df["subscription_type"].value_counts()


# %%
trip_tbl_s = pd.crosstab(trip_df["start_date"], trip_df["start_station_id"])
display(trip_tbl_s)
# %%
trip_tbl_s = trip_df.groupby(["start_date", "start_station_id"])["trip_id"].count()
display(trip_tbl_s)
trip_tbl_s = trip_tbl_s.reset_index()
trip_tbl_s.columns = ["date2", "station_id", "count_start"]

display(trip_tbl_s)


# %%
trip_tbl_e = trip_df.groupby(["end_date", "end_station_id"])["trip_id"].count()
display(trip_tbl_e)
trip_tbl_e = trip_tbl_e.reset_index()
trip_tbl_e.columns = ["date2", "station_id", "count_end"]

display(trip_tbl_e)

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

display(status_df.head())
display(status_df.info())

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
# カテゴリ変数
status_df = data_pre00(status_df)
# %%
# 移動平均
# 35
status_df["dow_hour_5mean"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=5, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["dow_hour_5std"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=5, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["dow_hour_5min"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=5, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["dow_hour_5max"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=5, min_periods=2).max())
    .interpolate(method="bfill")
)

# 105
status_df["dow_hour_15mean"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=15, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["dow_hour_15std"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=15, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["dow_hour_15min"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=15, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["dow_hour_15max"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=15, min_periods=2).max())
    .interpolate(method="bfill")
)


# 210
status_df["dow_hour_30mean"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["dow_hour_30std"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["dow_hour_30min"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["dow_hour_30max"] = (
    status_df.groupby(["station_id", "week_num", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).max())
    .interpolate(method="bfill")
)

# 7
status_df["hour_7mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=7, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["hour_7std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=7, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["hour_7min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=7, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["hour_7max"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=7, min_periods=2).max())
    .interpolate(method="bfill")
)

# 30
status_df["hour_30mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["hour_30std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["hour_30min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["hour_30max"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=30, min_periods=2).max())
    .interpolate(method="bfill")
)


# 90
status_df["hour_90mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=90, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["hour_90std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=90, min_periods=2).std())
    .interpolate(method="bfill")
)

status_df["hour_90min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=90, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["hour_90max"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=90, min_periods=2).max())
    .interpolate(method="bfill")
)


# 180
status_df["hour_180mean"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=180, min_periods=2).mean())
    .interpolate(method="bfill")
)

status_df["hour_180std"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=180, min_periods=2).std())
    .interpolate(method="bfill")
)


status_df["hour_180min"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=180, min_periods=2).min())
    .interpolate(method="bfill")
)

status_df["hour_180max"] = (
    status_df.groupby(["station_id", "hour"])["bikes_available"]
    .transform(lambda x: x.rolling(window=180, min_periods=2).max())
    .interpolate(method="bfill")
)
# %%
status_df.tail(30)


# %%
# データセット作成
# 2014/09/01以前をtrain、以後をtestに分割(testはpredict = 1のものに絞る)
# =================================================


train = status_df  # [status_df["date"] < "2014-09-01"]

valid = status_df[(status_df["date"] >= "2014-09-01") & (status_df["predict"] == 0)]


test = status_df[(status_df["date"] >= "2014-09-01") & (status_df["predict"] == 1)]


# %%
# trainはbikes_availableがnanではないものに絞る
train = train[train["bikes_available"].notna()]
valid = valid[valid["bikes_available"].notna()]

# %%
train.columns

# %%
# 使用カラム

train_df = train[
    [
        "year",
        "month",
        "day",
        "hour",
        "station_id",
        "week_num",
        "dock_count",
        "lat",
        "long",
        "events",
        "precipitation",
        "count_start/dc",
        "count_end/dc",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "daytime",
        # "bikes_available_at0",
        "bikes_available",
        "dow_hour_5mean",
        "dow_hour_5std",
        "dow_hour_5min",
        "dow_hour_5max",
        "dow_hour_15mean",
        "dow_hour_15std",
        "dow_hour_15min",
        "dow_hour_15max",
        "dow_hour_30mean",
        "dow_hour_30std",
        "dow_hour_30min",
        "dow_hour_30max",
        "hour_7mean",
        "hour_7std",
        "hour_7min",
        "hour_7max",
        "hour_30mean",
        "hour_30std",
        "hour_30min",
        "hour_30max",
        "hour_90mean",
        "hour_90std",
        "hour_90min",
        "hour_90max",
        "hour_180mean",
        "hour_180std",
        "hour_180min",
        "hour_180max",
    ]
]

valid_df = valid[
    [
        "year",
        "month",
        "day",
        "hour",
        "station_id",
        "week_num",
        "dock_count",
        "lat",
        "long",
        "events",
        "precipitation",
        "count_start/dc",
        "count_end/dc",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "daytime",
        # "bikes_available_at0",
        "bikes_available",
        "dow_hour_5mean",
        "dow_hour_5std",
        "dow_hour_5min",
        "dow_hour_5max",
        "dow_hour_15mean",
        "dow_hour_15std",
        "dow_hour_15min",
        "dow_hour_15max",
        "dow_hour_30mean",
        "dow_hour_30std",
        "dow_hour_30min",
        "dow_hour_30max",
        "hour_7mean",
        "hour_7std",
        "hour_7min",
        "hour_7max",
        "hour_30mean",
        "hour_30std",
        "hour_30min",
        "hour_30max",
        "hour_90mean",
        "hour_90std",
        "hour_90min",
        "hour_90max",
        "hour_180mean",
        "hour_180std",
        "hour_180min",
        "hour_180max",
    ]
]


id_train = train[["yearmonth"]]


test_df = test[
    [
        "year",
        "month",
        "day",
        "hour",
        "station_id",
        "week_num",
        "dock_count",
        "lat",
        "long",
        "events",
        "precipitation",
        "count_start/dc",
        "count_end/dc",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "daytime",
        "dow_hour_5mean",
        "dow_hour_5std",
        "dow_hour_5min",
        "dow_hour_5max",
        "dow_hour_15mean",
        "dow_hour_15std",
        "dow_hour_15min",
        "dow_hour_15max",
        "dow_hour_30mean",
        "dow_hour_30std",
        "dow_hour_30min",
        "dow_hour_30max",
        "hour_7mean",
        "hour_7std",
        "hour_7min",
        "hour_7max",
        "hour_30mean",
        "hour_30std",
        "hour_30min",
        "hour_30max",
        "hour_90mean",
        "hour_90std",
        "hour_90min",
        "hour_90max",
        "hour_180mean",
        "hour_180std",
        "hour_180min",
        "hour_180max",
        # "bikes_available_at0",
    ]
]
print(train_df.shape, valid_df.shape, test_df.shape)

# %%
# データセット
x_train = train_df.drop("bikes_available", axis=1)
y_train = train["bikes_available"]

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
            with open(fname_lgb, "wb") as f:
                pickle.dump(model, f, protocol=4)

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

    return imp, metrics


# %%
# train
imp, metrics = train_lgb(
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
# imp
imp_sort = imp.sort_values("imp", ascending=False)
display(imp_sort)

# %%
# 重要度をcsvに出力
imp_sort.to_csv(os.path.join(OUTPUT_EXP, f"imp_{name}.csv"), index=False, header=False)

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
# 推論関数の定義 =================================================


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
# 推論処理
# =================================================
valid_x = valid_df.drop("bikes_available", axis=1)
valid_y = valid_df["bikes_available"]

id_valid = valid[["yearmonth"]]


# %%
valid_oof = predict_lgb(
    valid_x,
    valid_y,
    id_valid,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
valid_oof.head()

# %%


# %%
# ここまで
# =================================================

# %%
# 推論処理
# =================================================

id_test = test[["yearmonth"]]

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
test_pred = predict_sub_lgb(
    test_df,
    id_test,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
)


# %%
test_pred.head()


# %%
# submitファイルの出力
# =================================================
# sub_index = status_df[status_df["predict"] == 1].index

# df_submit = pd.DataFrame(list(zip(sub_index, test_pred)))


# print(df_submit.shape)
# display(df_submit.head())

# %%
test_pred.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.csv"),
    header=False,  # ,index=False
)


# %%
