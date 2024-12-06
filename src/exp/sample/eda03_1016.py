# %% [markdown]
## EDA3回目！
# =================================================
# ステーションをカラムにしていって傾向を見てみる：


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
import plotly.express as px


# lightGBM
import lightgbm as lgb

# sckit-learn
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)
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
serial_number = 21  # スプレッドシートAの番号

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
# 変換関数
# holiday
def holiday(df):
    # date yyyy-mm-dd
    df["holiday"] = np.zeros(len(df))
    for i in holiday_list:
        df["holiday"][df["date"] == i] = 1  # "祝日"


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


# leisure
def leisure(df):
    if df["workingday"] in [0] and df["hour"] in [11, 12, 13, 14, 15, 16, 17]:
        return 1  # "レジャー"
    else:
        return 0  # "その他"


# 季節
def month_to_season(month):
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
# date yyyy-mm-dd
status_df["date"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    # + " "
    # + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date"] = pd.to_datetime(status_df["date"])  # .dt.strftime("%Y-%m-%d")
# %%
# date_h yyyy-mm-dd hh
status_df["date_h"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
    + " "
    + status_df["hour"].astype(str).str.zfill(2)
)
status_df["date_h"] = pd.to_datetime(status_df["date_h"]).dt.strftime("%Y-%m-%d %H")

# %%
# dayofweek 曜日カラムの作成 Sun
status_df["dayofweek"] = status_df["date"].dt.strftime("%a")

# %%
# date_h_a yyyy-mm-dd hh(a)
status_df["date_h_a"] = (
    status_df["date_h"].astype(str) + " " + " (" + status_df["dayofweek"] + ")"
)

# %%
# yearmonth yyyy-mm データを分ける用
status_df["yearmonth"] = status_df["date"].astype(str).apply(lambda x: x[:7])

# %%
status_df.head()

# %%
# station読み込み
# =================================================
station_df = load_data(1)

# 日付に変更
station_df["installation_date"] = pd.to_datetime(station_df["installation_date"])

# %%
# weather読み込み
# =================================================
weather_df = load_data(4)

# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])

# %%
# trip読み込み
# =================================================
trip_df = load_data(3)


# time型変換
trip_df["start_date"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d %H")
trip_df["end_date"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d %H")


trip_df["start_date2"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m-%d")
trip_df["end_date2"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m-%d")

trip_df["start_ym"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y-%m")
trip_df["end_ym"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y-%m")

trip_df["start_month"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%m")
trip_df["end_month"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%m")


trip_df["start_year"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%Y")
trip_df["end_year"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%Y")

trip_df["start_hour"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%H")
trip_df["end_hour"] = pd.to_datetime(trip_df["end_date"]).dt.strftime("%H")

trip_df["dayofweek_st"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%a")
trip_df["dayofweek_end"] = pd.to_datetime(trip_df["start_date"]).dt.strftime("%a")

# %%
# station_id_df = pd.pivot_table(
#     data=status_df,
#     index="date_h_a",
#     columns="station_id",
#     values="bikes_available",
#     aggfunc="sum",
# )
# # グラフを表示
# px.line(station_id_df.iloc[:, :])

# %%
# stationについて理解
# sns.countplot(data=station_df, x="city")
# %%
# cityごとのstationナンバー
for i in station_df["city"].unique():
    print(i)
    display(station_df[station_df["city"] == i]["station_id"].unique())


# %%
# weatherについて理解
weather_df["events"].value_counts()

# %%
weather_df[weather_df["events"] == "Rain"][["precipitation", "cloud_cover"]].describe()


# %%
# tripとwetherを結合
weather_df["date"] = weather_df["date"].astype(str)
trip_df = pd.merge(
    trip_df,
    weather_df,
    left_on="start_date2",
    right_on="date",
    how="left",
)

trip_df.head()


# %%
trip_df.info()


# %%
# eventsをラベルエンコーディング
le_ = LabelEncoder()
trip_df["events"] = le_.fit_transform(trip_df["events"])
# %%
# 型変換
trip_df["events"] = trip_df["events"].astype(int)
trip_df["events"].value_counts()

# 3：なし、2：雨、１、きりと雨、０きり


# %%
# 天気が悪いかどうか
def events_to_events2(eve_number):
    if eve_number < 3:
        return 1
    else:
        return 0


trip_df["events2"] = trip_df["events"].apply(events_to_events2)

trip_df["events2"].value_counts()
# %%
# カウント用
trip_df["count"] = np.ones(len(trip_df))

# %%
# 天気が悪い日と、利用回数の関係 >天気が悪い日は利用が少ない傾向
a = trip_df.groupby("date")[["events2", "count"]].sum()
# グラフを表示
px.line(a)


# %%
# holiday(trip_df) 祝日なら1
trip_df["holiday"] = np.zeros(len(trip_df))
for i in holiday_list:
    trip_df["holiday"][trip_df["date"] == i] = 1  # "祝日"
# %%
# 関数でやるとなぜか0がnanになるので注意
trip_df["holiday"].value_counts()
# %%
# カラムの名前の変更
trip_df.rename(columns={"dayofweek_st": "dayofweek"}, inplace=True)


# %%
# 平日か土日祝か 平日1、土日祝0
trip_df["workingday"] = trip_df.apply(workingday, axis=1)
trip_df["workingday"].value_counts()

# %%
trip_df.columns
# %%
# 利用者数
a = trip_df.groupby("date")[["count"]].sum()
# 天気、祝日、平日など
b = trip_df.groupby("date")[["events2", "holiday", "workingday"]].mean() * 200
# %%
# グラフを表示
c = pd.merge(a, b, on="date")
px.line(c)

# %%
# ちゃんとholidayが反映されているか確認
trip_df[trip_df["holiday"] == 1]["workingday"].head(10)

# %%
# 降水量の理解
sns.histplot(trip_df[trip_df["events2"] > 0]["precipitation"])

# %%
# 雨降ている日の確認
trip_df[trip_df["precipitation"] > 0.02].groupby("date")["precipitation"].mean()

# %%
# 降水量と利用量の分析
preci = trip_df.groupby("date")[["precipitation"]].sum()
d = pd.merge(c, preci, on="date")

px.line(d)

# %%
# 0.3以上をバッドウェザーとする
trip_df["bad_weather"] = np.zeros(len(trip_df))


def bad_wether(amount):
    if amount > 0.3:
        return 1
    else:
        return 0


trip_df["bad_weather"] = trip_df["precipitation"].apply(bad_wether)
# %%
# 確認
trip_df[trip_df["bad_weather"] > 0]["date"].unique()
# %%
# season特徴量作成
trip_df["start_month"].astype(int)

trip_df["season"] = trip_df["start_month"].apply(month_to_season)

# %%
# 確認
trip_df.columns
# %%
# 確認
trip_df.head()
# %%
# サブスクカスタマーの理解
trip_df["subscription_type"].value_counts()

# %%
# カスタマーが土日祝に利用する時の傾向を探る
trip_df[
    (trip_df["workingday"] == 0) & (trip_df["subscription_type"] == "Customer")
].describe().T
# %%
# よく利用されている特徴量を作成
trip_df["comfort"] = np.zeros(len(trip_df))


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


trip_df["comfort"] = trip_df.apply(comforty, axis=1)
display(trip_df["comfort"].value_counts())
# %%
# # 各ステーション毎に、欠損値を後の値で埋める
# train_dataset_df_new = pd.DataFrame()
# for station_id in train_dataset_df["station_id"].unique().tolist():
#     temp_df = train_dataset_df[train_dataset_df["station_id"] == station_id]
#     temp_df["bikes_available"] = (
#         temp_df["bikes_available"]
#         .astype("object")
#         .fillna(method="backfill")
#         .astype("float")
#     )
#     train_dataset_df_new = pd.concat([train_dataset_df_new, temp_df])

# print(train_dataset_df_new.isnull().sum())
# %%
trip_df.columns
# %%
temp_a_df = trip_df[trip_df["start_station_id"] == 32]
# temp_b_df = temp_a_df[temp_a_df["workingday"] == 1]
# %%
temp_c_df = (
    temp_a_df.pivot_table(
        index="start_hour",
        columns="dayofweek_st",
        values="count",
        aggfunc="sum",
        # margins=True,
        # margins_name="Total",
    )
    / 52
)
# %%
temp_c_df.fillna(0, inplace=True)
temp_c_df

# %%
dow_ = temp_c_df[["Fri"]]

# %%
dow_["shift1"] = dow_.shift(1).interpolate(method="bfill")
dow_
# dow_.drop("shit1",axis=1,inplace=True)

# %%
dow_["cumsum"] = dow_["Fri"].cumsum()
dow_
# dow_.drop("Fri_cumsum",axis=1,inplace=True)
# %%
dow_.reset_index()
# %%
melt = pd.melt(
    dow_.reset_index(),
    id_vars=["start_hour", "shift1", "cumsum"],
    value_vars="Fri",
    var_name="dayofweek",
)
melt
# %%
melt.shape
# %%
# np.sort(trip_df["start_station_id"].unique())
# #%%
# for i in np.sort(trip_df["start_station_id"].unique()):
#     print(i)
for j in trip_df["dayofweek"].unique():
    print(j)


# %%
# forを使って一気に
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
sum_cumsum_start = shift_cumsum1(trip_df, "start_station_id", "start_hour")

sum_cumsum_start.head()
# %%
sum_cumsum_end = shift_cumsum1(trip_df, "end_station_id", "end_hour")
sum_cumsum_end.head()
# %%
sum_cumsum_start.tail()

# %%
sum_cumsum_start["station_id"].value_counts()
# %%
# staion_idごとの利用率
sns.countplot(trip_df, x="start_station_id")
# %%
sns.countplot(trip_df, x="end_station_id")

# %%
trip_df["end_station_id"].value_counts() / len(trip_df)
# %%
trip_df["start_station_id"].value_counts() / len(trip_df)

# %%
# 全体に対しての利用割合
(
    (trip_df["start_station_id"].value_counts().sort_index() / len(trip_df))
    + (trip_df["end_station_id"].value_counts().sort_index() / len(trip_df))
) / 2
# %%
