# %% [markdown]
## モデルチューニング
# =================================================
# モデルチューニング（テンソーフロー）を使う

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
import zipfile

import numpy as np
import pandas as pd


# import matplotlib.pyplot as plt
import seaborn as sns
# import japanize_matplotlib


# lightGBM
# import lightgbm as lgb

# sckit-learn
# from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.metrics import mean_absolute_error

# 8-49:ライブラリのインポート
from sklearn.preprocessing import LabelEncoder

# import tensorflow
import tensorflow as tf

# import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization,
    Activation,
    Concatenate,
)  # type:ignore
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
)  # type:ignore
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Embedding, Flatten

# %%Config
# Config
# =================================================

######################
# serial #
######################
serial_number = 3  # スプレッドシートAの番号


######################
# Data #
######################
comp_name = "mlb-player-digital-engagement-forecasting"
# 評価：mean_absolute_error

skip_run = False  # 飛ばす->True，飛ばさない->False

# name = 'exp002_model_tenso.py' #ファイル名は適宜変更すること いらなかったら消す

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
# target_columns = 'TARGET'
# sub_index = 'SK_ID_CURR'

######################
# ハイパーパラメータの設定
######################
# lgbm
# params={
# 	'boosting_type':'gbdt',
# 	'objective':'regression_l1',
# 	'metric':'mean_absolute_error',
# 	'learning_rate':0.05,
# 	'num_leaves':32,
# 	'subsample':0.7,
# 	'subsample_freq':1,
# 	'feature_fraction':0.8,
# 	'min_data_in_leaf':50,
# 	'min_sum_hessian_in_leaf':50,
# 	'n_estimators':1000,
# 	'random_state':123,
# 	'importance_type':'gain',
# 	}


# %% Utilities
# =================================================
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


# %%
# メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype
        #  col_type != object:
        # not isinstance(col_type, object)
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


# %%
# 8-4:train_updated.csv専用の変換関数の作成
def unpack_json(json_str):
    return np.nan if pd.isna(json_str) else pd.read_json(json_str)


def extract_data(input_df, col="events", show=False):
    output_df = pd.DataFrame()
    for i in np.arange(len(input_df)):
        if show:
            print(f"\r{i + 1}/{len(input_df)}", end="")
        try:
            output_df = pd.concat(
                [output_df, unpack_json(input_df[col].iloc[i])],
                axis=0,
                ignore_index=True,
            )
        except Exception as e:
            # 全ての例外に対する処理
            pass
            print(f"エラーが発生しました: {e}")
    if show:
        print("")
    if show:
        print(output_df.shape)
    if show:
        display(output_df.head())
    return output_df


# %%

# 8-13:学習データと検証データの期間の設定
list_cv_month = [
    [
        [
            "2020-05",
            "2020-06",
            "2020-07",
            "2020-08",
            "2020-09",
            "2020-10",
            "2020-11",
            "2020-12",
            "2021-01",
            "2021-02",
            "2021-03",
            "2021-04",
        ],
        ["2021-05"],
    ],
    [
        [
            "2020-06",
            "2020-07",
            "2020-08",
            "2020-09",
            "2020-10",
            "2020-11",
            "2020-12",
            "2021-01",
            "2021-02",
            "2021-03",
            "2021-04",
            "2021-05",
        ],
        ["2021-06"],
    ],
    [
        [
            "2020-07",
            "2020-08",
            "2020-09",
            "2020-10",
            "2020-11",
            "2020-12",
            "2021-01",
            "2021-02",
            "2021-03",
            "2021-04",
            "2021-05",
            "2021-06",
            "2021-07",
        ],
        ["2021-07"],
    ],
]
# %%lgbm
# 8-22:学習用関数の作成


# def train_lgb(
#     input_x,
#     input_y,
#     input_id,
#     params,
#     list_nfold=[0, 1, 2],
#     mode_train="train",
# ):
#     # 推論値を格納する変数の作成
#     df_valid_pred = pd.DataFrame()
#     # 評価値を入れる変数の作成
#     metrics = []
#     # 重要度を格納するデータフレームの作成
#     df_imp = pd.DataFrame()

#     # validation
#     cv = []
#     for month_tr, month_va in list_cv_month:
#         cv.append(
#             [
#                 input_id.index[input_id["yearmonth"].isin(month_tr)],
#                 input_id.index[
#                     input_id["yearmonth"].isin(month_va)
#                     & (input_id["playerForTestSetAndFuturePreds"] == 1)
#                 ],
#             ]
#         )

#     # モデル学習(target/foldごとに学習)
#     for nfold in list_nfold:
#         for i, target in enumerate(["target1", "target2", "target3", "target4"]):
#             print("-" * 20, target, ",fold", nfold, "-" * 20)
#             # tainとvalidに分離
#             idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
#             x_tr, y_tr, id_tr = (
#                 input_x.loc[idx_tr, :],
#                 input_y.loc[idx_tr, target],
#                 input_id.loc[idx_tr, :],
#             )
#             x_va, y_va, id_va = (
#                 input_x.loc[idx_va, :],
#                 input_y.loc[idx_va, target],
#                 input_id.loc[idx_va, :],
#             )
#             print(x_tr.shape, y_tr.shape, id_tr.shape)
#             print(x_va.shape, y_va.shape, id_va.shape)

#             # 保存するモデルのファイル名
#             filepath = os.path.join(EXP_MODEL, f"model_lgb_{target}_fold{nfold}.h5")

#             if not os.file.isfile(filepath):
#                 # if mode_train == 'train':
#                 print("trainning start!")
#                 model = lgb.LGBMRegressor(**params)
#                 model.fit(
#                     x_tr,
#                     y_tr,
#                     eval_set=[(x_tr, y_tr), (x_va, y_va)],
#                     callbacks=[
#                         lgb.early_stopping(stopping_rounds=50, verbose=True),
#                         lgb.log_evaluation(100),
#                     ],
#                 )
#                 with open(filepath, "wb") as f:
#                     pickle.dump(model, f, protocol=4)
#             else:
#                 print("model load.")
#                 with open(filepath, "rb") as f:
#                     model = pickle.load(f)
#                 print("Done")

#             # validの推論値取得
#             y_va_pred = model.predict(x_va)
#             tmp_pred = pd.concat(
#                 [
#                     id_va,
#                     pd.DataFrame(
#                         {
#                             "target": target,
#                             "nfold": nfold,
#                             "true": y_va,
#                             "pred": y_va_pred,
#                         }
#                     ),
#                 ],
#                 axis=1,
#             )
#             df_valid_pred = pd.concat(
#                 [df_valid_pred, tmp_pred], axis=0, ignore_index=True
#             )

#             # 評価値の算出
#             metric_va = mean_absolute_error(y_va, y_va_pred)
#             metrics.append([target, nfold, metric_va])

#             # 重要度の取得
#             tmp_imp = pd.DataFrame(
#                 {
#                     "col": x_tr.columns,
#                     "imp": model.feature_importances_,
#                     "target": "target1",
#                     "nfold": nfold,
#                 }
#             )
#             df_imp = pd.concat([df_imp, tmp_imp], axis=0, ignore_index=True)

#     print("-" * 10, "result", "-" * 10)
#     # 評価値
#     df_metrics = pd.DataFrame(metrics, columns=["target", "nfold", "mae"])
#     print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

#     # validの推論値
#     df_valid_pred_all = pd.pivot_table(
#         df_valid_pred,
#         index=[
#             "engagementMetricsDate",
#             "playerId",
#             "date_playerId",
#             "date",
#             "yearmonth",
#             "playerForTestSetAndFuturePreds",
#         ],
#         columns=["target", "nfold"],
#         values=["true", "pred"],
#         aggfunc=np.sum,
#     )
#     df_valid_pred_all.columns = [
#         "{}_fold{}_{}".format(j, k, i) for i, j, k in df_valid_pred_all.columns
#     ]
#     df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

#     print("-" * 20, "importance", "-" * 20)
#     print(
#         df_imp.groupby(["col"])["imp"]
#         .agg(["mean", "std"])
#         .sort_values("mean", ascending=False)[:10]
#     )

#     # stdout と stderr を一時的にリダイレクト
#     stdout_logger = logging.getLogger("STDOUT")
#     stderr_logger = logging.getLogger("STDERR")

#     sys_stdout_backup = sys.stdout
#     sys_stderr_backup = sys.stderr

#     sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
#     sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

#     print("-" * 10, "result", "-" * 10)
#     # 評価値
#     df_metrics = pd.DataFrame(metrics, columns=["target", "nfold", "mae"])
#     print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

#     # validの推論値
#     df_valid_pred_all = pd.pivot_table(
#         df_valid_pred,
#         index=[
#             "engagementMetricsDate",
#             "playerId",
#             "date_playerId",
#             "date",
#             "yearmonth",
#             "playerForTestSetAndFuturePreds",
#         ],
#         columns=["target", "nfold"],
#         values=["true", "pred"],
#         aggfunc=np.sum,
#     )
#     df_valid_pred_all.columns = [
#         "{}_fold{}_{}".format(j, k, i) for i, j, k in df_valid_pred_all.columns
#     ]
#     df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

#     print("-" * 20, "importance", "-" * 20)
#     print(
#         df_imp.groupby(["col"])["imp"]
#         .agg(["mean", "std"])
#         .sort_values("mean", ascending=False)[:10]
#     )

#     # リダイレクトを解除
#     sys.stdout = sys_stdout_backup
#     sys.stderr = sys_stderr_backup

#     return df_valid_pred_all, df_metrics, df_imp


# %%
# 8-29:推論用データセット作成の関数


# def makedataset_for_predict(input_test, input_prediction):
#     test = input_test.copy()
#     prediction = input_prediction.copy()

#     # dateを日付型に変換
#     prediction["date"] = pd.to_datetime(prediction["date"], format="%Y%m%d")

#     # engagementMetricsDateplayerIdのカラムを作成
#     prediction["engagementMetricsDate"] = prediction["date_playerId"].apply(
#         lambda x: x[:8]
#     )
#     prediction["engagementMetricsDate"] = pd.to_datetime(
#         prediction["engagementMetricsDate"], format="%Y%m%d"
#     )
#     prediction["playerId"] = prediction["date_playerId"].apply(lambda x: int(x[9:]))

#     # 日付から曜日と年月を作成
#     prediction["dayofweek"] = prediction["date"].dt.dayofweek
#     prediction["yearmonth"] = prediction["date"].astype(str).apply(lambda x: x[:7])

#     # dateカラムの作成・加工
#     df_rosters = extract_data(test, col="rosters", show=True)
#     df_rosters = df_rosters.rename(columns={"gameDate": "date"})
#     df_rosters["date"] = pd.to_datetime(df_rosters["date"], format="%Y-%m-%d")

#     # テーブルの結合
#     df_test = pd.merge(prediction, df_players, on=["playerId"], how="left")
#     df_test = pd.merge(df_test, df_rosters, on=["date", "playerId"], how="left")
#     df_test = pd.merge(df_test, df_agg_target, on=["playerId", "yearmonth"], how="left")

#     # 説明関数の作成
#     x_test = df_test[
#         [
#             "playerId",
#             "dayofweek",
#             "birthCity",
#             "birthStateProvince",
#             "birthCountry",
#             "heightInches",
#             "weight",
#             "primaryPositionCode",
#             "primaryPositionName",
#             "playerForTestSetAndFuturePreds",
#         ]
#         + col_rosters
#         + col_agg_target
#     ]

#     id_test = df_test[
#         [
#             "engagementMetricsDate",
#             "playerId",
#             "date_playerId",
#             "date",
#             "yearmonth",
#             "playerForTestSetAndFuturePreds",
#         ]
#     ]

#     # カテゴリ変数をcategory型に変換
#     data_pre00(x_test)

#     return x_test, id_test


# %%
# 8-34:推論値処理の関数
# def predict_lgb(
#     input_test,
#     input_id,
#     list_nfold=[0, 1, 2],
# ):
#     df_test_pred = input_id.copy()

#     for target in ["target1", "target2", "target3", "target4"]:
#         for nfold in list_nfold:
#             # modelのロード
#             filepath = os.path.join(EXP_MODEL, f"model_lgb_{target}_fold{nfold}.h5")
#             with open(filepath, "rb") as f:
#                 model = pickle.load(f)
#                 # 推論
#                 pred = model.predict(input_test)
#                 # 予測値の格納
#                 df_test_pred[f"{target}_{nfold}"] = pred
#     # 推論値の取得：各foldの平均値
#     for target in ["target1", "target2", "target3", "target4"]:
#         df_test_pred[target] = df_test_pred[
#             df_test_pred.columns[df_test_pred.columns.str.contains(target)]
#         ].mean(axis=1)

#     return df_test_pred


# %% ファイル


# ====================
# 0 awards.csv
# ====================
# 1 example_sample_submission.csv
# ====================
# 2 example_test.csv
# ====================
# 3 players.csv
# ====================
# 4 seasons.csv
# ====================
# 5 teams.csv
# ====================
# 6 train.csv
# ====================
# 7 train_0.csv
# ====================
# 8 train_1.csv
# ====================
# 9 train_2.csv
# ====================
# 10 train_down.pkl
# ====================
# 11 train_updated.csv
# ====================
# 0 competition.cpython-37m-x86_64-linux-gnu.so
# ====================
# 1 __init__.py


# %% [markdown]
##Main 分析start!
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
# 8-2:ファイルの読み込み
# ファイルの読み込み train_updated
# =================================================
# train = pd.read_csv(INPUT_PATH+"train_updated.csv")
# print('train_updated:app_train')
# print(train.shape)
# display(train.head())

# #おそらくファイルサイズが大きくて読み込めないので、分割して読み込む
# # 読み込みたい大きなCSVファイルのパス
# input_file = INPUT_PATH+'train_updated.csv'
# # 出力するファイルの名前のベース
# output_file_base = 'train_'
# # 一度に読み込む行数（ここでは500行ごとに分割）
# chunk_size = 500
# # チャンクごとにファイルを分割
# for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
# # 出力ファイル名
# 	output_file = f'{output_file_base}{i}.csv'
# 	# 分割したデータを保存
# 	chunk.to_csv(output_file, index=False)
# 	print(f'{output_file} が作成されました。')

# %%
# 8-2:ファイル読み込み続き、結合
#
# train_0 = pd.read_csv(INPUT_PATH + 'train_0.csv')
# print(train_0.shape)
# display(train_0.head())

# train_1 = pd.read_csv(INPUT_PATH + 'train_1.csv')
# print(train_1.shape)
# display(train_1.head())

# train_2 = pd.read_csv(INPUT_PATH + 'train_2.csv')
# print(train_2.shape)
# display(train_2.head())

# train = pd.concat([train_0, train_1, train_2])

# %%
# 8-3:データの絞り込み
# train = train.loc[train['date']>=20200401, :].reset_index(drop=True)
# print(train.shape)

# train.to_csv('train_down.csv',index=None)
# train.to_pickle(INPUT_PATH + 'train_down.pkl')

# %%ファイル読み込み
# train = pd.read_csv(INPUT_PATH + 'train_down.csv')
# print(train.shape)
# display(train.head())

# train = load_data(10)
train = pd.read_pickle(INPUT_PATH + "train_down.pkl")


# %%
# 8-5train_updated.csvから「nextDayPlayerEngagement」を取り出して表形式に変換
df_engagement = extract_data(train, col="nextDayPlayerEngagement", show=True)

# %%
# 8-6:結合キーであるdat_playIdの作成
df_engagement["date_playerId"] = (
    df_engagement["engagementMetricsDate"].str.replace("-", "")
    + "_"
    + df_engagement["playerId"].astype(str)
)
df_engagement.tail()
# %%
# 8-7:日付から簡単な特徴量を作成
# 推論実施日のカラム作成（推論実施日＝推論対象日の前日）
df_engagement["date"] = pd.to_datetime(
    df_engagement["engagementMetricsDate"], format="%Y-%m-%d"
) + dt.timedelta(days=-1)

# 推論実施日から「曜日」と「年月」の特徴量作成
df_engagement["dayofweek"] = df_engagement["date"].dt.dayofweek
df_engagement["yearmonth"] = df_engagement["date"].astype(str).apply(lambda x: x[:7])
df_engagement.head()

# %%
# 8-8: players.csvの読み込み
df_players = load_data(3)
# pd.read_csv(INPUT_PATH + "players.csv")
# print(df_players.shape)
df_players["playerForTestSetAndFuturePreds"] = np.where(
    df_players["playerForTestSetAndFuturePreds"] == True,  # noqa: E712
    1,
    0,
)
# %%

df_players.head()


# %%
# 8-38:train_updated.csvからrostersカラムのデータ取り出し
df_rosters = extract_data(train, col="rosters", show=True)


# %%
# 8-39:rostersのデータ前処理加工
df_rosters = df_rosters.rename(columns={"gameDate": "date"})
df_rosters["date"] = pd.to_datetime(df_rosters["date"], format="%Y-%m-%d")

# 追加するカラムリストの作成（dateとplayerIdは結合キー）
col_rosters = ["teamId", "statusCode", "status"]

df_rosters.head()
# %%

# テーブル結合
df_train = pd.merge(df_engagement, df_players, on="playerId", how="left")
print(df_train.shape)
df_train.head()

# %%

# 8-40:targetの特徴量の計算
df_agg_target = df_train.groupby(["yearmonth", "playerId"])[
    ["target1", "target2", "target3", "target4"]
].agg(["mean", "median", "std", "min", "max"])

df_agg_target.columns = [f"{i}_{j}" for i, j in df_agg_target.columns]
df_agg_target = df_agg_target.reset_index(drop=False)
df_agg_target.head()

# %%
# 8-41:ラグ特徴量の作成

# 年月でソート（時系列順に並んでいないとシフト時におかしくなるので）
df_agg_target = df_agg_target.sort_values("yearmonth").reset_index(drop=True)
# yearmonthを1ヶ月シフトして過去にする
df_agg_target["yearmonth"] = df_agg_target.groupby(["playerId"])["yearmonth"].shift(-1)
# yearmonthの欠損値を[2021-08]で埋める
df_agg_target["yearmonth"] = df_agg_target["yearmonth"].fillna("2021-08")
# 集計値がラグ特徴量と分かるように名称を変更
df_agg_target.columns = [
    col + "_lag1month" if col not in ["playerId", "yearmonth"] else col
    for col in df_agg_target.columns
]
# 追加したカラムリスト作成
col_agg_target = list(
    df_agg_target.columns[df_agg_target.columns.str.contains("lag1month")]
)

df_agg_target.head()


# %%
# 8-42 学習用データセット作成
# データ結合
df_train = pd.merge(df_engagement, df_players, on="playerId", how="left")
print(df_train.shape)
df_train = pd.merge(df_train, df_rosters, on=["date", "playerId"], how="left")

df_train = pd.merge(df_train, df_agg_target, on=["playerId", "yearmonth"], how="left")
# %%[markdown]
# ↑↑これまでの処理
# ==========================================================


# %%[markdown]
# 今回の実験（モデルチューニング：ニューラルネットワーク）
# ==========================================================
# %%
# 8-48:目的変数間の相関係数の算出
df_engagement[["target1", "target2", "target3", "target4"]].corr()
# %%


# %%
# 8-xx:再現性のためのシート指定
def seed_everything(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    # session_conf = tf.compat.v1.ConfigProto(
    #     intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    # )
    # sess = tf.compat.v1.Session(
    #     graph=tf.compat.v1.get_default_graph(), config=session_conf
    # )
    # tf.compat.v1.keras.backend.set_session(sess)


# %%8.5.1 データセット作成
# 8-50:学習セットの作成
x_train = df_train[
    [
        "playerId",
        "dayofweek",
        "birthCity",
        "birthStateProvince",
        "birthCountry",
        "heightInches",
        "weight",
        "primaryPositionCode",
        "primaryPositionName",
        "playerForTestSetAndFuturePreds",
    ]
    + col_rosters
    + col_agg_target
]

y_train = df_train[["target1", "target2", "target3", "target4"]]

id_train = df_train[
    [
        "engagementMetricsDate",
        "playerId",
        "date_playerId",
        "date",
        "yearmonth",
        "playerForTestSetAndFuturePreds",
    ]
]

print(x_train.shape, y_train.shape, id_train.shape)
# %%
x_train.columns
# %%
# 8-51: 数値とカテゴリ変数のカラムリストを作成
col_num = ["heightInches", "weight", "playerForTestSetAndFuturePreds"] + col_agg_target
col_cat = [
    "playerId",
    "dayofweek",
    "birthCity",
    "birthStateProvince",
    "birthCountry",
    "primaryPositionCode",
    "primaryPositionName",
] + col_rosters
print(len(col_num), len(col_cat))
# %%
# 8-52: 数値データの欠損値補間・数値化
dict_num = {}
for col in col_num:
    print(col)
    #     # 欠損値補間：平均値で埋める
    #     value_fillna = x_train[col].mean()
    # 欠損値補間：0で埋める
    value_fillna = 0
    x_train[col] = x_train[col].fillna(value_fillna)

    # 正規化（0～1になるように変換）
    value_min = x_train[col].min()
    value_max = x_train[col].max()
    x_train[col] = (x_train[col] - value_min) / (value_max - value_min)

    # testデータにも適用できるように保存
    dict_num[col] = {}
    dict_num[col]["fillna"] = value_fillna
    dict_num[col]["min"] = value_min
    dict_num[col]["max"] = value_max

print("Done.")
# %%
# 8-53: カテゴリ変数の欠損値補間・数値化
dict_cat = {}
for col in col_cat:
    print(col)
    # 欠損値補間：unknownで埋める
    value_fillna = "unknown"
    x_train[col] = x_train[col].fillna(value_fillna)

    # str型に変換
    x_train[col] = x_train[col].astype(str)

    # ラベルエンコーダー：0からはじまる整数に変換
    le = LabelEncoder()
    le.fit(x_train[col])
    # 推論時に未知の値があっても対応できるように未知ラベル(unknown)を用意。
    list_label = sorted(list(set(le.classes_) | set(["unknown"])))
    map_label = {j: i for i, j in enumerate(list_label)}
    x_train[col] = x_train[col].map(map_label)

    # testデータにも適用できるように保存
    dict_cat[col] = {}
    dict_cat[col]["fillna"] = value_fillna
    dict_cat[col]["map_label"] = map_label
    dict_cat[col]["num_label"] = len(list_label)

print("Done.")


# %%
# 8-54: 欠損値補間・正規化の関数化（推論用）
def transform_data(input_x):
    output_x = input_x.copy()

    # 数値データの欠損値補間・正規化
    for col in col_num:
        # 欠損値補間：平均値で埋める
        value_fillna = dict_num[col]["fillna"]
        output_x[col] = output_x[col].fillna(value_fillna)

        # 正規化（0~1になるように変換）
        value_min = dict_num[col]["min"]
        value_max = dict_num[col]["max"]
        output_x[col] = (output_x[col] - value_min) / (value_max - value_min)

    for col in col_cat:
        print(col)
        # 欠損値補間：unknownで埋める
        value_fillna = dict_cat[col]["unknown"]
        output_x[col] = output_x[col].fillna(value_fillna)

        # str型に変換
        output_x[col] = output_x[col].astype(str)

        # 推論時に未知の値があっても対応できるように未知ラベル(unknown)を用意。
        map_label = dict_cat[col]["map_label"]
        output_x[col] = output_x[col].map(map_label)

        # 対応するものがない場合はunknownのラベルで埋める
        output_x[col] = output_x[col].fillna(map_label["unknown"])

        return output_x


# %%8.5.2 モデル学習
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


# %%
# 8-56:モデル構造の確認
create_model(col_num=col_num, col_cat=col_cat, show=True)

# %%
# 8-57:学習用の関数をニューラルネットワーク用にカスタマイズ


def train_tf(
    input_x,
    input_y,
    input_id,
    list_nfold=[0, 1, 2],
    mode_train="train",
    batch_size=1024,
    epochs=100,
):
    # 推論値を格納する変数の作成
    df_valid_pred = pd.DataFrame()
    # 評価値を入れる変数の作成
    metrics = []

    # validation
    cv = []
    for month_tr, month_va in list_cv_month:
        cv.append(
            [
                input_id.index[input_id["yearmonth"].isin(month_tr)],
                input_id.index[
                    input_id["yearmonth"].isin(month_va)
                    & (input_id["playerForTestSetAndFuturePreds"] == 1)
                ],
            ]
        )

    # モデル学習(target/foldごとに学習)
    for nfold in list_nfold:
        print("-" * 20, "fold", nfold, "-" * 20)
        # tainとvalidに分離
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

        x_num_tr, x_cat_tr, y_tr = (
            input_x.loc[idx_tr, col_num].values,
            input_x.loc[idx_tr, col_cat].values,
            input_y.loc[idx_tr, :].values,
        )
        x_num_va, x_cat_va, y_va, id_va = (
            input_x.loc[idx_va, col_num].values,
            input_x.loc[idx_va, col_cat].values,
            input_y.loc[idx_va, :].values,
            input_id.loc[idx_va, :],
        )
        print(x_num_tr.shape, x_cat_tr.shape, y_tr.shape)
        print(x_num_va.shape, x_cat_va.shape, y_va.shape)

        # 保存するモデルのファイル名
        filepath = os.path.join(EXP_MODEL, f"model_tf_fold{nfold}.weights.h5")

        if not os.path.isfile(filepath):
            # if mode_train == 'train':
            print("trainning start!")
            seed_everything(seed=123)
            model = create_model(col_num=col_num, col_cat=col_cat, show=False)
            model.fit(
                x=[x_num_tr, x_cat_tr],
                y=y_tr,
                validation_data=(
                    [x_num_va, x_cat_va],
                    y_va,
                ),
                batch_size=batch_size,
                epochs=epochs,
                callbacks=[
                    ModelCheckpoint(
                        filepath=filepath,
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
                        monitor="valloss", mode="min", factor=0.1, patience=5, verbose=1
                    ),
                ],
                verbose=1,
            )

        else:
            print("model load.")
            model = create_model(col_num=col_num, col_cat=col_cat, show=False)
            model.load_weights(filepath)

        # validの推論値取得
        y_va_pred = model.predict([x_num_va, x_cat_va])
        tmp_pred = pd.concat(
            [
                id_va,
                pd.DataFrame(
                    y_va,
                    columns=[
                        "target1_true",
                        "target2_true",
                        "target3_true",
                        "target4_true",
                    ],
                ),
                pd.DataFrame(
                    y_va_pred,
                    columns=[
                        "target1_pred",
                        "target2_pred",
                        "target3_pred",
                        "target4_pred",
                    ],
                ),
            ],
            axis=1,
        )
        tmp_pred["nfold"] = nfold
        df_valid_pred = pd.concat([df_valid_pred, tmp_pred], axis=0, ignore_index=True)

        # 評価値の算出
        metrics.append(
            ["target1", nfold, np.mean(np.abs(y_va[:, 0] - y_va_pred[:, 0]))]
        )
        metrics.append(
            ["target2", nfold, np.mean(np.abs(y_va[:, 1] - y_va_pred[:, 1]))]
        )
        metrics.append(
            ["target3", nfold, np.mean(np.abs(y_va[:, 2] - y_va_pred[:, 2]))]
        )
        metrics.append(
            ["target4", nfold, np.mean(np.abs(y_va[:, 3] - y_va_pred[:, 3]))]
        )

    print("-" * 10, "result", "-" * 10)
    # 評価値
    df_metrics = pd.DataFrame(metrics, columns=["target", "nfold", "mae"])
    print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

    # validの推論値
    df_valid_pred_all = pd.pivot_table(
        df_valid_pred,
        index=[
            "engagementMetricsDate",
            "playerId",
            "date_playerId",
            "date",
            "yearmonth",
            "playerForTestSetAndFuturePreds",
        ],
        columns=["nfold"],
        values=list(
            df_valid_pred.columns[df_valid_pred.columns.str.contains("target")]
        ),
        aggfunc=np.sum,
    )
    df_valid_pred_all.columns = [
        "{}_fold{}_{}".format(i.split("_")[0], j, i.split("_")[1])
        for i, j in df_valid_pred_all.columns
    ]
    df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

    # stdout と stderr を一時的にリダイレクト
    stdout_logger = logging.getLogger("STDOUT")
    stderr_logger = logging.getLogger("STDERR")

    sys_stdout_backup = sys.stdout
    sys_stderr_backup = sys.stderr

    sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
    sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)

    print("-" * 10, "result", "-" * 10)
    # 評価値
    df_metrics = pd.DataFrame(metrics, columns=["target", "nfold", "mae"])
    print(f'MCMAE:{df_metrics["mae"].mean():.4f}')

    # validの推論値
    df_valid_pred_all = pd.pivot_table(
        df_valid_pred,
        index=[
            "engagementMetricsDate",
            "playerId",
            "date_playerId",
            "date",
            "yearmonth",
            "playerForTestSetAndFuturePreds",
        ],
        columns=["nfold"],
        values=list(
            df_valid_pred.columns[df_valid_pred.columns.str.contains("target")]
        ),
        aggfunc=np.sum,
    )
    df_valid_pred_all.columns = [
        "{}_fold{}_{}".format(i.split("_")[0], j, i.split("_")[1])
        for i, j in df_valid_pred_all.columns
    ]
    df_valid_pred_all = df_valid_pred_all.reset_index(drop=False)

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return df_valid_pred_all, df_metrics


# %%
# 8-58: 学習の実行
df_valid_pred, df_metrics = train_tf(
    x_train,
    y_train,
    id_train,
    list_nfold=[0, 1, 2],
    mode_train="train",
    batch_size=1024,
    epochs=1000,
)
# %%
# 8-59: 評価値の確認
print(f"MCMAE:{df_metrics['mae'].mean():.4f}")
display(
    pd.pivot_table(
        df_metrics,
        index="nfold",
        columns="target",
        values="mae",
        aggfunc=np.mean,
        margins=True,
    )
)
# %%8.5.3 モデル推論
# 8-60:データセットの作成関数をニューラルネットワーク用


def makedataset_for_predict(input_test, input_prediction):
    test = input_test.copy()
    prediction = input_prediction.copy()

    # dateを日付型に変換
    prediction["date"] = pd.to_datetime(prediction["date"], format="%Y%m%d")

    # engagementMetricsDateplayerIdのカラムを作成
    prediction["engagementMetricsDate"] = prediction["date_playerId"].apply(
        lambda x: x[:8]
    )
    prediction["engagementMetricsDate"] = pd.to_datetime(
        prediction["engagementMetricsDate"], format="%Y%m%d"
    )
    prediction["playerId"] = prediction["date_playerId"].apply(lambda x: int(x[9:]))

    # 日付から曜日と年月を作成
    prediction["dayofweek"] = prediction["date"].dt.dayofweek
    prediction["yearmonth"] = prediction["date"].astype(str).apply(lambda x: x[:7])

    # dateカラムの作成・加工
    df_rosters = extract_data(test, col="rosters")
    df_rosters = df_rosters.rename(columns={"gameDate": "date"})
    df_rosters["date"] = pd.to_datetime(df_rosters["date"], format="%Y-%m-%d")

    # テーブルの結合
    df_test = pd.merge(prediction, df_players, on=["playerId"], how="left")
    df_test = pd.merge(df_test, df_rosters, on=["date", "playerId"], how="left")
    df_test = pd.merge(df_test, df_agg_target, on=["playerId", "yearmonth"], how="left")

    # 説明関数の作成
    x_test = df_test[
        [
            "playerId",
            "dayofweek",
            "birthCity",
            "birthStateProvince",
            "birthCountry",
            "heightInches",
            "weight",
            "primaryPositionCode",
            "primaryPositionName",
            "playerForTestSetAndFuturePreds",
        ]
        + col_rosters
        + col_agg_target
    ]

    id_test = df_test[
        [
            "engagementMetricsDate",
            "playerId",
            "date_playerId",
            "date",
            "yearmonth",
            "playerForTestSetAndFuturePreds",
        ]
    ]

    # カテゴリ変数をcategory型に変換
    # data_pre00(x_test)

    return x_test, id_test


# %%
# 8-61: 推論用関数をニューラルネットワーク用
def predict_tf(
    input_x,
    input_id,
    list_nfold=[0, 1, 2],
):
    # 推論値を入れる変数の作成
    test_pred = np.zeros((len(input_x), 4))

    # 数値とカテゴリ変数に分離
    x_num_test, x_cat_test = input_x[col_num], input_x[col_cat]

    for nfold in list_nfold:
        # modelのロード
        filepath = os.path.join(EXP_MODEL, f"model_tf_fold{nfold}.weights.h5")
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


# %%
