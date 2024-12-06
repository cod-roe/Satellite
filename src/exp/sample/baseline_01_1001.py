# %% [markdown]
## ベースラインチュートリアル第2弾！
# =================================================
# LSTMを使う


# %%
# ライブラリ読み込み
# =================================================
import datetime as dt
from datetime import timedelta

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

# 次元圧縮
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# lightGBM
# import lightgbm as lgb

# lightGBM精度測定
# import shap

# パラメータチューニング
# import optuna

# from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
# from scipy.spatial.distance import squareform
# from scipy.stats import spearmanr

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.optimizers import Adam


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 1  # スプレッドシートAの番号

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


# %%
# make dirs
# =================================================
def make_dirs():
    for d in [EXP_MODEL]:
        os.makedirs(d, exist_ok=True)
    print("フォルダ作成完了")


# %%
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


# # %%
# # 学習関数の定義
# # =================================================
# def train_lgb(
#     input_x,
#     input_y,
#     input_id,
#     params,
#     list_nfold=[0, 1, 2, 3, 4],
#     n_splits=5,
# ):
#     metrics = []
#     imp = pd.DataFrame()
#     train_oof = np.zeros(len(input_x))

#     # cross-validation
#     cv = list(
#         StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
#             input_x, input_y
#         )
#     )

#     # 1.学習データと検証データに分離
#     for nfold in list_nfold:
#         print("-" * 20, nfold, "-" * 20)
#         print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

#         idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

#         x_tr, y_tr = (
#             input_x.loc[idx_tr, :],
#             input_y[idx_tr],
#         )
#         x_va, y_va = (
#             input_x.loc[idx_va, :],
#             input_y[idx_va],
#         )

#         print(x_tr.shape, x_va.shape)

#         # モデルの保存先名
#         fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")

#         if not os.path.isfile(
#             os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
#         ):  # if trained model, no training
#             # train
#             print("-------training start-------")
#             model = lgb.LGBMClassifier(**params)
#             model.fit(
#                 x_tr,
#                 y_tr,
#                 eval_set=[(x_tr, y_tr), (x_va, y_va)],
#                 callbacks=[
#                     lgb.early_stopping(stopping_rounds=100, verbose=True),
#                     lgb.log_evaluation(100),
#                 ],
#             )

#             # モデルの保存
#             with open(fname_lgb, "wb") as f:
#                 pickle.dump(model, f, protocol=4)

#         else:
#             print("すでに学習済みのためモデルを読み込みます")
#             with open(fname_lgb, "rb") as f:
#                 model = pickle.load(f)

#         # evaluate
#         y_tr_pred = model.predict_proba(x_tr)[:, 1]
#         y_va_pred = model.predict_proba(x_va)[:, 1]
#         metric_tr = mean_squared_error(y_tr, y_tr_pred)
#         metric_va = mean_squared_error(y_va, y_va_pred)
#         metrics.append([nfold, metric_tr, metric_va])
#         print(f"[rmse] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

#         # oof
#         train_oof[idx_va] = y_va_pred

#         # imp
#         _imp = pd.DataFrame(
#             {"col": input_x.columns, "imp": model.feature_importances_, "nfold": nfold}
#         )
#         imp = pd.concat([imp, _imp])

#     print("-" * 20, "result", "-" * 20)

#     # metric
#     metrics = np.array(metrics)
#     print(metrics)
#     print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
#         va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

#     print(f"[oof]{mean_squared_error(input_y, train_oof):.4f}")

#     # oof
#     train_oof = pd.concat(
#         [
#             input_id,
#             pd.DataFrame({"pred": train_oof}),
#         ],
#         axis=1,
#     )

#     # importance
#     imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
#     imp.columns = ["col", "imp", "imp_std"]

#     # stdout と stderr を一時的にリダイレクト
#     stdout_logger = logging.getLogger("STDOUT")
#     stderr_logger = logging.getLogger("STDERR")

#     sys_stdout_backup = sys.stdout
#     sys_stderr_backup = sys.stderr

#     sys.stdout = StreamToLogger(stdout_logger, logging.INFO)
#     sys.stderr = StreamToLogger(stderr_logger, logging.ERROR)
#     print("-" * 20, "result", "-" * 20)
#     print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

#     print(metrics)
#     print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
#         va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

#     print("-" * 20, "importance", "-" * 20)
#     print(imp.sort_values("imp", ascending=False)[:10])

#     # リダイレクトを解除
#     sys.stdout = sys_stdout_backup
#     sys.stderr = sys_stderr_backup

#     return train_oof, imp, metrics


# %%
# 推論関数の定義 =================================================
def predict_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1, 2, 3, 4],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

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


# %%
status_df["date"] = (
    status_df["year"].astype(str)
    + status_df["month"].astype(str).str.zfill(2)
    + status_df["day"].astype(str).str.zfill(2)
)
status_df["date"] = pd.to_datetime(status_df["date"])
status_df["weekday"] = status_df["date"].apply(get_weekday_jp)

status_df.head(10)

# %%

status_df[status_df["predict"] == 1].head()

# %%
plt.figure(figsize=(12, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plot_df = status_df[
        (status_df["weekday"] == "土曜日") & (status_df["station_id"] == i)
    ]
    sns.lineplot(x="hour", y="bikes_available", data=plot_df)

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
    temp_df["bikes_available"] = temp_df["bikes_available"].astype("object").bfill().astype("float16")
    
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
train_lstm_df =train_lstm_df.drop("predict",axis=1)
train_lstm_df.head()
#%%
# データの標準化
#特徴量を標準化するための変数
scaler = MinMaxScaler(feature_range=(0, 1))

#標準化された出力をもとにスケールに変換(inverse)するために必要な変数
scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
train_lstm_df_scale = scaler.fit_transform(train_lstm_df.iloc[:, 3:])


bikes_available_scale = scaler_for_inverse.fit_transform(train_lstm_df[["bikes_available"]])
print(train_lstm_df_scale.shape)


# %%
# バリデーション
length = len(train_lstm_df_scale)
train_size = int(length * 0.8)
test_size = length - train_size
train_lstm, test_lstm = train_lstm_df_scale[0:train_size, :], train_lstm_df_scale[train_size:length, :]
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
            temp_dataset= temp_dataset.sort_values(
                ["station_id", "date", "hour"]
            )
            temp_dataset["bikes_available"] = temp_dataset["bikes_available"].astype("object").bfill().astype("float16")
            temp_dataset = temp_dataset.sort_values(["date", "hour","station_id"],ascending=True)
            #予測には、前日の1時からのデータしか使用しないので、0時のデータは除く
            output_df = pd.concat([output_df,temp_dataset.iloc[70:,:]])
        else:
            output_df = pd.concat([output_df,temp_dataset.iloc[70:,:]])
    return output_df


# %%
# 予測日とその前日を含むデータフレームを作成する関数
# def make_sameday_thedaybefore_dataset(dataset, prediction_date):
#     # 前日の日付をtimedeltaで取得
#     before_date = prediction_date - timedelta(days=1)
#     prediction_date = str(prediction_date).split(" ")[0]
#     before_date = str(before_date).split(" ")[0]
#     # 予測日とその前日を含むものだけを抽出
#     temp_dataset = dataset[dataset["date"].isin([before_date, prediction_date])]

#     return temp_dataset


# # 評価用のデータセットを作成する関数
# def make_evaluation_dataset(dataset):
#     output_df = pd.DataFrame()
#     prediction_date_list = dataset[dataset["predict"] == 1]["date"].tolist()
#     for date in sorted(list(set(prediction_date_list))):
#         # 前日の日付をtimedeltaで取得
#         before_date = date - timedelta(days=1)
#         date = str(date).split(" ")[0]
#         before_date = str(before_date).split(" ")[0]
#         temp_dataset = dataset[dataset["date"].isin([before_date, date])]
#         # 前日のbikes_availableに欠損値が含まれるかどうかの判定
#         if (
#             temp_dataset[temp_dataset["date"] == before_date]["bikes_available"][1:]
#             .isnull()
#             .any()
#         ):
#             # 各ステーションで予測日の０時で前日の1時以降のデータを置換
#             # 予測日のbikes_availableの置換は、後ほど別途処理するので今回は無視
#             temp_dataset = (
#                 temp_dataset.sort_values(["station_id", "date", "hour"])
#                 .astype("object").bfill().astype("float16")
#             )
#             temp_dataset = temp_dataset.sort_values(
#                 ["date", "hour", "station_id"], ascending=True
#             )
#             # 予測には、前日の1時からのデータしか使用しないので、0時のデータは除く
#             output_df = pd.concat([output_df, temp_dataset.iloc[70:, :]])
#         else:  # 欠損値なし　→ そのまま前日分のデータを利用
#             output_df = pd.concat([output_df, temp_dataset.iloc[70:, :]])

#     return output_df


# %%
# 評価用のデータセット
evaluation_df = make_evaluation_dataset(evaluation_dataset_df)
evaluation_df.head()
# %%
evaluation_df.info()
# %%
#LSTMの出力結果でデータを補完しながら、提出用データフレームを作成する関数
def predict_eva_dataset(eva_dataset):
    submission_df = pd.DataFrame()
    #予測したbikes_availableを元のスケールに戻すための変数
    scaler_for_inverse = MinMaxScaler(feature_range=(0, 1))
    scale_y = scaler_for_inverse.fit_transform(eva_dataset[["bikes_available"]])
    prediction_date_list = eva_dataset[eva_dataset["predict"]==1]["date"].tolist()
    for date in sorted(list(set(prediction_date_list))):
        _,temp_eva_dataset = make_sameday_thedaybefore_dataset(eva_dataset,date)
        for i in range(0,1610,70):
            #モデルに入れるためのデータセット(1680×columns)
            temp_eva_dataset_train = temp_eva_dataset.iloc[i:1680+i,:]
            #predictは特徴量に使わないため、ここで削除
            temp_eva_dataset_train = temp_eva_dataset_train.drop("predict",axis=1)
            #データを標準化する
            scaler = MinMaxScaler(feature_range=(0, 1))
            temp_eva_dataset_scale = scaler.fit_transform(temp_eva_dataset_train.iloc[:,3:])
            
            #モデルに入力する形にデータを整形
            train = []
            xset = []
            for j in range(temp_eva_dataset_scale.shape[1]):
                a = temp_eva_dataset_scale[:, j]
                xset.append(a)
            train.append(xset)
            train = np.array(train)
            train = np.reshape(train, (train.shape[0], train.shape[1], train.shape[2]))
            
            #学習済みlstmモデルで予測
            predict_scale = model.predict(train)
            predict = scaler_for_inverse.inverse_transform(predict_scale)

            #次に使うbikes_availableに出力結果を補完
            temp_eva_dataset.iloc[1680+i:1750+i,3] = predict[0]

        submission_df= pd.concat([submission_df,temp_eva_dataset.iloc[1610:,:]])
        
    return submission_df
# %%
evaluation_df
# %%
#予測した結果を時系列で可視化して確認
submission_df = predict_eva_dataset(evaluation_df)
sns.lineplot(x ='date', y ='bikes_available',data = submission_df)

# %%
submission_df
#%%

#%%
lstm_submit_df = submission_df[submission_df["predict"]==1].sort_values(["station_id","date"])[["bikes_available"]]
lstm_submit_df["bikes_available"] = lstm_submit_df["bikes_available"].map(lambda x:0 if x < 0 else x)
lstm_submit_df.index = status_df[status_df["predict"]==1].index
# lstm_submit_df.to_csv("lstm_submission.csv",header=None)#
lstm_submit_df.head()
# %%
