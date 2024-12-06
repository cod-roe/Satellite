# %% [markdown]
## ベースライン！
# =================================================
# ベースライン作成：


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
# import category_encoders as ce

from sklearn.model_selection import StratifiedKFold, train_test_split  # , KFold
# from sklearn.metrics import mean_squared_error,accuracy_score, roc_auc_score ,confusion_matrix


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
comp_name = "JR_Snowman"
# 評価：weighted mean absolute error（WMAE:重み付き平均絶対誤差） 回帰

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


# %% ファイルの読み込み
# Load Data
# =================================================
# 8 train.csv
# train = load_data(8)
train = pd.read_csv("../input/JR_Snowman/train.csv", encoding="shift-jis")

# %%
display(train.shape)
display(train.info())
display(train.head())
display(train.describe().T)

# %%
# out_of_service.csv
oos = pd.read_csv("../input/JR_Snowman/out_of_service.csv", encoding="shift-jis")
# %%
display(oos.shape)
display(oos.info())
display(oos.head())
display(oos.describe().T)

# %%
# sample_submit.csv
sample_sub = pd.read_csv("../input/JR_Snowman/sample_submit.csv", encoding="shift-jis")

display(sample_sub.shape)
display(sample_sub.info())
display(sample_sub.head())
display(sample_sub.describe().T)


# %%
# test.csv
test = pd.read_csv("../input/JR_Snowman/test.csv", encoding="shift-jis")

display(test.shape)
display(test.info())
display(test.head())
display(test.describe().T)

# %%
# stop_station_location.csv
stop_loc = pd.read_csv(
    "../input/JR_Snowman/stop_station_location.csv", encoding="shift-jis"
)

display(stop_loc.shape)
display(stop_loc.info())
display(stop_loc.head())
display(stop_loc.describe().T)


# %%
# tunnel_location.csv
tunnel_loc = pd.read_csv(
    "../input/JR_Snowman/tunnel_location.csv", encoding="shift-jis"
)

display(tunnel_loc.shape)
display(tunnel_loc.info())
display(tunnel_loc.head())
display(tunnel_loc.describe().T)


# %%
# wind_location.csv
wind_loc = pd.read_csv("../input/JR_Snowman/wind_location.csv", encoding="shift-jis")

display(wind_loc.shape)
display(wind_loc.info())
display(wind_loc.head())
display(wind_loc.describe().T)
# %%
# snowfall_location.csv
snow_loc = pd.read_csv(
    "../input/JR_Snowman/snowfall_location.csv", encoding="shift-jis"
)

display(snow_loc.shape)
display(snow_loc.info())
display(snow_loc.head())
display(snow_loc.describe().T)
# %%
# diagram.csv
diagram = pd.read_csv("../input/JR_Snowman/diagram.csv", encoding="shift-jis")

display(diagram.shape)
display(diagram.info())
display(diagram.head())

# %%
# kanazawa_nosnow.csv
k_nosnow = pd.read_csv("../input/JR_Snowman/kanazawa_nosnow.csv", encoding="shift-jis")

display(k_nosnow.shape)
display(k_nosnow.info())
display(k_nosnow.head())
display(k_nosnow.describe().T)
# %%
# 気象庁データ（weather.csv）
weather = pd.read_csv("../input/JR_Snowman/weather.csv", encoding="cp932")

display(weather.shape)
display(weather.info())
display(weather.head())
display(weather.describe().T)
# %%
# ■積雪深計データ（snowfall.csv）
snowfall = pd.read_csv("../input/JR_Snowman/snowfall.csv", encoding="cp932")

display(snowfall.shape)
display(snowfall.info())
display(snowfall.head())
display(snowfall.describe().T)

# %%
# ■風速計データ 下条川.csv
wind_01 = pd.read_csv("../input/JR_Snowman/wind_0\下条川.csv", encoding="cp932")

display(wind_01.shape)
display(wind_01.info())
display(wind_01.head())
# display(wind_01.describe().T)


# %%
# データセット作成
# =================================================
train.columns
# %%
train.head()

# %%
train["年月日"] = pd.to_datetime(train["年月日"], format="%Y-%m-%d")

# %%
train["month"] = train["年月日"].astype(str).apply(lambda x: x[5:7])
# %%
train["day"] = train["年月日"].astype(str).apply(lambda x: x[8:])
# %%
train["yearmonth"] = train["年月日"].astype(str).apply(lambda x: x[:7])

# %%

# %%
snow_man = train[train["合計"] > 0]

# %%
snow_man["month"].value_counts()

# %%
len(train[train["合計"] == 0]) / train.shape[0]
# %%
train.groupby("month")["合計"].describe()

# %%
sns.barplot(train.groupby("month")["合計"].mean())


# %%
# カテゴリ変数
data_pre00(train)



# %%
# 学習データと検証データの期間の設定
list_cv_month = [
    [["2016-01", "2016-02"], ["2016-03"]],
    [["2016-01", "2016-02", "2016-03"], ["2016-12"]],
]
# %%
# データセット

x_train = train[[
    # '年月日', 
    '列車番号',
    # '停車駅名', 
    # 'フェンダー部分(東京方向)', 
    # '台車部分', 
    # 'フェンダー部分(金沢方向)', 
    # '合計',
    'month', 'day', 
    # 'yearmonth'
    ]]
y_train = train["合計"]

id_train = train[["yearmonth"]]

# %%
# WMAE


# WMAEの計算関数
def wmae(y_true, y_pred, weights):
    error = np.abs(y_true - y_pred)
    return np.sum(weights * error) / len(y_true)


# LightGBM用のカスタム評価関数
def wmae_sklearn(y_true, y_pred):
    weights = np.abs(y_true) * 10**4 + 1 
    error = np.abs(y_true - y_pred)
    wmae = np.sum(weights * error) / len(y_true)
    return "wmae", wmae, False  # Falseは「小さいほど良い」を意味


# lgbm初期値
params = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "None",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}


# %%
# 学習関数の定義
# =================================================
def train_lgb(
    input_x,
    input_y,
    input_id,
    params,
    list_nfold=[0, 1],
):
    metrics = []
    imp = pd.DataFrame()
    # train_oof = np.zeros(len(input_x))
    # shap_v = pd.DataFrame()
    # 重みの計算
    weights = np.abs(y_train) * 10**4 + 1

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
                eval_metric=wmae_sklearn,  # カスタム評価関数を指定
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

        metric_tr = wmae(y_tr, y_tr_pred, weights)
        metric_va = wmae(y_va, y_va_pred, weights)
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
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0, 1],
)

#%%
# 説明変数の重要度の確認
imp.groupby(['col'])['imp'].agg(['mean','std']).sort_values('mean',ascending=False)[:10]


#%%
# 推論処理
# =================================================
#%%
test.head()

#%%
# %%
test["年月日"] = pd.to_datetime(test["年月日"], format="%Y-%m-%d")


test["month"] = test["年月日"].astype(str).apply(lambda x: x[5:7])

test["day"] = test["年月日"].astype(str).apply(lambda x: x[8:])

test["yearmonth"] = test["年月日"].astype(str).apply(lambda x: x[:7])

#%%
# カテゴリ変数
data_pre00(test)

#%%
# データセット

x_test = test[[
    # '年月日', 
    '列車番号',
    # '停車駅名', 
    # 'フェンダー部分(東京方向)', 
    # '台車部分', 
    # 'フェンダー部分(金沢方向)', 
    # '合計',
    'month', 'day', 
    # 'yearmonth'
    ]]
#%%
id_test = test[["Unnamed: 0","yearmonth"]]
#%%
id_test = id_test.rename(columns={"Unnamed: 0":"id"})
#%%
id_test.head()
# %%
# 推論関数の定義 =================================================
def predict_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1],
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
            input_id["id"],
            pd.DataFrame(pred.mean(axis=1)),
        ],
        axis=1,
    )
    print("Done.")

    return pred


# %%
# 推論処理
# =================================================

test_pred = predict_lgb(x_test,id_test)
#%%
test_pred.head()
#%%

# %%
# %%
# submitファイルの出力
# =================================================


# %%
test_pred.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.csv"), index=False, header=False
)


# %%
