# %% [markdown]
## バリデーション！
# =================================================
#  検証月を最後にしてみる＊8：


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
serial_number = 12  # スプレッドシートAの番号

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
display(station_df["city"].nunique())
sns.histplot(data=station_df["city"])
# %%
sns.scatterplot(data=station_df, x="lat", y="long", hue="city", alpha=0.1)


# %%
# weather読み込み
weather_df = load_data(4)
# %%
# 日付の型変更
weather_df["date"] = pd.to_datetime(weather_df["date"])
weather_df.info()
# %%
sns.histplot(weather_df["events"])

# %%
weather_df.describe().T


# %%
# statusにstationのstation_idをマージ
status_df = pd.merge(status_df, station_df, on="station_id", how="left")
display(status_df.head())

# weatherのprecipitationをマージ
status_df = pd.merge(status_df, weather_df, on="date", how="left")

display(status_df.head())
display(status_df.info())
# %%

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
# 今回の変更
# データを分けるカラム作成
status_df["yearmonth"] = status_df["date"].astype(str).apply(lambda x: x[:7])
status_df["yearmonth"].head()


# %%

# データセット作成
# 2014/09/01以前をtrain、以後をtestに分割(testはpredict = 1のものに絞る)
# =================================================
train = status_df[status_df["date"] < "2014-09-01"]

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
        "hour",
        "station_id",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "precipitation",
        "week_num",
        "bikes_available_at0",
        "bikes_available",
    ]
]

valid = valid[
    [
        "hour",
        "station_id",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "precipitation",
        "week_num",
        "bikes_available_at0",
        "bikes_available",
    ]
]


id_train = train[["yearmonth"]]


test = test[
    [
        "hour",
        "station_id",
        "CE_city_1",
        "CE_city_2",
        "CE_city_3",
        "CE_city_4",
        "CE_city_5",
        "precipitation",
        "week_num",
        "bikes_available_at0",
    ]
]


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
        ],
        ["2014-08"],
    ],
    [
        [
            "2013-09",
            "2013-10",
            "2013-11",
            "2013-12",
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
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
)


# %%
# imp
imp_sort = imp.sort_values("imp", ascending=False)
display(imp_sort)

# %%
imp_sort.to_csv(os.path.join(OUTPUT_EXP, f"imp_{name}.csv"), index=False, header=False)


# %%
# 推論関数の定義 =================================================
def predict_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1, 2, 3, 4, 5, 6, 7, 8],
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
# 推論処理
# =================================================
valid_x = valid.drop("bikes_available",axis=1)
valid_y =valid["bikes_available"]

id_valid = pd.DataFrame(valid.index)

test_pred = predict_lgb(
    valid_x,
    id_valid,
    list_nfold= [0,1,2,3,4,5,6,7,8],
)

#%%
test_pred.head()

#%%
metric_valid = mean_squared_error(valid_y, test_pred.iloc[:,1], squared=False)

print(f"[rmse] valid:{metric_valid:.4f}")









#%%
# ここまで


# %%
# submitファイルの出力
# =================================================
sub_index = status_df[status_df["predict"] == 1].index

df_submit = pd.DataFrame(list(zip(sub_index, test_pred)))


print(df_submit.shape)
display(df_submit.head())

# %%
df_submit.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.csv"), index=False, header=False
)


# %%
