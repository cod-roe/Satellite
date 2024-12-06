# %% [markdown]
## 全体像サンプル！
# =================================================
# 一連の流れ、チューニングまで行っているコード、ホームクレジットより、少々中身が古い

# %%インポート
# ライブラリ読み込み
# =================================================
import datetime
import gc
import re
import os
import pickle
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

sns.set(font="IPAexGothic")
#!%matplotlib inline
# import ydata_profiling as pdp


# sckit-learn
from sklearn.preprocessing import OneHotEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# lightGBM
import lightgbm as lgb


# %%設定
# Config
# =================================================

######################
# serial #
######################
serial_number = 4  # スプレッドシートA列の番号


######################
# Data #
######################
input_path = (
    "/tmp/work/src/input/Home Credit Default Risk/"  # フォルダ名適宜変更すること
)
file_path = "/tmp/work/src/script/bl1.2_exp002_tu01.py"  # ファイル名は適宜変更すること
file_name = os.path.splitext(os.path.basename(file_path))[0]


######################
# Dataset #
######################
target_columns = "TARGET"
sub_index = "SK_ID_CURR"

######################
# ハイパーパラメータの設定
######################
# exp003_tune01
params = {
    "num_leaves": 32,
    "min_child_samples": 144,
    "min_sum_hessian_in_leaf": 0.002822464888447712,
    "feature_fraction": 0.5134059690079296,
    "bagging_fraction": 0.8964992342830249,
    "lambda_l1": 3.507716474434041,
    "lambda_l2": 7.3693406833353325,
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "verbosity": -1,
    "learning_rate": 0.05,
    "n_estimators": 100000,
    "bagging_freq": 1,
    "random_state": 123,
}


# 初期値
# params = {
# 	'boosting_type': 'gbdt',
# 	'objective': 'binary',
# 	'metric': 'auc',
# 	'learning_rate': 0.05,
# 	'num_leaves': 32,
# 	'n_estimators':100000,
# 	'random_state': 123,
# 	'importance_type': 'gain',
# }


# =================================================
# Utilities #
# =================================================


# 今の日時
def dt_now():
    dt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9)))
    return dt_now


# %%
# メモリ削減関数
# =================================================
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print(f"Memory usage of dataframe is {start_mem:.2f} MB")

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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
# 学習関数の定義
# =================================================
def train_lgb(
    input_x,
    input_y,
    input_id,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
):
    metrics = []
    imp = pd.DataFrame()
    train_oof = np.zeros(len(input_x))

    # cross-validation
    cv = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
            input_x, input_y
        )
    )

    # output配下に現在のファイル名のフォルダを作成し、移動
    os.chdir("/tmp/work/src/output")
    if not os.path.isdir(file_name):
        os.makedirs(file_name)
        print(f"{file_name}フォルダ作成しました")
    os.chdir("/tmp/work/src/output/" + file_name)
    print(f"保存場所: {os.getcwd()}")

    # 1.学習データと検証データに分離
    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)
        print(dt_now().strftime("%Y年%m月%d日 %H:%M:%S"))

        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]

        x_tr, y_tr, id_tr = (
            input_x.loc[idx_tr, :],
            input_y[idx_tr],
            input_id.loc[idx_tr, :],
        )
        x_va, y_va, id_va = (
            input_x.loc[idx_va, :],
            input_y[idx_va],
            input_id.loc[idx_va, :],
        )

        print(x_tr.shape, x_va.shape)

        # train
        model = lgb.LGBMClassifier(**params)
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
        fname_lgb = f"model_lgb_fold{nfold}.pickle"
        with open(fname_lgb, "wb") as f:
            pickle.dump(model, f, protocol=4)

        # evaluate
        y_tr_pred = model.predict_proba(x_tr)[:, 1]
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # oof
        train_oof[idx_va] = y_va_pred

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

    print(f"[oof]{roc_auc_score(input_y, train_oof):.4f}")

    # oof
    train_oof = pd.concat(
        [
            input_id,
            pd.DataFrame({"pred": train_oof}),
        ],
        axis=1,
    )

    # importance
    imp = imp.groupby("col")["imp"].agg(["mean", "std"]).reset_index(drop=False)
    imp.columns = ["col", "imp", "imp_std"]

    print("-" * 20, "importance", "-" * 20)
    print(imp.sort_values("imp", ascending=False)[:10])

    return train_oof, imp, metrics


# %%
# 推論関数の定義 =================================================
def predict_lgb(
    input_x,
    input_id,
    list_nfold=[0, 1, 2, 3, 4],
):
    # モデル格納場所へ移動
    os.chdir("/tmp/work/src/output/" + file_name)

    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = f"model_lgb_fold{nfold}.pickle"
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

    # 平均値算出
    pred = pd.concat(
        [
            input_id,
            pd.DataFrame({"pred": pred.mean(axis=1)}),
        ],
        axis=1,
    )
    print("Done.")

    return pred


# %%
# 前処理の定義 カテゴリ変数をcategory型に
# =================================================
def data_pre00(df):
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].astype("category")
    print("カテゴリ変数をcategory型に変換しました")
    return df


# %%
# 特徴量生成
# =================================================
def data_pre01(df):
    # exp001 app特徴量追加と欠損値処理
    # 欠損値の対処（nullに変換）
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
    display(df["DAYS_EMPLOYED"].value_counts())
    print(f'正の値の割合{(df["DAYS_EMPLOYED"]>0).mean():.4f}')
    print(f'正の値の個数{(df["DAYS_EMPLOYED"]>0).sum()}')
    print('"DAYS_EMPLOYED"の正の値を0に変更しました')
    print("=" * 20)
    # 特徴量生成
    print("今まで:", df.shape)
    # 特徴量1:総所得金額を世帯人数で割った値
    df["INCOME_div_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]
    # 特徴量2:総所得金額を就労期間で割った値
    df["INCOME_div_EMPLOYED"] = df["AMT_INCOME_TOTAL"] / df["DAYS_EMPLOYED"]
    # 特徴量3:外部スコアの平均値など
    df["EXT_SOURCE_mean"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].mean(
        axis=1
    )
    df["EXT_SOURCE_max"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].max(
        axis=1
    )
    df["EXT_SOURCE_min"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].min(
        axis=1
    )
    df["EXT_SOURCE_std"] = df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].std(
        axis=1
    )
    df["EXT_SOURCE_count"] = (
        df[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].notnull().sum(axis=1)
    )
    # 特徴量4:就労期間を年齢で割った値
    df["DAYS_EMPLOYED_div_BIRTH"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    # 特徴量5:年金支払いを所得金額で割った値
    df["ANNUITY_div_INCOME"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    # 特徴量6:年金支払額を借入金で割った値
    df["ANNUITY_div_CREDIT"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    print("処理後:", df.shape)
    print("appデータに特徴量を生成しました(完了)")
    print("=" * 40)
    return df


def data_pre02(pos, app_date):
    # exp02:posとappデータの結合
    # ①カテゴリ変数をone-ho-encodingで数値に変換
    pos_ohe = pd.get_dummies(pos, columns=["NAME_CONTRACT_STATUS"], dummy_na=True)
    col_ohe = sorted(list(set(pos_ohe.columns) - set(pos.columns)))
    print(f'①"NAME_CONTRACT_STATUS"を{len(col_ohe)}個のカラムにエンコードしました。')

    # ②SK_IDCURRをキーに集約処理
    pos_ohe_agg = pos_ohe.groupby("SK_ID_CURR").agg(
        {
            # 数値の集約
            "MONTHS_BALANCE": ["mean", "std", "min", "max"],
            "CNT_INSTALMENT": ["mean", "std", "min", "max"],
            "CNT_INSTALMENT_FUTURE": ["mean", "std", "min", "max"],
            "SK_DPD": ["mean", "std", "min", "max"],
            "SK_DPD_DEF": ["mean", "std", "min", "max"],
            # カテゴリ変数をone-hot-encodingした値の集約
            "NAME_CONTRACT_STATUS_Active": ["mean"],
            "NAME_CONTRACT_STATUS_Amortized debt": ["mean"],
            "NAME_CONTRACT_STATUS_Approved": ["mean"],
            "NAME_CONTRACT_STATUS_Canceled": ["mean"],
            "NAME_CONTRACT_STATUS_Completed": ["mean"],
            "NAME_CONTRACT_STATUS_Demand": ["mean"],
            "NAME_CONTRACT_STATUS_Returned to the store": ["mean"],
            "NAME_CONTRACT_STATUS_Signed": ["mean"],
            "NAME_CONTRACT_STATUS_XNA": ["mean"],
            "NAME_CONTRACT_STATUS_nan": ["mean"],
            # IDのユニーク数をカウント（ついでにレコード数もカウント）
            "SK_ID_PREV": ["count", "nunique"],
        }
    )

    # カラム名の付与
    pos_ohe_agg.columns = [i + "_" + j for i, j in pos_ohe_agg.columns]
    pos_ohe_agg = pos_ohe_agg.reset_index(drop=False)

    print(f"②集約し{len(pos_ohe_agg.columns)}個のカラムを作成しました")
    # pos_ohe_agg.head()

    # ③SK_ID_CURRをキーにして結合
    print(f"結合前: [app]{app_date.shape} ,[pos]{pos_ohe_agg.shape}")
    df = pd.merge(app_date, pos_ohe_agg, on="SK_ID_CURR", how="left")
    print("結合後", df.shape)
    print("③posのデータを加工後、appデータと結合しました（完了）")
    print("=" * 40)

    display(df.head()[:5])

    return df, pos_ohe_agg


# %% ファイル
# ファイルの読み込み application_test
# =================================================
# app_test = reduce_mem_usage(pd.read_csv(input_path+"application_test.csv"))
# print('application_test:app_test')
# print(app_test.shape)
# display(app_test.head())


# ファイルの読み込み application_train
# =================================================
# app_train = reduce_mem_usage(pd.read_csv(input_path+"application_train.csv"))
# print('application_train:app_train')
# print(app_train.shape)
# display(app_train.head())


# ファイルの読み込み bureau
# =================================================
# bureau = reduce_mem_usage(pd.read_csv(input_path+"bureau.csv"))
# print('bureau')
# print(bureau.shape)
# display(bureau.head())


# ファイルの読み込み bureau_balance
# =================================================
# bureau_balance = reduce_mem_usage(pd.read_csv(input_path+"bureau_balance.csv"))
# print('bureau_balance')
# print(bureau_balance.shape)
# display(bureau_balance.head())


# ファイルの読み込み credit_card_balance
# =================================================
# credit_card_balance = reduce_mem_usage(pd.read_csv(input_path+"credit_card_balance.csv"))
# print('credit_card_balance')
# print(credit_card_balance.shape)
# display(credit_card_balance.head())


# ファイルの読み込み installments_payments
# # =================================================
# installments_payments = reduce_mem_usage(pd.read_csv(input_path+"installments_payments.csv"))
# print('installments_payments')
# print(installments_payments.shape)
# display(installments_payments.head())


# ファイルの読み込み POS_CASH_balance
# =================================================
# pos = reduce_mem_usage(pd.read_csv(input_path+"POS_CASH_balance.csv"))
# print('POS_CASH_balance:pos')
# print(pos.shape)
# display(pos.head())


# ファイルの読み込み previous_application
# # =================================================
# previous_application = reduce_mem_usage(pd.read_csv(input_path+"previous_application.csv"))
# print('previous_application')
# print(previous_application.shape)
# display(previous_application.head())


# %% [markdown]
## 分析start!
# ==========================================================

# %% ファイルの読み込み
# application_train
# =================================================
app_train = reduce_mem_usage(pd.read_csv(input_path + "application_train.csv"))
print("application_train:app_train")
print(app_train.shape)
display(app_train.head())


# application_test
# =================================================
app_test = reduce_mem_usage(pd.read_csv(input_path + "application_test.csv"))
print("application_test:app_test")
print(app_test.shape)
display(app_test.head())


# POS_CASH_balance
# =================================================
pos = reduce_mem_usage(pd.read_csv(input_path + "POS_CASH_balance.csv"))
print("POS_CASH_balance:pos")
print(pos.shape)
display(pos.head())


# %%
# 出力表示数増やす
# pd.set_option('display.max_rows',None)
# pd.set_option('display.max_columns',None)


# %%
# ファイルの確認
# =================================================
# datainput = []
# for dirname, _, filenames in os.walk(input_path):
# 	for i, datafilename in enumerate(filenames):
# 		# print(os.path.join(dirname,filename))
# 		print('='*40)
# 		print(i,datafilename)
# 		datainput.append(datafilename[:-4])
# print(datainput)


# %%[markdown]
# これまでの処理
# ==========================================================
# exp01:app特徴量追加と欠損値処理
app_train = data_pre01(app_train)

# exp02:posとappデータの結合
df_train, pos_ohe_agg = data_pre02(pos, app_train)

# %%[markdown]
# 今回の実験（）
# ==========================================================

# ***********実験***********

# %%
# 7-30:データセットの作成
x_train = df_train.drop(columns=[target_columns, sub_index])
y_train = df_train[target_columns]
id_train = df_train[[sub_index]]


# カテゴリ型に変換
x_train = data_pre00(x_train)

# %%
# 7-31:モデル学習
print(x_train.info())

train_oof, imp, metrics = train_lgb(
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)

# %%
# 7-32:説明変数の重要度の確認
imp.to_csv(f"imp_{file_name}.csv", index=None)
imp.sort_values("imp", ascending=False)[:10]


# %%
# 7-33:推論データのデータセット作成

# %% 特徴量生成
# これまでの特徴量生成
# exp01
app_test = data_pre01(app_test)

# exp02
df_test = pd.merge(app_test, pos_ohe_agg, on="SK_ID_CURR", how="left")
print(df_test.shape)

# 今回の分：データ結合

# データセット作成
x_test = df_test.drop(columns=[sub_index])
id_test = df_test[[sub_index]]

# カテゴリ型に変換
x_test = data_pre00(x_test)
x_test.info()

# %%
# 7-34:推論処理
test_pred = predict_lgb(
    x_test,
    id_test,
    list_nfold=[0, 1, 2, 3, 4],
)

# %%
# 7-35:提出ファイルの作成
df_submit = test_pred.rename(columns={"pred": "TARGET"})
print(df_submit.shape)
display(df_submit.head())
df_submit.to_csv(f"submission_{file_name}.csv", index=None)


# %%
