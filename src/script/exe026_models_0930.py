# %% [markdown]
##予測外したところの確認からのモデル複数作ってアンサンブル！
# =================================================
# lightgmb単体exe021_log_0927が元
# ['T_Bil', 'pc01', 'AST_GOT', 'AG_ratio', 'ALP', 'Alb/ALT_ex3', 'TP', 'D_Bil']

# %%
# ライブラリ読み込み
# =================================================
import datetime as dt

# import gc
from IPython.display import display

# import json
import logging

import os
import pickle

# import re
import sys
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
)  # MinMaxScaler, LabelEncoder, OneHotEncoder

# バリデーション、評価測定
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import roc_auc_score, confusion_matrix  # accuracy_score

# 次元圧縮
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# lightGBM
import lightgbm as lgb

# lightGBM精度測定
import shap

# パラメータチューニング
# import optuna

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 30  # スプレッドシートAの番号

######################
# Data #
######################
comp_name = "Chronic_liver_disease"
# 評価：AUC（Area Under the Curve）

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
target_columns = "disease"
# sub_index = "index"

######################
# ハイパーパラメータの設定
######################
# lgbm初期値
# lgbm初期値
params = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}


params_dart = {
    "boosting_type": "dart",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 32,
    "n_estimators": 10000,
    "random_state": 123,
    "importance_type": "gain",
}

# params = {
#     "max_depth": 12,
#     "num_leaves": 124,
#     "min_child_samples": 6,
#     "min_sum_hessian_in_leaf": 0.007289404984433297,
#     "feature_fraction": 0.5491498779894461,
#     "bagging_fraction": 0.547193048553452,
#     "lambda_l1": 1.6793356827390677,
#     "lambda_l2": 1.2047378854765676,
#     "boosting_type": "gbdt",
#     "objective": "binary",
#     "metric": "auc",
#     "verbosity": -1,
#     "learning_rate": 0.05,
#     "n_estimators": 100000,
#     "bagging_freq": 1,
#     "random_state": 123,
# }


# # %%
# # Utilities #
# # =================================================


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
def data_pre_catg(df):
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].astype("category")
    print("カテゴリ変数をcategory型に変換しました")
    df.info()
    return df


# %%
# 特徴量生成 exp002
# =================================================
def data_ft01(df):
    # 特徴量1:直接ビリルビン/総ビリルビン比（D/T比） 3/2
    df["D/T_ex2"] = df["D_Bil"] / df["T_Bil"]

    # 特徴量2:AST/ALT比（De Ritis比 # 6/5
    df["AST/ALT_ex2"] = df["AST_GOT"] / df["ALT_GPT"]

    # 特徴量3:.フェリチン/AST比 フェリチンの代わりにタンパク質 7/6
    df["TP/AST_ex2"] = df["TP"] / df["AST_GOT"]

    # 特徴量4:.グロブリン  1/(8*9)
    df["Globulin_ex2"] = np.reciprocal(df["Alb"] / df["AG_ratio"])

    print("処理後:", df.shape)
    print("特徴量を生成しました(完了)")
    print("=" * 40)
    return df


# %%
# 特徴量作成 exp03
# =================================================
def data_ft02(df):
    # 特徴量1:ビリルビン/酵素比　総ビリルビン / ALT または 総ビリルビン / AST 2/5 2/6
    df["TB/ALT_ex3"] = df["T_Bil"] / df["ALT_GPT"]
    df["TB/AST_ex3"] = df["T_Bil"] / df["AST_GOT"]

    # 特徴量2: 総ビリルビン/ALP比  2/4
    df["TB/ALP_ex3"] = df["T_Bil"] / df["ALP"]

    # 特徴量3:アルブミン/ALT比 8/5
    df["Alb/ALT_ex3"] = df["Alb"] / df["ALT_GPT"]

    # 特徴量4:総タンパク/ALT比 7/5
    df["TP/ALT_ex3"] = df["TP"] / df["ALT_GPT"]

    # 特徴量5:ALP/AST比またはALP/ALT比 4/6 4/5
    df["ALP/AST_ex3"] = df["ALP"] / df["AST_GOT"]
    df["ALP/ALT_ex3"] = df["ALP"] / df["ALT_GPT"]

    # 特徴量6:総ビリルビン / アルブミン 2/8
    df["TB/Alb_ex3"] = df["T_Bil"] / df["Alb"]

    print("処理後:", df.shape)
    print("特徴量を生成しました(完了)")
    print("=" * 40)
    return df


# %%特徴量作成比率一気に
def create_features(train, features, target):
    target_col = train[target]
    train = train[features]
    for col in features:
        for col2 in features:
            if col != col2:
                # print(col, col2)
                col_name = col + "/" + col2
                train[col_name] = train[col] / train[col2]
        # train2.columns
    train[target] = target_col
    return train


# %%
def data_pca(df, n_components):
    # Genderを数値化
    # df["Gender"] = pd.get_dummies(df["Gender"], drop_first=True, dtype="uint8")

    df2 = df.copy()

    # 対数処理
    for i in ["T_Bil", "D_Bil", "ALP", "ALT_GPT", "AST_GOT"]:
        try:
            df2[i] = np.log10(df2[i] + 1)
        except Exception as e:
            print(f"How exceptional! {e}")
            pass

    # 標準化 pca,UMAPの適用のため 結合データはもとの標準化前のデータにする
    std = StandardScaler().fit_transform(df2)
    df_std = pd.DataFrame(std, columns=df.columns)

    pca = PCA(n_components=n_components, random_state=123)
    pca.fit(df_std)
    print(np.cumsum(pca.explained_variance_ratio_))

    plt.plot(range(1, n_components + 1), np.cumsum(pca.explained_variance_ratio_))
    plt.xticks(range(1, n_components + 1))
    plt.xlabel("components")
    plt.xlabel("components")
    plt.ylabel("cumulative explained variance")

    X_pc = pca.transform(df_std)

    df_pc = pd.DataFrame(np.concatenate([X_pc], axis=1))
    df_pc_col = []
    for i in range(n_components):
        df_pc_col.append(f"pc0{i}")
    df_pc.columns = df_pc_col
    print("処理前:", df.shape)
    df = pd.concat([df, df_pc], axis=1)
    print("処理後:", df.shape)

    df.head()

    return df


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
    shap_v = pd.DataFrame()

    # cross-validation
    cv = list(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123).split(
            input_x, input_y
        )
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
        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_{m_name}_fold{nfold}.pickle")

        if not os.path.isfile(fname_lgb):  # if trained model, no training
            # train
            print("-------training start-------")
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
            with open(fname_lgb, "wb") as f:
                pickle.dump(model, f, protocol=4)

        else:
            print("すでに学習済みのためモデルを読み込みます")
            with open(fname_lgb, "rb") as f:
                model = pickle.load(f)

        # evaluate
        y_tr_pred = model.predict_proba(x_tr)[:, 1]
        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_tr = roc_auc_score(y_tr, y_tr_pred)
        metric_va = roc_auc_score(y_va, y_va_pred)
        metrics.append([nfold, metric_tr, metric_va])
        print(f"[auc] tr:{metric_tr:.4f}, va:{metric_va:.4f}")

        # shap_v  & 各特徴量のSHAP値の平均絶対値で重要度を算出
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_x)

        _shap_importance = np.abs(shap_values).mean(axis=0)
        _shap = pd.DataFrame(
            {"col": input_x.columns, "shap": _shap_importance, "nfold": nfold}
        )
        shap_v = pd.concat([shap_v, _shap])

        # 重要度が高い特徴を選択
        # selected_features = np.argsort(shap_importance)[::-1]

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

    oof = f"[oof]{roc_auc_score(input_y, train_oof):.4f}"

    print(oof)

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

    # shap値
    shap_v = shap_v.groupby("col")["shap"].agg(["mean", "std"]).reset_index(drop=False)
    shap_v.columns = ["col", "shap", "shap_std"]

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
    print(input_x.shape)
    print(metrics)
    print(f"[cv] tr:{metrics[:,1].mean():.4f}+-{metrics[:,1].std():.4f}, \
        va:{metrics[:,2].mean():.4f}+-{metrics[:,1].std():.4f}")

    print(oof)

    print("-" * 20, "importance", "-" * 20)
    print(imp.sort_values("imp", ascending=False)[:10])

    # リダイレクトを解除
    sys.stdout = sys_stdout_backup
    sys.stderr = sys_stderr_backup

    return train_oof, imp, metrics, shap_v


# %%
# 推論関数の定義 =================================================
def predict_lgb_en(
    input_x,
    list_nfold=[0, 1, 2, 3, 4],
):
    pred = np.zeros((len(input_x), len(list_nfold)))

    for nfold in list_nfold:
        print("-" * 20, nfold, "-" * 20)

        fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_{m_name}_fold{nfold}.pickle")
        with open(fname_lgb, "rb") as f:
            model = pickle.load(f)

        # 推論
        pred[:, nfold] = model.predict_proba(input_x)[:, 1]

    # 平均値算出
    pred = pd.DataFrame(pred.mean(axis=1))
    print("Done.")

    return pred


# %% setup
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
# train

df_train = load_data(2)
display(df_train.shape)
df_train.info()


# %%
df_train.columns
# %%
# これまでの処理
# ==========================================================
# 特徴量作成4つ
df_train = data_ft01(df_train)
# 特徴量作成8つ
df_train = data_ft02(df_train)


# %%
# モデル1つ目　exe019_select02_0927のやつの再現
# train_3_0 = train_[
#     [
#         "Age",
#         "Gender",
#         "T_Bil",
#         "D_Bil",
#         "ALP",
#         "ALT_GPT",
#         "AST_GOT",
#         "TP",
#         "Alb",
#         "AG_ratio",
#         "D_Bil/T_Bil",
#         "AST_GOT/ALT_GPT",
#         "TP/AST_GOT",
#         "Alb/AG_ratio",
#         "T_Bil/ALT_GPT",
#         "T_Bil/AST_GOT",
#         "T_Bil/ALP",
#         "Alb/ALT_GPT",
#         "TP/ALT_GPT",
#         "ALP/AST_GOT",
#         "ALP/ALT_GPT",
#         "T_Bil/Alb",
#         "disease",
#     ]
# ]

# train_3_0.columns

# %%
set_file = df_train
x_train01 = set_file.drop([target_columns], axis=1)
y_train01 = set_file[target_columns]
id_train01 = pd.DataFrame(set_file.index)

print(x_train01.shape, y_train01.shape, id_train01.shape)

x_train01 = data_pre_catg(x_train01)
# 標準化=>PCA処理
# Genderを数値化
x_train01["Gender"] = pd.get_dummies(
    x_train01["Gender"], drop_first=True, dtype="uint8"
)
x_train01 = data_pca(x_train01, 8)
x_train01.columns

# %%

x_train01 = x_train01[
    ["T_Bil", "pc00", "AST_GOT", "AG_ratio", "ALP", "Alb/ALT_ex3", "TP", "D_Bil"]
]

x_train01.shape

# %%学習モデル01 lgbm
m_name = "lgbm01"
train_oof_01, imp_01, metrics_01, shap_01 = train_lgb(
    x_train01,
    y_train01,
    id_train01,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# =================================================
# =================================================
# =================================================


# %% [markdown]
## Main 分析start!
# =========================================================

# %%
df_train = load_data(2)
display(df_train.shape)
df_train.info()
# Genderを数値化
df_train["Gender"] = pd.get_dummies(df_train["Gender"], drop_first=True, dtype="uint8")
# %%特徴量作成
df_train66 = df_train.copy()
features = df_train66.drop(["disease", "Gender", "Age"], axis=1).columns
print("特徴量生成前：", df_train66.shape)
train_ = create_features(df_train66, features, ["disease", "Gender", "Age"])
print("特徴量生成後：", train_.shape)

print(train_.info)
# %%
train_.columns


# %%
# 相関高いもの消す
train_2 = drop_features(train_, "disease")
print(train_2.shape)
# %%
train_2.columns


# %%
# クラス分け一つ目
# remove_collinear_features1 前から1つ
train_3_1, clusters, columns = remove_collinear_features(
    train_2, "disease", threshold=1.0, s=0
)
print(train_3_1.shape)
print(train_3_1.columns)
# %%
# 元のやつと結合して、重複削除
df_train_3_1 = pd.concat([df_train, train_3_1], axis=1)
df_train_3_1 = df_train_3_1.loc[:, ~df_train_3_1.columns.duplicated()]
df_train_3_1.columns

# %%
# データセット作成
# =================================================
set_file = df_train_3_1
x_train_3_1 = set_file.drop([target_columns], axis=1)
y_train_3_1 = set_file[target_columns]
id_train_3_1 = pd.DataFrame(set_file.index)


print(x_train_3_1.shape, y_train_3_1.shape, id_train_3_1.shape)

#
# %%コラムセレクト
# なし


# %%学習モデル02 lgbm
# m_name = "dart"
m_name = "lgbm02"
train_oof_02, imp_02, metrics_02, shap_v_02 = train_lgb(
    x_train_3_1,
    y_train_3_1,
    id_train_3_1,
    params,  # params_dart,params
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# =================================================
# =================================================
# =================================================
# %%
# クラス分け2つ目
# remove_collinear_features2 後ろから１
train_3_2, clusters, columns = remove_collinear_features(
    train_2, "disease", threshold=1.0, s=-1
)
print(train_3_2.shape)
print(train_3_2.columns)

# %%
# 元のやつと結合して、重複削除
df_train_3_2 = pd.concat([df_train, train_3_2], axis=1)
df_train_3_2 = df_train_3_2.loc[:, ~df_train_3_2.columns.duplicated()]
df_train_3_2.columns

# %%
# データセット作成
# =================================================
set_file = df_train_3_2
x_train_3_2 = set_file.drop([target_columns], axis=1)
y_train_3_2 = set_file[target_columns]
id_train_3_2 = pd.DataFrame(set_file.index)

print(x_train_3_2.shape, y_train_3_2.shape, id_train_3_2.shape)
# %%
# 標準化=>PCA処理
x_train_3_2 = data_pca(x_train_3_2, 8)
# %%
x_train_3_2.head()
# %%
# ベストセレクション
list2 = [
    "T_Bil",
    "pc00",
    "ALT_GPT",
    "AG_ratio/D_Bil",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "AG_ratio/AST_GOT",
    "TP",
]

x_train_3_2 = x_train_3_2[list2]

# %%学習モデル03 lgbm
# m_name = "dart"
m_name = "lgbm03"
train_oof_03, imp_03, metrics_03, shap_v_03 = train_lgb(
    x_train_3_2,
    y_train_3_2,
    id_train_3_2,
    params,  # params_dart,params
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# =================================================
# =================================================
# =================================================

# %%
# クラス分け3つ目
# remove_collinear_features3 後ろから2
train_3_3, clusters, columns = remove_collinear_features(
    train_2, "disease", threshold=1.0, s=-2
)
print(train_3_3.shape)
print(train_3_3.columns)

# %%
# 元のやつと結合して、重複削除
df_train_3_3 = pd.concat([df_train, train_3_3], axis=1)
df_train_3_3 = df_train_3_3.loc[:, ~df_train_3_3.columns.duplicated()]
df_train_3_3.columns

# %%
# データセット作成
# =================================================
set_file = df_train_3_3
x_train_3_3 = set_file.drop([target_columns], axis=1)
y_train_3_3 = set_file[target_columns]
id_train_3_3 = pd.DataFrame(set_file.index)

print(x_train_3_3.shape, y_train_3_3.shape, id_train_3_3.shape)
# %%
# 標準化=>PCA処理
x_train_3_3 = data_pca(x_train_3_3, 8)
# %%
x_train_3_3.head()
# %%
# ベストセレクション
list3 = [
    "T_Bil",
    "pc00",
    "AG_ratio/ALT_GPT",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "TP",
    "D_Bil",
    "Alb/T_Bil",
    "TP/ALP",
]


x_train_3_3 = x_train_3_3[list3]

# %%学習モデル04 lgbm
# m_name = "dart"
m_name = "lgbm04"
train_oof_04, imp_04, metrics_04, shap_v_04 = train_lgb(
    x_train_3_3,
    y_train_3_3,
    id_train_3_3,
    params,  # params_dart,params
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# =================================================
# =================================================
# =================================================


# %%
# %%
# クラス分け4つ目
# remove_collinear_features3 後ろから2
train_3_4, clusters, columns = remove_collinear_features(
    train_2, "disease", threshold=1.0, s=1
)
print(train_3_4.shape)
print(train_3_4.columns)

# %%
# 元のやつと結合して、重複削除
df_train_3_4 = pd.concat([df_train, train_3_4], axis=1)
df_train_3_4 = df_train_3_4.loc[:, ~df_train_3_4.columns.duplicated()]
df_train_3_4.columns

# %%
# データセット作成
# =================================================
set_file = df_train_3_4
x_train_3_4 = set_file.drop([target_columns], axis=1)
y_train_3_4 = set_file[target_columns]
id_train_3_4 = pd.DataFrame(set_file.index)

print(x_train_3_4.shape, y_train_3_4.shape, id_train_3_4.shape)
# %%
# 標準化=>PCA処理
# x_train_3_4 = data_pca(x_train_3_4, 8)
# %%
x_train_3_4.head()
# %%
# ベストセレクション
# なし
list4 = [
    "T_Bil",
    "ALT_GPT",
    "T_Bil/AG_ratio",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "D_Bil",
    "TP",
    "ALP/AST_GOT",
    "D_Bil/T_Bil",
]

x_train_3_4 = x_train_3_4[list4]

# %%学習モデル05 lgbm
# m_name = "dart"
m_name = "lgbm05"
train_oof_05, imp_05, metrics_05, shap_v_05 = train_lgb(
    x_train_3_4,
    y_train_3_4,
    id_train_3_4,
    params,  # params_dart,params
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)
# =================================================
# =================================================
# =================================================


# セット
df_train_en = pd.DataFrame(
    {
        "pred1": train_oof_01["pred"],
        "pred2": train_oof_02["pred"],
        "pred3": train_oof_03["pred"],
        "pred4": train_oof_04["pred"],
        "pred5": train_oof_05["pred"],
        "pred_ensemble5": (
            train_oof_01["pred"]
            + train_oof_02["pred"]
            + train_oof_03["pred"]
            + train_oof_04["pred"]
            + train_oof_05["pred"]
        )
        / 5,
        "true": y_train01,
    }
)


# %%
def evaluate_ensemble(input_df, col_pred):
    print(
        f'[auc] lgbm1:{roc_auc_score(input_df["true"], input_df["pred1"]):.4f}, lgbm2:{roc_auc_score(input_df["true"], input_df["pred2"]):.4f},lgbm3:{roc_auc_score(input_df["true"], input_df["pred3"]):.4f},lgbm4:{roc_auc_score(input_df["true"], input_df["pred4"]):.4f},lgbm5:{roc_auc_score(input_df["true"], input_df["pred5"]):.4f},  -> ensemble:{roc_auc_score(input_df["true"], input_df[col_pred]):.4f}'
    )


evaluate_ensemble(df_train_en, col_pred="pred_ensemble5")

# %%

# %%
# %%
# テストファイルの読み込み
# =================================================
df_test = load_data(1)
display(df_test.shape)
df_test.info()


# %%
# これまでの処理
# ==========================================================
# 特徴量作成4つ
df_test = data_ft01(df_test)
# 特徴量作成8つ
df_test = data_ft02(df_test)


#

# %%
set_file = df_test
x_test01 = set_file
id_test01 = pd.DataFrame(set_file.index)

print(x_test01.shape, id_test01.shape)

x_test01 = data_pre_catg(x_test01)
# 標準化=>PCA処理
# Genderを数値化
x_test01["Gender"] = pd.get_dummies(x_test01["Gender"], drop_first=True, dtype="uint8")
x_test01 = data_pca(x_test01, 8)
x_test01.columns

# %%

x_test01 = x_test01[
    ["T_Bil", "pc00", "AST_GOT", "AG_ratio", "ALP", "Alb/ALT_ex3", "TP", "D_Bil"]
]

x_test01.shape

# %%推論処理01 lgbm
print("lgbm")
m_name = "lgbm01"
test_pred_01 = predict_lgb_en(
    x_test01,
    list_nfold=[0, 1, 2, 3, 4],
)
# =================================================
# =================================================
# =================================================


# %% [markdown]
## Main 分析start!
# =========================================================

# %%
df_test = load_data(1)
display(df_test.shape)
df_test.info()
# Genderを数値化
df_test["Gender"] = pd.get_dummies(df_test["Gender"], drop_first=True, dtype="uint8")
# %%特徴量作成
df_test66 = df_test.copy()
features = df_test66.drop(["Gender", "Age"], axis=1).columns
print("特徴量生成前：", df_test66.shape)
test_ = create_features(df_test66, features, ["Gender", "Age"])
print("特徴量生成後：", test_.shape)

print(test_.info)
# %%
test_.columns


# %%
# クラス分け一つ目

df_test_3_1 = test_[
    [
        "Age",
        "Gender",
        "T_Bil",
        "D_Bil",
        "ALP",
        "ALT_GPT",
        "AST_GOT",
        "TP",
        "Alb",
        "AG_ratio",
        "T_Bil/D_Bil",
        "T_Bil/ALT_GPT",
        "T_Bil/ALP",
        "ALP/ALT_GPT",
    ]
]

# %%
# データセット作成
# =================================================
set_file = df_test_3_1
x_test_3_1 = set_file
id_test_3_1 = pd.DataFrame(set_file.index)


print(x_test_3_1.shape, id_test_3_1.shape)

#
# %%コラムセレクト
# なし


# %%推論処理02 lgbm
print("lgbm")
m_name = "lgbm02"
test_pred_3_1 = predict_lgb_en(
    x_test_3_1,
    list_nfold=[0, 1, 2, 3, 4],
)
# =================================================
# =================================================
# =================================================
# %%
# クラス分け2つ目

df_test_3_2 = test_[
    [
        "Age",
        "Gender",
        "T_Bil",
        "D_Bil",
        "ALP",
        "ALT_GPT",
        "AST_GOT",
        "TP",
        "Alb",
        "AG_ratio",
        "AG_ratio/D_Bil",
        "AG_ratio/ALP",
        "Alb/ALP",
        "AST_GOT/ALT_GPT",
        "AG_ratio/T_Bil",
        "AG_ratio/AST_GOT",
    ]
]

# %%
# データセット作成
# =================================================
set_file = df_test_3_2
x_test_3_2 = set_file
id_test_3_2 = pd.DataFrame(set_file.index)

print(x_test_3_2.shape, id_test_3_2.shape)
# %%
# 標準化=>PCA処理
x_test_3_2 = data_pca(x_test_3_2, 8)
# %%
x_test_3_2.head()
# %%
# ベストセレクション
list2 = [
    "T_Bil",
    "pc00",
    "ALT_GPT",
    "AG_ratio/D_Bil",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "AG_ratio/AST_GOT",
    "TP",
]

x_test_3_2 = x_test_3_2[list2]


# %%推論処理03 lgbm
print("lgbm")
m_name = "lgbm03"
test_pred_3_2 = predict_lgb_en(
    x_test_3_2,
    list_nfold=[0, 1, 2, 3, 4],
)
# =================================================
# =================================================
# =================================================

# %%
# クラス分け3つ目
# remove_collinear_features3 後ろから2

df_test_3_3 = test_[
    [
        "Age",
        "Gender",
        "T_Bil",
        "D_Bil",
        "ALP",
        "ALT_GPT",
        "AST_GOT",
        "TP",
        "Alb",
        "AG_ratio",
        "AST_GOT/D_Bil",
        "Alb/AG_ratio",
        "TP/ALP",
        "AST_GOT/T_Bil",
        "Alb/T_Bil",
        "AG_ratio/ALT_GPT",
    ]
]

# %%
# データセット作成
# =================================================
set_file = df_test_3_3
x_test_3_3 = set_file
id_test_3_3 = pd.DataFrame(set_file.index)

print(x_test_3_3.shape, id_test_3_3.shape)
# %%
# 標準化=>PCA処理
x_test_3_3 = data_pca(x_test_3_3, 8)
# %%
x_test_3_3.head()
# %%
# ベストセレクション
list3 = [
    "T_Bil",
    "pc00",
    "AG_ratio/ALT_GPT",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "TP",
    "D_Bil",
    "Alb/T_Bil",
    "TP/ALP",
]


x_test_3_3 = x_test_3_3[list3]

# %%推論処理04 lgbm
print("lgbm")
m_name = "lgbm04"
test_pred_3_3 = predict_lgb_en(
    x_test_3_3,
    list_nfold=[0, 1, 2, 3, 4],
)
# =================================================
# =================================================
# =================================================


# %%
# %%
# クラス分け4つ目

df_test_3_4 = test_[
    [
        "Age",
        "Gender",
        "T_Bil",
        "D_Bil",
        "ALP",
        "ALT_GPT",
        "AST_GOT",
        "TP",
        "Alb",
        "AG_ratio",
        "D_Bil/T_Bil",
        "ALP/AG_ratio",
        "T_Bil/AST_GOT",
        "T_Bil/AG_ratio",
        "ALP/AST_GOT",
    ]
]

# %%
# データセット作成
# =================================================
set_file = df_test_3_4
x_test_3_4 = set_file
id_test_3_4 = pd.DataFrame(set_file.index)

print(x_test_3_4.shape, id_test_3_4.shape)
# %%
# 標準化=>PCA処理
# x_test_3_4 = data_pca(x_test_3_4, 8)
# %%
x_test_3_4.head()
# %%
# ベストセレクション
# なし
list4 = [
    "T_Bil",
    "ALT_GPT",
    "T_Bil/AG_ratio",
    "AST_GOT",
    "AG_ratio",
    "ALP",
    "D_Bil",
    "TP",
    "ALP/AST_GOT",
    "D_Bil/T_Bil",
]

x_test_3_4 = x_test_3_4[list4]

# %%推論処理05 lgbm
print("lgbm")
m_name = "lgbm05"
test_pred_3_4 = predict_lgb_en(
    x_test_3_4,
    list_nfold=[0, 1, 2, 3, 4],
)
# =================================================
# =================================================
# =================================================

# %%
test_pred = pd.concat(
    [test_pred_01, test_pred_3_1, test_pred_3_2, test_pred_3_3, test_pred_3_4], axis=1
)
test_pred.head()

# %%
test_pred = pd.concat(
    [
        id_test01,
        test_pred.mean(axis=1),
    ],
    axis=1,
)

print(test_pred.shape)
test_pred.head()

# %%
# submitファイルの出力
# =================================================

test_pred.to_csv(
    os.path.join(OUTPUT_EXP, f"submission_{name}.csv"), index=False, header=False
)


# %%
# seed_values = [123,456,789]

# for i, seed_value in enumerate(seed_values):
#     print(f"Training model with seed {seed_value}")
#     params['random_state'] = seed_value  # シード値を変更
#     if not skip_run:
#         train_oof, imp, metrics = train_lgb(
#             x_train,
#             y_train,
#             id_train,
#             params,
#             list_nfold=[0, 1, 2, 3, 4],
#             n_splits=5,
#         )


# %%
# データセット作成
# =================================================
set_file = df_train
x_train = set_file
y_train = set_file[target_columns]
id_train = pd.DataFrame(set_file.index)


print(x_train.shape, y_train.shape, id_train.shape)
# %%学習(
train_oof, imp, metrics, shap_v = train_lgb(
    x_train,
    y_train,
    id_train,
    params,
    list_nfold=[0, 1, 2, 3, 4],
    n_splits=5,
)

# %%
# 説明変数の重要度の確認上位20
# =================================================
imp_sort = imp.sort_values("imp", ascending=False)
display(imp_sort[:20])
# imp_sort.to_csv(os.path.join(OUTPUT_EXP, f"imp_{name}.csv"), index=None)

shap_sort = shap_v.sort_values("shap", ascending=False)
display(shap_sort[:20])

shap_value = shap_sort.copy()
imp_value = imp_sort.copy()

# %%
# 一個ずつ加えて精度確認
select_list = []
scores = []
for i in shap_value["col"]:  # [:20]:
    select_list.append(i)
    print(select_list)
    x_trains = x_train[select_list]
    print(x_trains.shape)
    train_oof, imp, metrics, shap_v = train_lgb(
        x_trains,
        y_train,
        id_train,
        params,
        list_nfold=[0, 1, 2, 3, 4],
        n_splits=5,
    )
    scores.append([len(select_list), roc_auc_score(y_train, train_oof["pred"])])


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
display(shap_sort[:15].reset_index())
print(shap_sort[:18].shape)
list(shap_sort["col"][:15].values)
# %%


# %% [markdown]
##予測外したところの確認
# =================================================
# %%
# train_oof,imp,metrics,


# 予測ラベル（0 または 1 のクラス予測）を取得
y_pred_class = (train_oof["pred"] > 0.5).astype(int)

# 混合行列
cm = confusion_matrix(y_train, y_pred_class)
display(cm)

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# %%
df_train.describe().T


# %%
# 特徴量の重要度を表示
for nfold in [0, 1, 2, 3, 4]:
    fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")

    with open(fname_lgb, "rb") as f:
        model = pickle.load(f)

    lgb.plot_importance(model, max_num_features=20)
    plt.show()

# %%
# 予測を間違えたところの表示
# y_pred_class = (train_oof["pred"] > 0.5).astype(int)
# 予測確率を取得
y_pred_prob = train_oof["pred"].copy()

# 予測ラベル（0 または 1 のクラス予測）
y_pred_class = (y_pred_prob > 0.5).astype(int)  # 予測1

# 誤分類されたインスタンスを抽出
# (1と予測したのに0だったところ)
misclassified = x_train[y_train != y_pred_class]

# 対応する実際のラベルと予測ラベル
actual_labels = y_train[y_train != y_pred_class]
predicted_labels = y_pred_class[y_train != y_pred_class]

# 誤分類されたデータポイントを確認
print("Misclassified instances:\n", misclassified)
print("Actual labels:\n", actual_labels)
print("Predicted labels:\n", predicted_labels)
# %%
predicted_labels.count()
# %%
display(misclassified)
# %%
misclassified.describe().T
# %%
# 正しく分類したところ
corclassified = x_train[y_train == y_pred_class]
print(y_pred_class[y_train == y_pred_class].count())
corclassified.describe().T

# %%
for nfold in [0, 1, 2, 3, 4]:
    fname_lgb = os.path.join(EXP_MODEL, f"model_lgb_fold{nfold}.pickle")

    with open(fname_lgb, "rb") as f:
        model = pickle.load(f)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_train)
        # 誤分類されたデータポイントのSHAP値を確認
        # shap.force_plot(explainer.expected_value, shap_values[misclassified.index], x_train.iloc[misclassified.index])
        shap.summary_plot(
            shap_values[misclassified.index],
            x_train.iloc[misclassified.index],
            plot_type="bar",
        )

# %%
misclassified.shape
# %%
x_train.shape
# %%
shap_values.shape
# %%
explainer.expected_value.shape
# %%
x_train.iloc[misclassified.index]
# %%
print(shap_values[1].shape)
# %%
fig, axes = plt.subplots(1, 2, figsize=(60, 40))
sns.histplot(data=misclassified, x="D_Bil", ax=axes[0])
sns.histplot(data=corclassified, x="D_Bil", ax=axes[1])

# %%
