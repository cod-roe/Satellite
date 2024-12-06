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

#クラスタで選別
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr


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
