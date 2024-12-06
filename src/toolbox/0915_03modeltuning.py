# %% [markdown]
## モデルチューニング！
# =================================================
# optuna使ったハイパーパラメータの設定
# 他のモデル：ロジスティック、SVM、ニューラルネットワーク
# アンサンブル：単純平均、重み付き、lasso使ったスタッキング

# %% ライブラリ読み込み
import datetime as dt
import gc
import glob
import json
import re
import os
import sys
import pickle
from IPython.display import display
import warnings

warnings.filterwarnings("ignore")
import zipfile

import numpy as np
import pandas as pd


# 可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib

sns.set(font="IPAexGothic")
#!%matplotlib inline


# 前処理
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
)

# モデリング
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import lightgbm as lgb


# 分布確認
# import ydata_profiling as php

# %%
file_path = "/tmp/work/src/input/Home Credit Default Risk/"

# ファイルの読み込み
df_train = pd.read_csv(file_path + "train.csv")
df_train.head()


# %% データセット 説明変数と目的変数
x_train, y_train, id_train = (
    df_train[["Pclass", "Fare"]],
    df_train[["Survived"]],
    df_train[["PassengerId"]],
)

print(x_train.shape, y_train.shape, id_train.shape)
# %%
# ===============================================
# optunaによるチューニング
# ===============================================

import optuna

# %%
# 探索しないパラメータ
params_base = {
    "boosting_type": "gbdt",
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "n_estimators": 100000,
    "bagging_freq": 1,
    "seed": 123,
}


def objective(trial):
    # 探索するパラメータ
    params_tuning = {
        "num_leaves": trial.suggest_int("num_leaves", 8, 256),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 200),
        "min_sum_hessian_in_leaf": trial.suggest_float(
            "min_sum_hessian_in_leaf", 1e-5, 1e-2, log=True
        ),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-2, 1e2, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-2, 1e2, log=True),
    }
    params_tuning.update(params_base)

    # モデル学習・評価
    list_metrics = []
    cv = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(
            x_train, y_train
        )
    )

    for nfold in np.arange(5):
        idx_tr, idx_va = cv[nfold][0], cv[nfold][1]
        x_tr, y_tr = x_train.loc[idx_tr, :], y_train.loc[idx_tr, :]
        x_va, y_va = x_train.loc[idx_va, :], y_train.loc[idx_va, :]

        model = lgb.LGBMClassifier(**params_tuning)
        model.fit(
            x_tr,
            y_tr,
            eval_set=[(x_tr, y_tr), (x_va, y_va)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=True),
                lgb.log_evaluation(0),
            ],
        )

        y_va_pred = model.predict_proba(x_va)[:, 1]
        metric_va = accuracy_score(y_va, np.where(y_va_pred >= 0.5, 1, 0))
        list_metrics.append(metric_va)

    metrics = np.mean(list_metrics)
    return metrics


# %%
sampler = optuna.samplers.TPESampler(seed=123)
study = optuna.create_study(sampler=sampler, direction="maximize")
study.optimize(objective, n_trials=30)
# %%
trial = study.best_trial
print(f"acc(best)={trial.value:.4f}")
display(trial.params)
# %%
params_beat = trial.params
params_beat.update(params_base)
display(params_beat)


# %%
# ===============================================
# 他のモデル
# ===============================================

# ロジスティック回帰

# ファイルの読み込み
df_train = pd.read_csv(file_path + "train.csv")

# データセット作成
x_train = df_train[["Pclass", "Age", "Embarked"]]
y_train = df_train[["Survived"]]
# %% 欠損値の補間
x_train.isna().sum()
# %%
# 欠損値の補間：数値データ
x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())
# 欠損値の補間：カテゴリ変数
x_train["Embarked"] = x_train["Embarked"].fillna(x_train["Embarked"].mode()[0])

x_train.isna().sum()

# %%カテゴリ変数の数値化
ohe = OneHotEncoder()
ohe.fit(x_train[["Embarked"]])
df_embarked = pd.DataFrame(
    ohe.transform(x_train[["Embarked"]]).toarray(),
    columns=[f"Embarked_{col}" for col in ohe.categories_[0]],
)

x_train = pd.concat([x_train, df_embarked], axis=1)
x_train = x_train.drop(columns=["Embarked"])
x_train.head()
# %%数値データの正規化
x_train["Pclass"] = (x_train["Pclass"] - x_train["Pclass"].min()) / (
    x_train["Pclass"].max() - x_train["Pclass"].min()
)

x_train["Age"] = (x_train["Age"] - x_train["Age"].min()) / (
    x_train["Age"].max() - x_train["Age"].min()
)

x_train.describe()

# %% バリデーション（ホールドアウト）
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=123
)
print(x_tr.shape, x_va.shape, y_tr.shape, y_va.shape)


# %% ロジスティック回帰モデル学習
# ===============================================

# モデル定義
from sklearn.linear_model import LogisticRegression

model_logis = LogisticRegression()

# 学習
model_logis.fit(x_tr, y_tr)

# 予測
y_va_pred = model_logis.predict(x_va)
print(f"accuracy:{accuracy_score(y_va, y_va_pred):.4f}")
print(y_va_pred[:5])
# %%
y_va_pred_proba = model_logis.predict_proba(x_va)
print(y_va_pred_proba[:5, :])


# %% SVM
# ===============================================
from sklearn.svm import SVC

model_svm = SVC(C=1.0, random_state=123, probability=True)

model_svm.fit(x_tr, y_tr)

y_va_pred = model_svm.predict(x_va)
print(f"accuracy:{accuracy_score(y_va, y_va_pred):.4f}")
print(y_va_pred[:5])

y_va_pred_proba = model_svm.predict_proba(x_va)
print(y_va_pred_proba[:5, :])


# %% =========ニューラルネットワーク================
# pip install tensorflow
# pip install tensorflow-hub

# %%
# pip install Keras


# %%

import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.keras.models import Sequential, Model  # type:ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization  # type:ignore
from tensorflow.keras.layers import Embedding, Flatten, Concatenate  # type:ignore
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
)  # type:ignore
from tensorflow.keras.optimizers import Adam, SGD  # type:ignore


# import tensorflow as tf
# import tensorflow.python.keras.backend as K
# from tensorflow.keras.models import Sequential, Model #type:ignore
# from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization #type:ignore
# from tensorflow.keras.layers import Embedding, Flatten, Concatenate #type:ignore
# from tensorflow.keras.callbacks import EarlyStopping,  ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler #type:ignore
# from tensorflow.keras.optimizers import Adam, SGD #type:ignore
# from keras import backend as K
# %%
def seed_everything(seed):
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(
        intra_op_parallelism_threads=1, inter_op_parallelism_threads=1
    )
    sess = tf.compat.v1.Session(
        graph=tf.compat.v1.get_default_graph(), config=session_conf
    )
    # tf.compat.v1.keras.backend.set_session(sess)
    K.set_session(sess)


# %%
# ファイル読み込み
df_train = pd.read_csv(file_path + "train.csv")

# データセット作成
x_train = df_train[["Pclass", "Age", "Embarked"]]
y_train = df_train[["Survived"]]
# %%
# 欠損値補間
x_train.isna().sum()
x_train.head()
x_train.describe()
# %%
x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())

for col in ["Pclass", "Age"]:
    value_min = x_train[col].min()
    value_max = x_train[col].max()
    x_train[col] = (x_train[col] - value_min) / (value_max - value_min)

x_train.describe()

# %%
# 欠損値、最頻値で補間
x_train["Embarked"] = x_train["Embarked"].fillna(x_train["Embarked"].mode()[0])

# エンコード
ohe = OneHotEncoder()
ohe.fit(x_train[["Embarked"]])
df_embarked = pd.DataFrame(
    ohe.transform(x_train[["Embarked"]]).toarray(),
    columns=[f"Embarked_{col}" for col in ohe.categories_[0]],
)
x_train = pd.concat([x_train.drop(columns=["Embarked"]), df_embarked], axis=1)

x_train.head()
# %%
x_tr, x_va, y_tr, y_va = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=123
)
print(x_tr.shape, x_va.shape, y_tr.shape, y_va.shape)


# %% モデル定義
def create_model():
    input_num = Input(shape=(5,))
    x_num = Dense(10, activation="relu")(input_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.3)(x_num)
    x_num = Dense(10, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.2)(x_num)
    x_num = Dense(5, activation="relu")(x_num)
    x_num = BatchNormalization()(x_num)
    x_num = Dropout(0.1)(x_num)
    out = Dense(1, activation="sigmoid")(x_num)

    model = Model(
        inputs=input_num,
        outputs=out,
    )

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["binary_crossentropy"],
    )

    return model


model = create_model()
model.summary()


# %%

seed_everything(seed=123)
model = create_model()
model.fit(
    x=x_tr,
    y=y_tr,
    validation_data=(x_va, y_va),
    batch_size=8,
    epochs=10000,
    callbacks=[
        ModelCheckpoint(
            filepath="model_keras.weights.h5",
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
            monitor="val_loss", mode="min", factor=0.1, patience=5, verbose=1
        ),
    ],
    verbose=1,
)


# np.infに変更したり、filepath="model_keras.weights.h5"（weights.追加）したり、import tensorflow.python.keras.backend as Kを追加したり
# %%
y_va_pred = model.predict(x_va, batch_size=8, verbose=1)
print(f"accuracy:{accuracy_score(y_va,np.where(y_va_pred>=0.5,1,0)):.4f}")


# %%ニューラルネットワーク 埋め込みそうあり
# ファイルの読み込み
df_train = pd.read_csv(file_path + "train.csv")

# データセット作成
x_train = df_train[["Pclass", "Age", "Cabin"]]
y_train = y_train[["Survived"]]

# %%
display(x_train.head())
print(x_train.isna().sum())
# %%
# 欠損値の補間
x_train["Age"] = x_train["Age"].fillna(x_train["Age"].mean())

# 正規化
for col in ["Pclass", "Age"]:
    value_min = x_train[col].min()
    value_max = x_train[col].max()
    x_train[col] = (x_train[col] - value_min) / (value_max - value_min)

x_train.describe()

# %% カテゴリ変数の欠損値処理とラベルエンコーディング
x_train["Cabin"] = x_train["Cabin"].fillna("None")

le = LabelEncoder()
le.fit(x_train[["Cabin"]])
x_train["Cabin"] = le.transform(x_train["Cabin"])

print(le.classes_)
print("count", len(le.classes_))
# %% 学習データと検証データの分離
x_train.describe()

# %%
x_train_num, x_train_cat = x_train[["Pclass", "Age"]], x_train[["Cabin"]]

x_num_tr, x_num_va, x_cat_tr, x_cat_va, y_tr, y_va = train_test_split(
    x_train_num, x_train_cat, y_train, test_size=0.2, stratify=y_train, random_state=123
)

print(
    x_num_tr.shape,
    x_num_va.shape,
    x_cat_tr.shape,
    x_cat_va.shape,
    y_tr.shape,
    y_va.shape,
)


# %%
def create_model_embedding():
    ############## num
    input_num = Input(shape=(2,))
    layer_num = Dense(10, activation="relu")(input_num)
    layer_num = BatchNormalization()(layer_num)
    layer_num = Dropout(0.2)(layer_num)
    layer_num = Dense(10, activation="relu")(layer_num)

    ############## cat
    input_cat = Input(shape=(1,))
    layer_cat = input_cat[:, 0]
    layer_cat = Embedding(input_dim=148, output_dim=74)(layer_cat)
    layer_cat = Dropout(0.2)(layer_cat)
    layer_cat = Flatten()(layer_cat)

    ############# concat
    hidden_layer = Concatenate()([layer_num, layer_cat])
    hidden_layer = Dense(50, activation="relu")(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer)
    hidden_layer = Dropout(0.1)(hidden_layer)
    hidden_layer = Dense(20, activation="relu")(hidden_layer)
    hidden_layer = BatchNormalization()(hidden_layer)
    hidden_layer = Dropout(0.1)(hidden_layer)
    output_layer = Dense(1, activation="sigmoid")(hidden_layer)

    model = Model(
        inputs=[input_num, input_cat],
        outputs=output_layer,
    )

    model.compile(
        optimizer="Adam",
        loss="binary_crossentropy",
        metrics=["binary_crossentropy"],
    )
    return model


model = create_model_embedding()
model.summary()

# %%
seed_everything(seed=123)
model = create_model_embedding()
model.fit(
    x=[x_num_tr, x_cat_tr],
    y=y_tr,
    validation_data=([x_num_va, x_cat_va], y_va),
    batch_size=8,
    epochs=10000,
    callbacks=[
        ModelCheckpoint(
            filepath="model_keras_embedding.weights.h5",
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
            monitor="val_loss",
            mode="min",
            factor=0.1,
            patience=5,
            verbose=1,
        ),
    ],
    verbose=1,
)
# %% モデル評価
y_va_pred = model.predict([x_num_va, x_cat_va], batch_size=8, verbose=1)
print(f"accuracy:{accuracy_score(y_va, np.where(y_va_pred > 0.5 ,1, 0)):4f}")


# %%
# ===============================================
# アンサンブル
# ===============================================

# 単純平均
np.random.seed(123)
df = pd.DataFrame(
    {
        "true": [0] * 700 + [1] * 300,
        "pred1": np.arange(1000) + np.random.rand(1000) * 1200,
        "pred2": np.arange(1000) + np.random.rand(1000) * 1000,
        "pred3": np.arange(1000) + np.random.rand(1000) * 800,
    }
)
df["pred1"] = np.clip(df["pred1"] / df["pred1"].max(), 0, 1)
df["pred2"] = np.clip(df["pred2"] / df["pred2"].max(), 0, 1)
df["pred3"] = np.clip(df["pred3"] / df["pred3"].max(), 0, 1)

df_train, df_test = train_test_split(
    df, test_size=0.8, stratify=df["true"], random_state=123
)
df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train.head()

# %%
df_train["pred_ensemble1"] = (
    df_train["pred1"] + df_train["pred2"] + df_train["pred3"]
) / 3
df_train.head()


# %%
def evaluate_ensemble(input_df, col_pred):
    print(
        f'[auc] model1:{roc_auc_score(input_df["true"], input_df["pred1"]):.4f}, model2:{roc_auc_score(input_df["true"], input_df["pred2"]):.4f}, model3:{roc_auc_score(input_df["true"], input_df["pred3"]):.4f} -> ensemble:{roc_auc_score(input_df["true"], input_df[col_pred]):.4f}'
    )


evaluate_ensemble(df_train, col_pred="pred_ensemble1")
# %%
df_test["pred_ensemble1"] = (df_test["pred1"] + df_test["pred2"] + df_test["pred3"]) / 3

evaluate_ensemble(df_test, col_pred="pred_ensemble1")
# %% 重み付き平均
weight = [0.3, 0.3, 0.4]
weight = weight / np.sum(weight)
print(weight)

df_train["pred_ensemble2"] = (
    df_train["pred1"] * weight[0]
    + df_train["pred2"] * weight[1]
    + df_train["pred3"] * weight[2]
)

df_train[["true", "pred1", "pred2", "pred3", "pred_ensemble2"]].head()

# %%
evaluate_ensemble(df_train, col_pred="pred_ensemble2")
# %%
df_test["pred_ensemble2"] = (
    df_test["pred1"] * weight[0]
    + df_test["pred2"] * weight[1]
    + df_test["pred3"] * weight[2]
)

evaluate_ensemble(df_test, col_pred="pred_ensemble2")

# %% スタッキング
from sklearn.linear_model import Lasso

x, y = df_train[["pred1", "pred2", "pred3"]], df_train[["true"]]
oof = np.zeros(len(x))
models = []

cv = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(x, y))

for nfold in np.arange(5):
    idx_tr, idx_va = cv[nfold][0], cv[nfold][0]
    x_tr, y_tr = x.loc[idx_tr, :], y.loc[idx_tr, :]
    x_va, y_va = x.loc[idx_va, :], y.loc[idx_va, :]

    model = Lasso(alpha=0.01)
    model.fit(x_tr, y_tr)
    models.append(model)

    y_va_pred = model.predict(x_va)
    oof[idx_va] = y_va_pred

df_train["pred_ensemble3"] = oof
df_train["pred_ensemble3"] = df_train["pred_ensemble3"].clip(lower=0, upper=1)
df_train[["true", "pred1", "pred2", "pred3", "pred_ensemble3"]].head()
# %%
evaluate_ensemble(df_train, col_pred="pred_ensemble3")
# %%
df_test["pred_ensemble3"] = 0

for model in models:
    df_test["pred_ensemble3"] += model.predict(
        df_test[["pred1", "pred2", "pred3"]]
    ) / len(models)

df_test["pred_ensemble3"] = df_test["pred_ensemble3"].clip(lower=0, upper=1)
evaluate_ensemble(df_test, col_pred="pred_ensemble3")
# %%
