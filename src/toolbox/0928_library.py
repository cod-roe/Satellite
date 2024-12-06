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
import random

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
)  # , MinMaxScaler, LabelEncoder, OneHotEncoder

# バリデーション、評価測定
from sklearn.model_selection import StratifiedKFold  # ,train_test_split, KFold
from sklearn.metrics import roc_auc_score  # accuracy_score,confusion_matrix

# tensorflow
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
from tensorflow.keras.metrics import AUC

# 次元圧縮
from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# import umap

# 学習器
# from sklearn.svm import SVC
from sklearn.linear_model import Lasso

# lightGBM
import lightgbm as lgb
# lightGBM精度測定
# import shap

# パラメータチューニング
import optuna


# %%
# Config
# =================================================

######################
# serial #
######################
serial_number = 27  # スプレッドシートAの番号

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



#%%
# set up
# =================================================
# utils
warnings.filterwarnings("ignore")
sns.set(font="IPAexGothic")
###!%matplotlib inline
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