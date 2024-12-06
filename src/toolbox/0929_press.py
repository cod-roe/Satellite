#%%

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



# 次元圧縮
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
#%%
# PCA

def data_pca(df, n_components):
    # Genderを数値化
    # df["Gender"] = pd.get_dummies(df["Gender"], drop_first=True, dtype="uint8")

    df2 = df.copy()

    # 対数処理
    for i in df2.columns:
        df2[i] = np.log10(df2[i] + 1)

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
# umap

def data_umap(df, n_components):
    # Genderを数値化
    # df["Gender"] = pd.get_dummies(df["Gender"], drop_first=True, dtype="uint8")

    df2 = df.copy()

    # 対数処理
    for i in df2.columns:
        df2[i] = np.log10(df2[i] + 1)

    # 標準化 pca,UMAPの適用のため 結合データはもとの標準化前のデータにする
    std = StandardScaler().fit_transform(df2)
    df_std = pd.DataFrame(std, columns=df.columns)

    umap1 = umap.UMAP(n_components=n_components, random_state=123)
    X_embedded = umap1.fit_transform(df_std)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap="Spectral", s=10)
    plt.title("UMAP projection of the Iris dataset")
    plt.show()

    df_umap = pd.DataFrame(X_embedded)

    df_umap_col = []
    for i in range(n_components):
        df_umap_col.append(f"umap0{i}")
    df_umap.columns = df_umap_col
    print("処理前:", df.shape)
    df = pd.concat([df, df_umap], axis=1)
    print("処理後:", df.shape)

    df.head()

    return df




# %%
# t-SNE

def data_tsne(
    df,
):
    # Genderを数値化
    # df["Gender"] = pd.get_dummies(df["Gender"], drop_first=True, dtype="uint8")

    df2 = df.copy()

    # 対数処理
    for i in df2.columns:
        df2[i] = np.log10(df2[i] + 1)

    # 標準化 pca,UMAPの適用のため 結合データはもとの標準化前のデータにする
    std = StandardScaler().fit_transform(df2)
    df_std = pd.DataFrame(std, columns=df.columns)

    tsne1 = TSNE(n_components=3, random_state=123)
    X_embedded = tsne1.fit_transform(df_std)

    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_train, cmap="viridis", s=10)
    plt.title("t-SNE projection of the Iris dataset")
    plt.show()

    df_tsne = pd.DataFrame(X_embedded)

    df_tsne_col = []
    for i in range(3):
        df_tsne_col.append(f"tsne0{i}")
    df_tsne.columns = df_tsne_col
    print("処理前:", df.shape)
    df = pd.concat([df, df_tsne], axis=1)
    print("処理後:", df.shape)

    df.head()

    return df