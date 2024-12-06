# %% [markdown]
## EDA,特徴量エンジニア！
#=================================================
#基本、前処理、簡単な特徴量作成、テキストデータ

# %% ライブラリ読み込み

import datetime as dt
import numpy as np
import pandas as pd
import os
import pickle
import gc
import glob
from IPython.display import display

import pandas as pd
import numpy as np

#分布確認
import ydata_profiling as php

#可視化
import matplotlib.pyplot as plt
import seaborn as sns
import japanize_matplotlib 

#前処理
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder


#モデリング
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import mean_absolute_error as mae

import lightgbm as lgb

import warnings

warnings.filterwarnings('ignore')
#!%matplotlib inline 
sns.set(font='IPAexGothic') 

# %%
file_path = '/tmp/work/input/'
#%%





# %% 今の階層
os.getcwd()
# %%
#ファイル名取得
#========================================
file_path = "/tmp/work/src/exp/sample.py"
def filename(file_path):
  file_name = os.path.splitext(os.path.basename(file_path))[0]
  # print(file_name)
  return file_name


# %%
#output配下に現在のファイル名のフォルダを作成
#========================================

def namefolder(file_path):
  # file_path = "/tmp/work/src/exp/sample.py"
  file_name = os.path.splitext(os.path.basename(file_path))[0]
  os.chdir('/tmp/work/src/output')
  os.makedirs(file_name)
  return file_name

file_path = "/tmp/work/src/exp/sample.py"
file_name = namefolder(file_path)
file_name = filename(file_path)
print(file_name)


# %%今の日付 docker image 作り直したらいらんかも
#========================================
def dt_now():
    dt_now = dt.datetime.now()
    return dt_now

#時間がずれていた時
dt_now = dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))
#表示：2024年09月15日 00:51:42
print(dt_now.strftime('%Y年%m月%d日 %H:%M:%S'))


# %%
#前処理関係の定義 
#========================================

# EDA基本
df = pd.read_csv(file_path + 'train.csv')

df.info()
df.shape
df.isna().sum()

#要約統計量 平均、中央、偏差、最大、最小、最頻
df.describe().T
df.describe(include='O').T
df.describe(exclude='number')


df[['Fare']].agg(['mean']).T
df[['Fare']].agg(['mean','std','min','max']).T
df[['Fare']].agg(['dtype','count','nunique','mean','std','min','max']).T
df.agg(['dtype','count','nunique']).T

df['Sex'].value_counts()
# %%
php.ProfileReport(df)
profile_report = php.ProfileReport(df)
profile_report.to_file('report.html')

#%%
# グラフ
# =================================================

#棒グラフ（カテゴリ型）
sns.countplot(data=df,x='Sex', hue='Sex')
#ヒストグラム（連続データ）
sns.histplot(data=df,x='Age',bins=8)
#折れ線(時系列)
#関係性->散布図、相関係数、ヒートマップ
#割合-> 帯グラフ、円グラフ

# %% 
# =================================================
# 欠損値
# =================================================
df.isna().sum()
#欠損値0埋め
df['Age_fillna_0'] = df['Age'].fillna(0)
df.loc[df['Age'].isna(), ['Age', 'Age_fillna_0']].head()

# 平均値埋め
df['Age_fillna_mean'] = df['Age'].fillna(df['Age'].mean())
df.loc[df['Age'].isna(), ['Age', 'Age_fillna_mean']].head()

# 空白埋め
df['Cabin_fillna_space'] = df['Cabin'].fillna('')
df.loc[df['Cabin'].isna(), ['Cabin', 'Cabin_fillna_space']].head()

# 最頻値埋め
df['Cabin_fillna_mode'] = df['Cabin'].fillna(df['Cabin'].mode()[0])
df.loc[df['Cabin'].isna(), ['Cabin', 'Cabin_fillna_mode']]


# %% 
# =================================================
# 外れ値
# =================================================
#見つけ方
df['Age'].agg(['min', 'max'])
sns.histplot(data=df['Age'])
df['Age'].hist()
# %%
quartile = df['Age'].quantile(q=0.75) - df['Age'].quantile(q=0.25)
print('四分位範囲:',quartile )
print('下限値:',df['Age'].quantile(q=0.25) - quartile*1.5)
print('上限値:',df['Age'].quantile(q=0.75) - quartile*1.5)
# %%
df.loc[df['Age']< 0, 'Age'] = np.nan


# %%
# =================================================
# 標準化
# =================================================

value_mean = df['Fare'].mean()
value_std = df['Fare'].std(ddof=0)
# value_std = df_train['Fare'].std() #標本の標準偏差
print('mean:', value_mean, 'std:' , value_std)


df['Fare_standard'] = (df['Fare'] - value_mean) / value_std
df[['Fare','Fare_standard']].head()

#サイキットラーン使う場合
# =================================================

std = StandardScaler()
std.fit(df[['Fare']])
print('mean:', std.mean_[0], ', std:',np.sqrt(std.var_[0]))

df['Fare_standard'] = std.transform(df[['Fare']])
df[['Fare','Fare_standard']].head()

# %% 
# =================================================
# 正規化
# =================================================

value_min = df['Fare'].min()
value_max = df['Fare'].max()
print('min:', value_min, 'max:' , value_max)
df['Fare_normalize'] = (df['Fare'] - value_min) / (value_max - value_min)
df[['Fare','Fare_normalize']].head()

#サイキットラーン使う場合
mms = MinMaxScaler(feature_range=(0,1))
mms.fit(df[['Fare']])
print('min:', mms.data_min_[0], ', max:' , mms.data_max_[0])
df['Fare_normalize'] = mms.transform(df[['Fare']])
df[['Fare','Fare_normalize']].head()



# =================================================
#  カテゴリ変数をcategory型に
# =================================================
def data_pre01(df):
	for col in df.columns:
		if df[col].dtype == 'O':
			df[col] = df[col].astype('category')
	print('カテゴリ変数をcategory型に変換しました')
	df.info()
	return df



# %% 
# =================================================
# 特徴量生成
# =================================================
# =====単変数=====
# 対数変換 桁が大きい時
df['Fare_log'] = np.log(df['Fare'] + 1e-5)
df[['Fare','Fare_log']].head()

# 離散化
df['Age_bin'] = pd.cut(
  df['Age'],
  bins=[0,10,20,30,40,50,100],
  right=False,
  labels=['10代未満','10代','20代','30代','40代','50代以上'],
  include_lowest=True
)
df['Age_bin'] = df['Age_bin'].astype(str)
df[['Age','Age_bin']].head()

# 欠損があるかどうかが大事な時
df['Age_na'] = df['Age'].isna()*1
df[['Age','Age_na']].head(7)


#%% 単変数、カテゴリ変数
# =================================================

#ワンホットエンコーダー
ohe_embarled = OneHotEncoder(sparse_output = False) #sparseではエラーになった
ohe_embarled.fit(df[['Embarked']])

# for i in ohe_embarled.categories_[0]
tmp_embarked = pd.DataFrame(
  ohe_embarled.transform(df[['Embarked']]),
  columns=[f'Embarked_{i}' for i in ohe_embarled.categories_[0]]
)

df = pd.concat([df, tmp_embarked], axis=1)
df[['Embarked','Embarked_C', 'Embarked_Q','Embarked_S','Embarked_nan' ]].head()

# pandas使うバージョン  
df_ohe = pd.get_dummies(df[['Embarked', 'Sex']], dummy_na=True, drop_first=False,dtype='uint8') #dtype指定しないとTrue,Falseになる
df_ohe.head()
# %%
ce_Embarked = df['Embarked'].value_counts().to_dict()
print(ce_Embarked)

df['Embarked_ce'] = df['Embarked'].map(ce_Embarked)
df[['Embarked', 'Embarked_ce']].head()


# ラベルエンコーディング
le_embarked = LabelEncoder()
le_embarked.fit(df['Embarked'])
df['Embarked_le'] = le_embarked.transform(df['Embarked'])
df[['Embarked', 'Embarked_le']].head()

# %%
df['Embarked_na'] = df['Embarked'].isna()*1
df.loc[df['Embarked'].isna(), ['Embarked', 'Embarked_na']]


# %% 2変数 数値×数値
# =================================================

df['Sibsp_+_Parch'] = df['SibSp'] + df['Parch']
df[['SibSp','Parch','Sibsp_+_Parch']].head()

# %% 2変数 数値×カテゴリ変数
# =================================================
# カテゴリをキーにして数値を集計 平均、標準偏差、最大、最小などあと合計とか カテゴリ変数の集計は合計や平均
df_agg = df.groupby('Sex')['Fare'].agg(['mean']).reset_index()
df_agg.columns = ['Sex', 'mean_Fare_by_Sex']
print('集約テーブル')
display(df_agg)

df = pd.merge(df, df_agg, on='Sex', how='left')
print('結合後テーブル')
display(df[['Sex', 'Fare', 'mean_Fare_by_Sex']].head())


# %%
df['mean_Fare_by_Sex'] = df.groupby('Sex')['Fare'].transform('mean')
df[['Sex', 'Fare', 'mean_Fare_by_Sex']].head()


# %% 2変数 カテゴリ変数×カテゴリ変数
# =================================================
# 出現回数、出現割合、条件式
df_tbl = pd.crosstab(df['Sex'], df['Embarked'])
print('集約テーブル（クロス集計)')
display(df_tbl)
# %%
df_tbl = df_tbl.reset_index()
df_tbl
# %%
df_tbl =pd.melt(df_tbl, id_vars='Sex', value_name='count_Sex_x_Embarked')
print('集約テーブル（縦持ち返還後）')
display(df_tbl)
# %%
df = pd.merge(df, df_tbl, on=['Sex', 'Embarked'], how='left')
print('結合後テーブル')
df[['Sex', 'Embarked', 'count_Sex_x_Embarked']].head()
# %%
df['count_Sex_x_Embarked'] = df.groupby(['Sex','Embarked'])['PassengerId'].transform('count')
df[['Sex', 'Embarked', 'count_Sex_x_Embarked']].head()

# %%
df_sam = df.groupby(['Sex','Embarked'])['PassengerId'].count()
df_sam
# %% 2カテ 出現割合
# =================================================

df_tbl = pd.crosstab(df['Sex'], df['Embarked'], normalize='index')
display(df_tbl)


# %%
df_tbl = df_tbl.reset_index()
df_tbl = pd.melt(df_tbl, id_vars='Sex', value_name='rate_Sex_x_Embarked')
display(df_tbl)
# %%
df = pd.merge(df, df_tbl, on=['Sex', 'Embarked'], how='left')
df[['Sex', 'Embarked', 'rate_Sex_x_Embarked']].head()

# %%
df['Sex=male_&_Embarked=S'] = np.where((df['Sex']=='male') & (df['Embarked']=='S'), 1, 0)
df[['Sex', 'Embarked', 'Sex=male_&_Embarked=S']].head()



# %%
# =================================================
# 時系列データ
# =================================================
# ラグ、ウインドウ（移動平均）、累積
#ラグ特徴量(一個ずれる)
df1 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "weather":["晴れ","晴れ","雨","くもり","くもり","晴れ","雨","晴れ","晴れ","晴れ"],
                    })
df1['weathere_shift1'] = df1['weather'].shift(1)
df1
# %%
df1['weathere_shift1'] = df1['weathere_shift1'].interpolate(method='bfill')
df1

# %%
df2 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                            ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })
df2['date'] = pd.to_datetime(df2['date'], format='%Y-%m-%d')

df2['money_shift1'] = df2.groupby('id')['money'].shift(1)
df2
# %%
df2['date_shift1'] = df2.groupby('id')['date'].shift(1)
df2['days_elapsed'] = df2['date'] - df2['date_shift1']
df2['days_elapsed'] = df2['days_elapsed'].dt.days
df2


# %%
# ウィンドウ特徴量（移動平均）
df3 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "temperature":[8,10,12,11,9,10,12,7,9,10],
                  })
df3['temperature_window3'] = df3['temperature'].rolling(window=3).mean()
df3
# %%
df4 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                          ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })
df4['date'] = pd.to_datetime(df4['date'], format='%Y-%m-%d')
df4['money_shift'] = df4.groupby('id')['money'].transform(lambda x:x.rolling(window=2).mean())
df4


# %% 累積特徴量
df5 = pd.DataFrame({"date":pd.date_range("2021-01-01","2021-01-10"),
                    "flag_rain":[0,0,1,0,0,0,1,0,0,0],
                  })
df5['flad_rain_cumsum'] = df5['flag_rain'].cumsum()
df5

# %%
df6 = pd.DataFrame({"id":["A"]*3 + ["B"]*2 + ["C"]*4,
                    "date":["2021-04-02","2021-04-10","2021-04-25",
                            "2021-04-18","2021-04-19",
                            "2021-04-01","2021-04-04","2021-04-09","2021-04-12",
                            ],
                    "money":[1000,2000,900,4000,1800,900,1200,1100,2900],
                    })

df6['date'] = pd.to_datetime(df6['date'], format='%Y-%m-%d')
df6['money_cumsum'] = df6.groupby('id')['money'].cumsum()
df6

# %% 
# =================================================
# テキストデータ
# =================================================

from sklearn.feature_extraction.text import CountVectorizer
vec =CountVectorizer(min_df=20)

vec.fit(df['Name'])

df_name = pd.DataFrame(vec.transform(df['Name']).toarray(), columns=vec.get_feature_names_out())

print(df_name.shape)
df_name.head()
# %%
#!apt-get install -y mecab libmecab-dev mecab-ipadic-utf8
#!pip install mecab-python3
#!pip install unidic-lite
os.environ['MECABRC'] = '/etc/mecabrc'

import MeCab
# %%
print("サンプルデータ:")
df_text = pd.DataFrame({"text": [
    "今日は雨ですね。天気予報では明日も雨です。",
    "雨なので傘を持って行った方がいいです。",
    "天気予報によると明後日は晴れのようです。",
]})
display(df_text)

print("形態素解析+分かち書き:")
wakati = MeCab.Tagger("-Owakati")
df_text["text_wakati"] = df_text["text"].apply(lambda x: wakati.parse(x).replace("\n",""))
display(df_text)
# %%
print('Bowによるベクトル')
vec = CountVectorizer()
vec.fit(df_text['text_wakati'])
df_text_vec = pd.DataFrame(vec.transform(df_text['text_wakati']).toarray(),columns=vec.get_feature_names_out())

df_text_vec.head()

# %%
#====================================================
#====================================================




def data_pre(df):
  nonnull_list = []
  for col in df.columns:
    nonnull = df[col].count()
    if nonnull == 0:
      nonnull_list.append(col)
  nonnull_list


  df = df.drop(nonnull_list,axis=1)

  df = df.drop('市区町村名', axis=1)

  df = df.drop('種類',axis=1)

  dis = {
      '30分?60分':45,
      '1H30?2H':105,
      '1H?1H30':75,
      '2H?':120
  }

  df['最寄駅：距離（分）'] = df['最寄駅：距離（分）'].replace(dis).astype(float)

  df['面積（㎡）'] = df['面積（㎡）'].replace('2000㎡以上',2000).astype(float)

  y_list = {}
  for i in df['建築年'].value_counts().keys():
    if '平成' in i:
      num = float(i.split('平成')[1].split('年')[0])
      year = 36- num
    if '令和' in i:
      num = float(i.split('令和')[1].split('年')[0])
      year = 6- num
    if '昭和' in i:
      num = float(i.split('昭和')[1].split('年')[0])
      year = 99- num
    y_list[i] = year
  y_list['戦前'] = 79
  df['建築年'] = df['建築年'].replace(y_list)

  
  year = {
      '年第１四半期':'.25',
      '年第２四半期':'.50',
      '年第３四半期':'.75',
      '年第４四半期':'.99',
  }
  year_list = {}
  for i in df['取引時点'].value_counts().keys():
    for k,j in year.items():
      if k in i:
        year_rep = i.replace(k,j)
      year_list[i] = year_rep
  df['取引時点'] = df['取引時点'].replace(year_list).astype(float) 

  for col in ['都道府県名','地区名','最寄駅：名称','間取り','建物の構造','今後の利用目的','用途','都市計画','改装','取引の事情等']:
    df[col] = df[col].astype('category')

  return df



def model_lgb(df):
  df_train,df_val = train_test_split(df,test_size=0.2)

  col = '取引価格（総額）_log'
  train_y = df_train[col]
  train_x = df_train.drop(col,axis=1)

  col = '取引価格（総額）_log'
  val_y = df_val[col]
  val_x = df_val.drop(col,axis=1)

  trains = lgb.Dataset(train_x, train_y)
  valids = lgb.Dataset(val_x, val_y)

  params = {
      'objective':'regression',
      'metrics':'mae'
  }

  model = lgb.train(params,trains,valid_sets=valids,num_boost_round=1000,callbacks=[lgb.early_stopping(stopping_rounds=100)])

  return model

