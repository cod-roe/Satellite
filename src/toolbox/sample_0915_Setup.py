# %% [markdown]
## セットアップ！
#=================================================
#ファイルのインプット、フォルダの作成

#%%
import os
import json
import warnings
import shutil
import logging
import joblib
import random
import datetime
import sys
import json
import zipfile


# %%
#Config
# =================================================

######################
# serial #
######################
serial_number = 1 #スプレッドシートAの番号

######################
# Data #
######################
comp_name = 'Titanic'

# /tmp/work/src/exp/tamesi.py' 
abs_path = os.path.abspath(__file__)

# name = 'run001'  #Google Colab用
name = os.path.splitext(os.path.basename(abs_path))[0]
######################
# 学習するか推論だけするか #
######################
only_inference = False #学習する False，推論だけ実行 True



#%% フォルダ作成
#フォルダ作成
if "google.colab" in sys.modules:
    print("This environment is Google Colab")

    # import library
    ! pip install japanize-matplotlib # type:ignore

    # use kaggle api (need kaggle token)
    api_path = "/content/drive/MyDrive/Python/Kaggle/kaggle.json"
    f = open(api_path, 'r')
    json_data = json.load(f)
    os.environ['KAGGLE_USERNAME'] = json_data['username']
    os.environ['KAGGLE_KEY'] = json_data['key']

    # set dirs
    DRIVE =  f'/content/drive/MyDrive/Python/Kaggle/{comp_name}'
    EXP = name
    INPUT = os.path.join(DRIVE, "input")
    INPUT_PATH = os.path.join(INPUT, comp_name)
    OUTPUT = os.path.join(DRIVE, "output")
    OUTPUT_EXP = os.path.join(OUTPUT, EXP)
    EXP_MODEL = os.path.join(OUTPUT_EXP, "model")

    # make dirs
    for d in [ INPUT_PATH,  EXP_MODEL]:
        os.makedirs(d, exist_ok=True)

    if len(os.listdir(INPUT_PATH)) == 0:
        # load dataset
        !kaggle competitions download titanic -p $INPUT_PATH # type:ignore
        # 解凍するZIPファイルのパス
        zip_file_path = f'{INPUT_PATH}/titanic.zip'
    
        # ZIPファイルを開いて解凍
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(INPUT_PATH)

        print(f"ZIPファイルが {INPUT_PATH} に解凍されました。")
    
    for dirname, _, filenames in os.walk(INPUT_PATH):
        for i, datafilename in enumerate(filenames):
            # print(os.path.join(dirname,filename))
            print('='*20)
            print(i,datafilename)


elif 'vscode' in sys.modules:
    print("This environment is Vs Code")


    # use kaggle api (need kaggle token)
    api_path = "/tmp/work/kaggle.json"
    f = open(api_path, 'r')
    json_data = json.load(f)
    os.environ['KAGGLE_USERNAME'] = json_data['username']
    os.environ['KAGGLE_KEY'] = json_data['key']

    # set dirs
    DRIVE =  os.path.dirname(os.getcwd())
    EXP = name
    INPUT_PATH = f'../input/{comp_name}/'
    OUTPUT = os.path.join(DRIVE, "output")
    OUTPUT_EXP = os.path.join(OUTPUT, EXP) 
    EXP_MODEL = os.path.join(OUTPUT_EXP, "model")

    # make dirs
    for d in [ INPUT_PATH, OUTPUT]:
        os.makedirs(d, exist_ok=True)

    if len(os.listdir(INPUT_PATH)) == 0:
        # load dataset
        !kaggle competitions download titanic -p $INPUT_PATH # type:ignore
        # 解凍するZIPファイルのパス
        zip_file_path = f'{INPUT_PATH}/titanic.zip'
        
        # ZIPファイルを開いて解凍
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(INPUT_PATH)

        print(f"ZIPファイルが {INPUT_PATH} に解凍されました。")
        
    for dirname, _, filenames in os.walk(INPUT_PATH):
        for i, datafilename in enumerate(filenames):
            # print(os.path.join(dirname,filename))
            print('='*20)
            print(i,datafilename)

#%%

