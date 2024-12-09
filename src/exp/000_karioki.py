
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
