
import os
import warnings

warnings.filterwarnings("ignore")

os.chdir("../../..")

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import pandas as pd
import torch

from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.data.examples import generate_ar_data
from pytorch_forecasting.metrics import MAE, SMAPE, MultivariateNormalDistributionLoss

#%%
data = generate_ar_data(seasonality=10.0, timesteps=400, n_series=100, seed=42)
data["static"] = 2
data["date"] = pd.Timestamp("2020-01-01") + pd.to_timedelta(data.time_idx, "D")
data.head()


#%%
data = data.astype(dict(series=str))
#%%
data.info()

#%%
# create dataset and dataloaders
max_encoder_length = 60
max_prediction_length = 20

training_cutoff = data["time_idx"].max() - max_prediction_length

context_length = max_encoder_length
prediction_length = max_prediction_length

training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="value",
    categorical_encoders={"series": NaNLabelEncoder().fit(data.series)},
    group_ids=["series"],
    static_categoricals=[
        "series"
    ],  # as we plan to forecast correlations, it is important to use series characteristics (e.g. a series identifier)
    time_varying_unknown_reals=["value"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
batch_size = 128
# synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
train_dataloader = training.to_dataloader(
    train=True, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False, batch_size=batch_size, num_workers=0, batch_sampler="synchronized"
)

#%%
# calculate baseline absolute error
baseline_predictions = Baseline().predict(val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True)
SMAPE()(baseline_predictions.output, baseline_predictions.y)

#%%
pl.seed_everything(42)
import pytorch_forecasting as ptf

trainer = pl.Trainer(accelerator="cpu", gradient_clip_val=1e-1)
net = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=MultivariateNormalDistributionLoss(rank=30),
    optimizer="Adam",
)


#%%
# find optimal learning rate
from lightning.pytorch.tuner import Tuner

res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
net.hparams.learning_rate = res.suggestion()



#%%
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)


net = DeepAR.from_dataset(
    training,
    learning_rate=1e-2,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=MultivariateNormalDistributionLoss(rank=30),
)

trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
# %%
# 正義
# =============ここから=============
# %%
data2 = {
    "date": pd.date_range(start="2013-09-01", periods=38400, freq="h"),
    "station_id": 1 * 38400,
    "bikes_available": [10 + i % 5 for i in range(38400)],
}


data = pd.DataFrame(data2)
data["time_idx"] = range(len(data))
data["bikes_available"] = data["bikes_available"].astype(float)
data["station_id"] = data["station_id"].astype(str)

# %%
data = data.astype(dict(station_id=str))
data.info()
# %%
data.head()
# %%
# create dataset and dataloaders
max_encoder_length = 72
max_prediction_length = 24
# 訓練データのカットオフ
# training_cutoff2 = data["date"].max() - pd.Timedelta(days=3)
training_cutoff = data["time_idx"].max() - max_prediction_length #1週間をバリでにしたい168時間24*7

context_length = max_encoder_length
prediction_length = max_prediction_length
#%%
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="bikes_available",
    categorical_encoders={"station_id": NaNLabelEncoder().fit(data.station_id)},
    group_ids=["station_id"],
    static_categoricals=[
        "station_id"
    ],  # as we plan to forecast correlations, it is important to use station_id characteristics (e.g. a station_id identifier)
    time_varying_unknown_reals=["bikes_available"],
    max_encoder_length=context_length,
    max_prediction_length=prediction_length,
)

validation = TimeSeriesDataSet.from_dataset(
    training, data, min_prediction_idx=training_cutoff + 1
)
batch_size = 64

# synchronize samples in each batch over time - only necessary for DeepVAR, not for DeepAR
train_dataloader = training.to_dataloader(
    train=True,
    batch_size=batch_size,
    num_workers=0,  # batch_sampler="synchronized"
)
val_dataloader = validation.to_dataloader(
    train=False,
    batch_size=batch_size,
    num_workers=0,  # batch_sampler="synchronized"
)

# %%
# calculate baseline absolute error
baseline_predictions = Baseline().predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)
SMAPE()(baseline_predictions.output, baseline_predictions.y)

# %%
pl.seed_everything(123)


trainer = pl.Trainer(
    accelerator="cpu",
)  # gradient_clip_val=1e-1
net = DeepAR.from_dataset(
    training,
    learning_rate=3e-2,
    hidden_size=30,
    rnn_layers=2,
    loss=NormalDistributionLoss(),
    optimizer="Adam",
)


# %%
# find optimal learning rate

res = Tuner(trainer).lr_find(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
    min_lr=1e-5,
    max_lr=1e0,
    early_stop_threshold=100,
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()
net.hparams.learning_rate = res.suggestion()

# %%
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)
# トレーナーの設定
trainer = pl.Trainer(
    max_epochs=30,
    accelerator="cpu",
    enable_model_summary=True,
    # gradient_clip_val=0.1,
    callbacks=[early_stop_callback],
    limit_train_batches=50,
    enable_checkpointing=True,
)

# モデルのインスタンス化
net = DeepAR.from_dataset(
    training,
    learning_rate=0.03981071705534971,
    log_interval=10,
    log_val_interval=1,
    hidden_size=30,
    rnn_layers=2,
    optimizer="Adam",
    loss=NormalDistributionLoss(),
)

# モデルのトレーニング
trainer.fit(
    net,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
# %%
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = DeepAR.load_from_checkpoint(best_model_path)
# %%
# best_model = net
predictions = best_model.predict(
    val_dataloader, trainer_kwargs=dict(accelerator="cpu"), return_y=True
)
MAE()(predictions.output, predictions.y)
# %%
raw_predictions = net.predict(
    val_dataloader,
    mode="raw",
    return_x=True,
    n_samples=100,
    trainer_kwargs=dict(accelerator="cpu"),
)
# %%
station_id = validation.x_to_index(raw_predictions.x)["station_id"]
for idx in range(1):  # plot 10 examples
    best_model.plot_prediction(
        raw_predictions.x, raw_predictions.output, idx=idx, add_loss_to_title=True
    )
    plt.suptitle(f"station_id: {station_id.iloc[idx]}")

# ===========ここまで======================