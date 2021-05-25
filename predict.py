from flood_forecast.config_sample import make_config_file_infer
from flood_forecast.deployment.inference import InferenceMode
from datetime import datetime
from typing import Dict, List
import os
import pandas as pd
from pandas import DataFrame, Series
import wandb
import numpy as np


def predict(forecast_history: int, forecast_length: int, file_path: str, ckpt_path: str, start_pred_date: datetime) -> DataFrame:
    """predict using a portion of wind_train.csv"""
    full_len: int = len(pd.read_csv(file_path))
    inf = InferenceMode(forecast_steps=forecast_length, num_prediction_samples=1,
                        model_params=make_config_file_infer(file_path, full_len),
                        csv_path=file_path, weight_path=os.path.join(ckpt_path, 'checkpoint.pth'), wandb_proj=None,
                        torch_script=False)
    df, tensor, history, forecast_start, test, samples = inf.infer_now(some_date=start_pred_date,
                                                                       csv_path=file_path, save_buck=None,
                                                                       save_name=None, use_torch_script=False)

    return df[['datetime', 'preds']]


if __name__ == "__main__":
    ckpt_path: str = r'C:\Users\Lorenzo\PycharmProjects\flow-forecast\notebooks'
    file_path: str = r'C:\Users\Lorenzo\PycharmProjects\flow-forecast\data\wind_train.csv'
    forecast_history: int = 120
    forecast_length: int = 18
    start_pred_date = datetime(2013, 5, 3)
    df_preds: DataFrame = predict(forecast_history, forecast_length, file_path, ckpt_path, start_pred_date)
    # print(df_preds.head(10))
    print(df_preds.tail(18))

