# Config file for WanDB sweeps
from typing import Dict

"""example of a config dictionary for EU wind dataset where I'm trying to forecast the wind energy for Austria"""


def make_config_file(file_path: str, df_len: int) -> Dict:
    train_number: float = df_len * .7
    validation_number: float = df_len * .9
    config_default = {
        "model_name": "DecoderTransformer",
        "model_type": "PyTorch",
        "takes_target": False, # this param appears to mean that the model forward pass takes in the target (not suited for DecoderTransformer")
        "model_params": {
            "n_time_series": 30,
            "n_head": 8,
            "forecast_history": 90,
            "n_embd": 1,
            "num_layer": 5,
            "dropout": 0.1,
            "q_len": 1,
            "scale_att": False,
            "forecast_length": 30,
            "additional_params": {}
        },
        "dataset_params":
            {
                "class": "default",
                "training_path": file_path,
                "validation_path": file_path,
                "test_path": file_path,
                "batch_size": 64,
                "forecast_history": 90,
                "forecast_length": 30,
                "train_end": int(train_number),
                "valid_start": int(train_number + 1),
                "valid_end": int(validation_number),
                "target_col": ['Austria'],
                "relevant_cols": ['Austria', 'Belgium', 'Bulgaria', 'Switzerland', 'Czechia',
                                  'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'Greece',
                                  'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania', 'Luxembourg',
                                  'Latvia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
                                  'Slovenia', 'Slovakia', 'Sweden', 'United Kingdom', 'month', 'weekday'],
                "scaler": "StandardScaler",
                "interpolate": False,
                "sort_column": "time",
            },
        "training_params":
            {
                "criterion": "DilateLoss",
                "optimizer": "Adam",
                "optim_params":
                    {
                    },
                "lr": 0.001,
                "epochs": 2,
                "batch_size": 64
            },
        "early_stopping": {
            "patience": 3
        },
        "GCS": False,
        "sweep": False,
        "wandb": False,
        "forward_params": {},
        "metrics": ["DilateLoss"],
        "inference_params":
            {
                "datetime_start": "2010-01-01",
                "hours_to_forecast": 2000,
                "test_csv_path": file_path,
                "decoder_params": {
                    "decoder_function": "simple_decode",
                    "unsqueeze_dim": 1
                },
                "dataset_params": {
                    "file_path": file_path,
                    "forecast_history": 90,
                    "forecast_length": 30,
                    "relevant_cols": ['Austria', 'Belgium', 'Bulgaria', 'Switzerland', 'Czechia',
                                      'Germany', 'Denmark', 'Estonia', 'Spain', 'Finland', 'France', 'Greece',
                                      'Croatia', 'Hungary', 'Ireland', 'Italy', 'Lithuania', 'Luxembourg',
                                      'Latvia', 'Netherlands', 'Norway', 'Poland', 'Portugal', 'Romania',
                                      'Slovenia', 'Slovakia', 'Sweden', 'United Kingdom', 'month', 'weekday'],
                    "target_col": ['Austria'],
                    "scaling": "StandardScaler",
                    "interpolate_param": False
                }
            },
    }

    return config_default
