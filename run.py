from flood_forecast.trainer import train_function
import argparse
from argparse import Namespace, ArgumentParser
import pandas as pd
from flood_forecast.config_sample import make_config_file
import wandb
from wandb.wandb_run import Run


def main():
    """
    Main function which is called from the command line. Entrypoint for training all ML models.
    """
    parser: ArgumentParser = argparse.ArgumentParser(description="Argument parsing for training and eval")
    parser.add_argument("-p", "--params", help="Path to model config file")
    args: Namespace = parser.parse_args()
    # with open(args.params) as f:
    #     training_config = json.load(f)
    # train_function(training_config["model_type"], training_config)
    file_path: str = 'data/wind_train.csv'
    full_len: int = len(pd.read_csv(file_path))
    run: Run = wandb.init(project="pretrained-wind-updated")
    train_function("PyTorch", make_config_file(file_path, full_len))
    # evaluate_model(trained_model)
    print("Process is now complete.")


if __name__ == "__main__":
    main()
