import os
import yaml
import argparse
import pandas as pd


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def get_data(config_path):
    config = read_params(config_path)
    data_path = config["data_source"]["remote_source"]
    df = pd.read_csv(data_path, sep=",", encoding="utf-8")
    new_col = [col.replace(" ", "_") for col in df.columns]
    df.to_csv("data/raw/data.csv", index=False, header=new_col)
    return df


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    param_path = os.path.join("reports", "params.yaml")
    args.add_argument("--config", default=param_path)
    parsed_args = args.parse_args()
    get_data(config_path=parsed_args.config)
