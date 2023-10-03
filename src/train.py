import pandas as pd
import argparse
from get_data import read_params
from sklearn.linear_model import ElasticNet
import joblib
import logging
import json
import os

# logging_str = "[%(asctime)s: %(levelname)s: %(module)s]: %(message)s"
# logging.basicConfig(level=logging.DEBUG, format=logging_str)


def save_reports(filepath: str, report: dict):
    with open(filepath, "w") as f:
        json.dump(report, f, indent=4)
    logging.info(f"details of the report: {report}")
    logging.info(f"reports saved at {filepath}")


def train(config_path):
    config = read_params(config_path)

    processed_data_dir = config["split_data"]["processed_data_dir"]
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]

    random_seed = config["base"]["random_state"]
    target = config["base"]["target_col"]

    reports_dir = config["reports"]["reports_dir"]
    params_file = config["reports"]["params"]

    ElasticNet_params = config["estimators"]["ElasticNet"]["params"]
    alpha = ElasticNet_params["alpha"]
    l1_ratio = ElasticNet_params["l1_ratio"]
    target = config["base"]["target_col"]

    train = pd.read_csv(train_data_path, sep=",")

    train_y = train[target]
    train_x = train.drop(target, axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_seed)
    lr.fit(train_x, train_y)

    model_dir = config["models"]["model_dir"]
    model_path = config["models"]["model_path"]

    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio,
    }

    save_reports(params_file, params)

    joblib.dump(lr, model_path)

    logging.info(f"model saved at {model_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    param_path = os.path.join("reports", "params.yaml")
    args.add_argument("--config", default=param_path)
    parsed_args = args.parse_args()

    try:
        data = train(config_path=parsed_args.config)
        logging.info("training stage completed")

    except Exception as e:
        logging.error(e)
        # raise e
