import pandas as pd 
import wandb
from tqdm import tqdm
import yaml
import os
from copy import copy

WANDB_API_KEY = "YOUR_API_KEY"
RPOJECT_NAME = "siviltaram/Qian-Sen-Training-1M-FineWeb-Fix-v2"
# the folder which contains the config files
CONFIG_FOLDER = "../model_training/data_mixture_1m"

SELECT_STEP = 1000
KEY_METRICS = ["metric/the_pile_unzip_pile_cc_val_loss",
               "metric/train_loss"]

RUN_NAME_PREFIX = "tinyllama_1M_model_fineweb_n"
PREFIX = "train_fineweb_"

# Please fill in your own API key
api = wandb.Api(api_key=WANDB_API_KEY)

# Project is specified by <entity/project-name>
runs = api.runs(RPOJECT_NAME, per_page=20)

def read_config(config_file):
    # read the yaml config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        for key in copy(list(config["train"].keys())):
            # remove train_doremi_sample prefix
            if key.startswith(PREFIX):
                config[key.replace(PREFIX, "")] = config["train"][key]
        del config["train"]

        flatten_dict = {}
        for key, value in config.items():
            if type(value) == float:
                flatten_dict[key] = round(value, 5)
            if type(value) == int:
                flatten_dict[key] = value
    return flatten_dict


def export_wandb_runs(write_file_path, config_folder, enable_empty_row=False):
    output_csv_list = []
    records = set()
    for run in tqdm(runs):
        if not run.name.startswith(RUN_NAME_PREFIX):
            print("skip", run.name)
            continue
        # .summary contains the output keys/values for metrics like accuracy.
        #  We call ._json_dict to omit large files 
        data_frame = run.history(samples=10000000)
        keep_columns = [col for col in data_frame.columns if col in KEY_METRICS + ["trainer/global_step"]]
        # only keep the pre-defined columns
        data_frame = data_frame[keep_columns]
        # select the row when global_step = SELECT_STEP
        data_frame = data_frame[data_frame["trainer/global_step"] == SELECT_STEP]
        # take the first non-nan value for each column
        first_non_nan_indices = data_frame.apply(lambda col: col.first_valid_index())
        if len(data_frame) == 0 or "NaN" in str(first_non_nan_indices):
            if enable_empty_row:
                # add a row of nan if no row is selected
                data_frame = pd.DataFrame([[float("nan") for _ in range(len(keep_columns))]], columns=keep_columns)
                data_frame["trainer/global_step"] = SELECT_STEP
            else:
                print("skip", run.name)
                continue
        else:
            new_df = pd.DataFrame({col: [data_frame[col][idx]] for col, idx in first_non_nan_indices.items()})
            data_frame = new_df
        
        if config_folder is not None and os.path.exists(config_folder):
            print("config_folder", config_folder)
            # get the config file path
            config_file = os.path.join(config_folder, run.name.replace(RUN_NAME_PREFIX, "") + ".yaml")
            # read the yaml config
            config = read_config(config_file)
            # merge to data_frame
            for key, value in config.items():
                data_frame[key] = value
        else:
            print("config_folder does not exist")
        
        # no duplicated run name
        if run.name in records:
            continue
        data_frame["model"] = run.name
        output_csv_list.append(data_frame.to_dict("records")[0])
        records.add(run.name)
    
    runs_df = pd.DataFrame.from_dict(output_csv_list)
    # delete global_step, model, and train_loss columns
    runs_df = runs_df.drop(columns=["trainer/global_step", "metric/train_loss"])
    runs_df.to_csv(write_file_path)
    expected_model_list = [RUN_NAME_PREFIX + "n{}".format(i) for i in range(1, 513)]
    exported_model_list = runs_df["model"].tolist()
    # check if all models are exported
    print("missing models", set(expected_model_list) - set(exported_model_list))


if __name__ == "__main__":
    export_wandb_runs("validation_loss_1m.csv", CONFIG_FOLDER, enable_empty_row=False)