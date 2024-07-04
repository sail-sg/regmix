import pandas as pd 
import wandb
from tqdm import tqdm
import yaml
import os
from copy import copy
import argparse

# find your API key at https://wandb.ai/authorize
WANDB_API_KEY = "YOUR_API_KEY"
# Project is specified by <entity/project-name>
RPOJECT_NAME = "YOUR_PROJECT_NAME"

# by default we only take the pile cc val loss, and you can also choose other as the target
KEY_METRICS = ["metric/the_pile_pile_cc_val_loss",
               "metric/train_loss"]

# this is the prefix for the wandb runs
RUN_NAME_PREFIX = "tinyllama_1M_n"

# 1000 step corresponds to the 1B token
SELECT_STEP = 1000

# Please fill in your own API key
api = wandb.Api(api_key=WANDB_API_KEY)
runs = api.runs(RPOJECT_NAME, per_page=20)


def export_wandb_runs(write_file_path, enable_empty_row=False):
    output_data = []
    records = set()
    for run in tqdm(runs):
        # skip invalid runs
        if not run.name.startswith(RUN_NAME_PREFIX):
            print("skip", run.name)
            continue

        run_index = int(run.name.replace(RUN_NAME_PREFIX, ""))
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
                
        # no duplicated run name
        if run.name in records:
            continue

        data_frame["index"] = run_index
        # set index as the index column
        output_data.append(data_frame.to_dict("records")[0])
        records.add(run.name)
    
    runs_df = pd.DataFrame.from_dict(output_data)
    # set the index column as the index
    runs_df.set_index("index", inplace=True)
    # order by index
    runs_df = runs_df.sort_index()
    
    # delete global_step, model, and train_loss columns
    runs_df = runs_df.drop(columns=["trainer/global_step"])
    runs_df.to_csv(write_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write_file_path", type=str, default="train_pile_loss_1m.csv")

    args = parser.parse_args()
    write_file_path = args.write_file_path
    export_wandb_runs(write_file_path, enable_empty_row=False)