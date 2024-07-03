import pandas as pd
from copy import copy
import yaml
import os

def read_config(config_file):
    # read the yaml config
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        new_config = {}
        train_keys = list(config["train"].keys())
        for key in train_keys:
            # remove train_doremi_sample prefix
            if key.startswith("train_"):
                new_config[key] = config["train"][key]

        flatten_dict = {}
        for key, value in new_config.items():
            if type(value) == float:
                flatten_dict[key] = round(value, 5)
            if type(value) == int:
                flatten_dict[key] = value
    return flatten_dict


def gather_mixture_data(write_file_path, config_folder):
    # read all files in the config folder
    output_dict = {}
    for file_path in os.listdir(config_folder):
        # only read yaml files
        if not file_path.endswith(".yaml"):
            print("skip", file_path)
            continue
        full_path = os.path.join(config_folder, file_path)
        # index name is the file path remove the prefix "n"
        index_name = int(file_path.split(".")[0].replace("n", ""))
        config = read_config(full_path)
        # only the train part is valid
        output_dict[index_name] = config
    # convert the dict to dataframe
    df = pd.DataFrame(output_dict).T
    # the index column is the index name
    df.index.name = "index"
    # order by index name
    df = df.sort_index()
    df.to_csv(write_file_path)

if __name__ == "__main__":
    gather_mixture_data("train_mixture_1m.csv", "../mixture_config/config_1m")