from huggingface_hub import snapshot_download
from argparse import ArgumentParser

# Use argparse to handle command line arguments
parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="Dataset name to download", default="regmix-data-sample",
                    choices=["regmix-data", "regmix-data-sample"])
args = parser.parse_args()

# You can choose to download regmix-data, or regmix-data-sample
LOCAL_DIR = args.dataset_name
snapshot_download(repo_id=args.dataset_name, 
                  repo_type='dataset',
                  local_dir=LOCAL_DIR,
                  local_dir_use_symlinks=False)
