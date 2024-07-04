from huggingface_hub import snapshot_download
from argparse import ArgumentParser

# Use argparse to handle command line arguments
parser = ArgumentParser()
parser.add_argument("--dataset_name", type=str, required=True, 
                    help="Dataset name to download", default="sail/regmix-data-sample",
                    choices=["sail/regmix-data-sample", "sail/regmix-data-sample"])
args = parser.parse_args()

# You can choose to download regmix-data, or regmix-data-sample
snapshot_download(repo_id=args.dataset_name, 
                  repo_type='dataset',
                  local_dir=args.dataset_name,
                  local_dir_use_symlinks=False)
