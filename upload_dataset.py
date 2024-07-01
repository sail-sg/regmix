from huggingface_hub import HfApi
api = HfApi()

# Upload all the content from the local folder to your remote Space.
# By default, files are uploaded at the root of the repo
api.upload_folder(
    folder_path="/home/aiops/liuqian/TinyLlama-Data/lit_dataset_pile",
    repo_id="sail/regmix-data",
    repo_type="dataset",
    token="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"
)
