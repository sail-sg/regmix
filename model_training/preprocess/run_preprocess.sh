python download_data.py --dataset_name sail/regmix-data-sample
# WARNING: you can choose to download the full dataset (around 1TB) by running the following command
# python download_data.py --dataset_name sail/regmix-data

# We use gptneox tokenizer for the dataset to be consistent with the flagship method DoReMi
python prepare_file_domain.py --source_path sail/regmix-data-sample --tokenizer_path tokenizer/gptneox --destination_path ../lit_dataset_regmix --short_name the_pile --split train

# 131136 = 2049 * 64, which means the chunk size is relatively smaller due to the size of the validation set, especially for low-resource domains
python prepare_file_domain.py --source_path sail/regmix-data-sample --tokenizer_path tokenizer/gptneox --destination_path ../lit_dataset_regmix --short_name the_pile --split valid --chunk_size 131136