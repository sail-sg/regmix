export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_API_KEY=YOUR_WANDB_API_KEY

export MODEL_NAME=tinyllama_1M_n$1
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPUS=1
# you can specify the config index here or pass it as an argument
export CONFIG_INDEX=$1

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPUS \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPUS \
    --train_data_dir lit_dataset_regmix \
    --val_data_dir lit_dataset_regmix \
    --data_yaml_file ../mixture_config/config_1m/n$CONFIG_INDEX.yaml \
    --out_name $MODEL_NAME \
    --resume True
