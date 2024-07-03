export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_API_KEY=YOUR_WANDB_API_KEY

export MODEL_NAME=tinyllama_1B_n$1
export WANDB_NAME=$MODEL_NAME
export NUMBER_OF_GPUS=8
# you can specify the config name here or pass it as an argument $1
export CONFIG_NAME=regmix

lightning run model \
    --node-rank=0  \
    --main-address=127.0.0.1 \
    --accelerator=cuda \
    --num-nodes=1 \
    --devices=$NUMBER_OF_GPUS \
    pretrain/tinyllama.py --devices $NUMBER_OF_GPUS \
    --train_data_dir lit_dataset_regmix \
    --val_data_dir lit_dataset_regmix \
    --data_yaml_file ../mixture_config/config_1b/$CONFIG_NAME.yaml \
    --out_name $MODEL_NAME \
    --resume True
