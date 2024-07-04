# Model Training

This document describes how to train a model with the awesome [TinyLlama](https://github.com/jzhang38/TinyLlama) codebase. And a large part of this README is borrowed from the TinyLlama README.

## Setup

You can choose to install the dependencies manually or use the provided Docker image. The Docker image is recommended for a hassle-free setup. If you choose to install the dependencies manually, please follow the instructions below.

### Installation Manually

You should install PyTorch with CUDA support, along with xformers and Flash-Attention 2.

#### Prerequisites

- CUDA 11.8 installed on your system

#### 1. Install PyTorch Nightly

First, you should install the nightly build of PyTorch with CUDA support:

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

#### 2. Install Xformers

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu118
```

#### 3. Install Flash-Attention 2

Next, we'll install Flash-Attention 2 and its associated operators:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && \
    python setup.py install && \
    cd csrc/rotary && pip install . && \
    cd ../layer_norm && pip install . && \
    cd ../xentropy && pip install . && \ 
    cd ../.. && rm -rf flash-attention
```

#### 4. Install Remaining Dependencies

Finally, install the remaining required packages:

```bash
pip install -r requirements.txt tokenizers sentencepiece transformers wandb datasets huggingface_hub
```

> [!NOTE]
> The build process for Flash-Attention may take 5 minutes or more.
> If you encounter any issues, ensure your CUDA installation is correct and compatible with the PyTorch version you're installing.
> It's recommended to use a virtual environment for this installation to avoid conflicts with other Python packages.

If you need any further assistance or encounter any issues during the installation, please don't hesitate to ask for help.

### Using Docker

You can also use the following command to pull the already built Docker:

```shell
docker pull siviltaramqian/tinyllama:latest
```

And you may run the docker image along with the following command:

```shell
docker run --gpus all -it --rm siviltaramqian/tinyllama:latest ./pretrain_tinyllama_1m.sh 1
```

## Wandb Integration

By default we use the wandb for collecting the data to avoid saving massive small models and logs on the local machine. If you want to use the wandb, you need to create an account on the [wandb](https://wandb.ai/site) and get the API key. Then you should set the following environment variable in both `pretrain_tinyllama_1m.sh` and `pretrain_tinyllama_1b.sh`:

```shell
# wandb project name, entity, and API key
export WANDB_PROJECT=YOUR_PROJECT_NAME
export WANDB_ENTITY=YOUR_WANDB_ENTITY
export WANDB_API_KEY=YOUR_WANDB_API_KEY
```

## Preprocess

Before training the model, you need to preprocess the data. We provide the easy-to-use script for preprocessing the data. You can use the following command to preprocess the data:

```shell
cd preprocess
bash run_preprocess.sh
```

By default you will first download the `regmix-data-sample` from the HuggingFace and then preprocess the data. The JSONL data will be saved in the `preprocess/sail/regmix-data-sample` directory, and the preprocessed data will be saved in the `lit_dataset_regmix` directory.

## Train

After preprocessing the data, you can train the model using the following command:

```shell
./pretrain_tinyllama_1m.sh 1
```

The passed argument is the configuration index. After the setup described in [mixture_config](../mixture_config/), you should have 512 configurations for training the proxy models. You can change the configuration index to train different configurations. The training of 1M models should take around 20 minutes to finish on a single A100 GPU, which is pre-trained on 1B tokens.

You can also train a larger model using the following command:

```shell
./pretrain_tinyllama_1b.sh regmix
```

The regmix is the configuration name for the 1B model. The full configuration is stored in the [mixture_config/config_1b](../mixture_config/config_1b/) directory. The training of 1B models should take around 2 days to finish on 8 x A100 GPU, which is pre-trained on 25B tokens.
