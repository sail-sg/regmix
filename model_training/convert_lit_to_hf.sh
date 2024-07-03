#!/bin/bash

# tinyllama_1_1b, tinyllama_1M, more can be found in lit_gpt/config.py
export ARCH_NAME="tinyllama_1_1b"
export INP_FOLDER="lit_checkpoint_folder"
export FILE_NAME="iter-025000-ckpt.pth"
export OUT_FOLDER="converted_huggingface_model_folder"

python convert_lit_checkpoint.py --checkpoint_name "$FILE_NAME" --inp_dir "$INP_FOLDER" --out_dir "$OUT_FOLDER" --model_name $ARCH_NAME