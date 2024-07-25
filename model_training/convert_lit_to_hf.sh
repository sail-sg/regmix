#!/bin/bash

# tinyllama_1_1b, tinyllama_1M, more can be found in lit_gpt/config.py
export ARCH_NAME="tinyllama_1_1b"
export INP_FOLDER="lit_checkpoint_folder"
export FILE_NAME="iter-025000-ckpt.pth"
export OUT_FOLDER="converted_huggingface_model_folder"

# convert the model into Huggingface compatible format, and save the config.json
python convert_lit_checkpoint.py --checkpoint_name "$FILE_NAME" --inp_dir "$INP_FOLDER" --out_dir "$OUT_FOLDER" --model_name $ARCH_NAME

# copy tokenizer and config into the new folder
# WARNING: if you use a different tokenizer, you need to modify the folder
cp -r preprocess/tokenizer/gptneox/* "$OUT_FOLDER"