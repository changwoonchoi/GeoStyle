#!/bin/bash

export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export INSTANCE_DIR="/path/to/input/images"
export OUTPUT_DIR="/path/to/output/folder"

python ./scripts/train_dreambooth_lora_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --repeats 1 \
  --output_dir=$OUTPUT_DIR \
  --train_text_encoder \
  --instance_prompt="A photo of TOK sculpture" \
  --resolution=512 \
  --train_batch_size=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --num_class_images=200 \
  --rank 16 \
  --learning_rate=1e-4 \
  --gradient_accumulation_steps=1 \
  --gradient_checkpointing \
  --max_train_steps=800

  