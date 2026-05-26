#!/bin/bash

export LORA_DIR="/path/to/lora/weight"
export OUTPUT_DIR="/path/to/output/folder"

python ./scripts/inference_dreambooth_lora_sdxl.py \
  --prompt="A TOK style sculpture" \
  --output_path=$OUTPUT_DIR \
  --LoRA=$LORA_DIR \
  --lora_scale 1 \
  --num_images_per_prompt 5

