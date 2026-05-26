

# Running the Code

## Data Download

You can access the source meshes, trained LoRA weights, images used for LoRA training, and segmentation maps for each source mesh [here](https://drive.google.com/file/d/1d30j6ngOTjvUllBsTq-45FHFlwoQ2RaO/view?usp=sharing), which the authors used to produce the paper's results.




## LoRA Training & Inference

To [train LoRA weights using DreamBooth](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora), you can use the script provided in `configs/dreambooth_lora/train.sh`.

    sh ./configs/dreambooth_lora/train.sh


To generate images using the pretrained LoRA weights, you can use the script provided in `configs/dreambooth_lora/inference.sh`.

    sh ./configs/dreambooth_lora/inference.sh



## Mesh Segmentation


We use [PartField](https://github.com/nv-tlabs/PartField) for mesh segmentation. 
Please follow the steps described in PartField's official repository.



## Mesh Deformation

You can try various deformations by modifying the config files that are provided in the `configs` folder.



To deform meshes using the SDS loss with SDXL, we first train an approximated VAE encoder.
If you do not specify `--approx_encoder_path`, (*e.g.*, set `approx_encoder_path: null` in the config file), our code will automatically train the encoder matrix first and resume the optimization process with the fitted encoder matrix. 
In short, just run the following command:

    python scripts/main.py --config ./configs/deformation/hand_maman.yml


If you already have the approximated VAE encoder, you don't need to fit the encoder matrix again. 
Instead, specify the path to the approximated VAE encoder (`img2latent_aug.pt`) in the config file.

Replace `--mesh` and `--lora_model` with your own paths to try different stylizations.
