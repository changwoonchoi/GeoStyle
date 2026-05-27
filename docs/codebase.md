# Running the Code

## 💾 Data Download

You can access the source meshes, trained LoRA weights, images used for LoRA training, and segmentation maps for each source mesh [here](https://drive.google.com/file/d/1d30j6ngOTjvUllBsTq-45FHFlwoQ2RaO/view?usp=sharing). These are the files we used to produce the paper's results.

Please download and unzip the data, then place the unzipped folder in the `data` directory.

While we provide the pre-trained LoRA weights and mesh part segmentation, you can also train the LoRA weights and segment the mesh parts yourself by following the steps below.

### LoRA Training & Inference

To [train LoRA weights using DreamBooth](https://huggingface.co/docs/peft/main/en/task_guides/dreambooth_lora), you can use the script provided in `configs/dreambooth_lora/train.sh`:

```bash
sh ./configs/dreambooth_lora/train.sh
```

To generate images using the pre-trained LoRA weights, use the script provided in `configs/dreambooth_lora/inference.sh`:

```bash
sh ./configs/dreambooth_lora/inference.sh
```

### Mesh Segmentation

We use [PartField](https://github.com/nv-tlabs/PartField) for mesh segmentation. 
Please follow the steps described in PartField's official repository to perform segmentation.

---

## 🪄 Mesh Deformation

You can experiment with various deformations by modifying the configuration files provided in the `configs` folder.

To deform meshes using the SDS loss with SDXL, we must first train an approximated VAE encoder. 
If you do not specify an `--approx_encoder_path` (*e.g.*, by setting `approx_encoder_path: null` in the config file), our code will automatically train the encoder matrix first and then resume the optimization process using the fitted encoder matrix. 

In short, simply run the following command:

```bash
python scripts/main.py --config ./configs/deformation/hand_maman.yml
```

If you already have the approximated VAE encoder, there is no need to fit the encoder matrix again. 
Instead, specify the path to the approximated VAE encoder (`img2latent_aug.pt`) directly in the configuration file.

*Note: Replace `--mesh` and `--lora_model` with your own paths to try different stylizations.*
