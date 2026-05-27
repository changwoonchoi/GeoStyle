# Installation

## 🛠 CUDA Installation
This code was tested on Ubuntu 20.04 with CUDA 11.8.

If you don't have this version already, simply download the `.run` file at [this address](https://developer.nvidia.com/cuda-11-8-0-download-archive) and follow the procedure below to install it.

```bash
sudo ./[Your_INSTALL_BINARIES].run
```

Press `continue` and enter `accept`.
Then, untick everything except the **CUDA Toolkit 11.8**, and press `Install`.

## ⚙️ Setting Up Your Environment

Start by cloning this repository and its submodules:

```bash
git clone --recursive git@github.com:changwoonchoi/GeoStyle.git
cd GeoStyle

# This shouldn't be necessary with `--recursive`, but just in case you missed it:
git submodule update --init --recursive
```

Create a conda environment by running the following:

```bash
conda create -n geostyle_env -y python=3.9
```

Once you're done, modify your `activate.sh` shell script to automatically set up your conda environment, CUDA/build environment, and any other necessary environment variables. 

**Make sure to replace the CUDA path in the script with your own.**

Then activate your environment with:

```bash
source activate.sh
```

## 📦 Installing Dependencies

### PyTorch

Install PyTorch 2.3.0+cu118:

```bash
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

### IGL Library

Install [libigl](https://libigl.github.io/). 
Make sure that you install version **2.2.1**:

```bash
conda install -y -c conda-forge igl==2.2.1
```

### Other Dependencies

Install the rest of the dependencies:

```bash
pip install -r requirements.txt
```

### Our Library

Install our library with:

```bash
pip install -e .
```

---

## 🐛 Troubleshooting

If you encounter an error such as `TypeError: expected np.ndarray (got numpy.ndarray)`:

```bash
pip uninstall numpy
pip install "numpy < 1.26.4"
```
*Note: We recommend using `numpy==1.26.3`.*

If you encounter an error related to `xformers` with `torch==2.3.0`:

```bash
pip install xformers==0.0.22.post4 --index-url https://download.pytorch.org/whl/cu118
```
