# Skin Lesion Classification on ISIC 2019

*CM4126, Oleksandr Reshetnikov (2409418)*

All code (except for Streamlit app in `app.py`), results and discussions are in [`CM4126_coursework.ipynb`](CM4126_coursework.ipynb) (it can be opened right in a browser with all outputs without cloning this repository). Final model is ResNet50 pretrained on ImageNet. On official ISIC 2019 test set its balanced accuracy is 0.592 (Baseline CNN has 0.497).

## Setup

Everything was checked on Windows 11 with Python 3.12 and NVIDIA RTX 3080.

On Windows (cmd)

```shell
git clone https://github.com/batreller/CM4126-Coursework
cd CM4126-Coursework
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux and macOS

```shell
git clone https://github.com/batreller/CM4126-Coursework
cd CM4126-Coursework
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` installs PyTorch for CUDA 12.6 on Windows and Linux, it **can work without GPU (but much slower)**. CUDA is not available for macOS, so on macOS (only Apple Silicon) PyTorch without CUDA is installed and everything runs on CPU.

## Running the notebook

```shell
jupyter notebook CM4126_coursework.ipynb
```

All **outputs are already saved in the notebook**, but to reproduce them run all cells.

**At the first launch notebook will download ISIC 2019** from official ISIC archive into `data/raw/isic2019`. Zip archives are removed after extraction, together with resized images in `data/processed` **total size is around 14GBs**.

Splitting into `train` and `validation` is saved into `splits` so that all models used same images.

Trained models are already in `models` and notebook loads them and doesn't train them again. To train everything again `models` folder needs to be removed.

## Streamlit app

```shell
streamlit run app.py
```

App uses final model from `models` folder so app can be started immediately after repository is cloned without starting the notebook and you can load your dermatoscopic image right away. Button to load random image from test set will only appear after notebook has downloaded and prepared data into `data/processed/isic2019/test`.

## Reproducibility

Seed is fixed and deterministic algorithms are enabled in PyTorch. Notebook loads saved models so on my computer `Run All` gives completely identical result as in notebook. On another computer with saved models results should be almost the same because different GPUs and CPUs calculate a bit differently. If models are trained again on different hardware, final numbers can be a bit different because PyTorch does not guarantee same results on different hardware ([PyTorch reproducibility notes](https://pytorch.org/docs/stable/notes/randomness.html))
