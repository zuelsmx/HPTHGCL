# HPT-HGCL

Training code for heterogeneous graph contrastive learning.

## 1. Project Structure

```text
HPTHGCL-main/
|-- main.py
|-- model.py
|-- datasets.py
|-- GTLayer.py
|-- hyperbolic_utils.py
|-- mf.py
|-- utils.py
|-- requirements.txt
|-- DATA_PREPARATION.md
```

## 2. Environment

- Python 3.9+
- PyTorch
- PyTorch Geometric
- numpy
- scipy
- scikit-learn
- pandas

Example (adjust versions/CUDA to your machine):

```bash
pip install torch torchvision torchaudio
pip install torch-geometric
pip install -r requirements.txt
```

## 3. Quick Start

Run from the project root:

```bash
python main.py --dataset acm --gpu 0
```

Supported datasets:

- `acm`
- `aminer`
- `freebase`
- `dblp`
- `imdb`

## 4. Data Preparation

Datasets are not committed to this repository. Please see
[DATA_PREPARATION.md](./DATA_PREPARATION.md).
