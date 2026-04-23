# Data Preparation

Datasets are not included in this repository. Some raw and processed graph
files are large, and generated PyTorch Geometric artifacts should not be
committed to Git.

Create a `data/` directory in the project root and place datasets under the
following layout:

```text
data/
|-- acm/
|   |-- raw/
|   |   `-- ACM.mat
|-- aminer/
|   |-- raw/
|   |   |-- labels.npy
|   |   |-- pr.txt
|   |   |-- pa.txt
|   |   |-- features_0.npy
|   |   |-- features_1.npy
|   |   `-- features_2.npy
|-- freebase/
|   |-- raw/
|   |   |-- labels.npy
|   |   |-- ma.txt
|   |   |-- md.txt
|   |   |-- mw.txt
|   |   |-- features_0.npy
|   |   |-- features_1.npy
|   |   |-- features_2.npy
|   |   `-- features_3.npy
```

The DBLP and IMDB datasets are loaded through PyTorch Geometric:

```python
from torch_geometric.datasets import DBLP, IMDB
```

After the raw files are placed correctly, run training from the project root:

```bash
python main.py --dataset acm --gpu 0
```

PyTorch Geometric will create `processed/` files under `data/<dataset>/` when
needed.
