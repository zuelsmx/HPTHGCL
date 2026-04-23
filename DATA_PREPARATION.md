# Data Preparation

The datasets are not included in this repository because the raw files and
processed graph artifacts can be large and are not suitable for a normal GitHub
repository.

Please prepare the datasets locally and place them under the `data/` directory
in the project root. The `data/` directory is intentionally ignored by Git, so
local datasets and generated PyTorch Geometric `processed/` files will not be
committed.

Expected layout:

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

For ACM, AMiner, and Freebase, download or prepare the corresponding raw files
from the original dataset sources and place them in the matching `raw/`
directory shown above.

After the raw files are placed correctly, run training from the project root.
For example:

```bash
python main.py --dataset acm --gpu 0
```

PyTorch Geometric will create `processed/` files under `data/<dataset>/` when
needed.
