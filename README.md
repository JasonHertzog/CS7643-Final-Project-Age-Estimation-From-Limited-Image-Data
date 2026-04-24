# CS7643 Final Project

Due May 4th.

## Environment Setup

To setup an appropriate Python environment, use the following commands. 
```
conda env create -f environment.yaml

conda activate cs7643-project
```

To deactivate environment, use `conda deactivate`

## Download Data

Run the downloader from the repository root:

```bash
python scripts/download_data.py
```

This downloads the official UTKFace in-the-wild archives into `data/raw/utkface/`.

For a smaller test run, you can download a single archive:

```bash
python scripts/download_data.py --files part3.tar.gz
```

After downloading, extract the archives into `data/raw/utkface/`:

```bash
tar -xzf data/raw/utkface/part1.tar.gz -C data/raw/utkface
tar -xzf data/raw/utkface/part2.tar.gz -C data/raw/utkface
tar -xzf data/raw/utkface/part3.tar.gz -C data/raw/utkface
```

## Preprocess Data

The preprocessing script was implemented in `scripts/preprocess_data.py`.

Before running it, update `RAW_DATA_DIR` near the top of that file so it points to the extracted UTKFace dataset directory.

If you extracted the dataset inside this repo, that path should be:

```python
RAW_DATA_DIR = Path("data/raw/utkface")
```

Run preprocessing from the repository root inside the conda environment as a module:

```bash
python -m scripts.preprocess_data
```

This writes the reusable split files for reproducibility to `data/processed/` as `train.csv`, `val.csv`, and `test.csv`.

Note: Please make sure that the images are NOT uploaded to GitHub directly.


## Training loop

A full training loop can be run by issuing the command 

python -m scripts.train --config configs/baseline.yaml

## Evaluation Step

Evaluation of a model can be triggered by issuing the command

python -m scripts.evaluate --config configs/baseline.yaml



