# CS7643 Final Project

Due May 4th.

## Step 0 - Environment Setup

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

Note: Please make sure that the images are NOT uploaded to GitHub directly.
