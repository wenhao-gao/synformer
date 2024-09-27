# SynFormer

SynFormer is a generative modeling framework designed to efficiently explore and navigate synthesizable chemical space.

[Paper]

## Install

### Environment

```bash
# Install conda environment
conda env create -f env.yml -n synformer
conda activate synformer

# Install SynFormer package
pip install -e .
```

### Building Block Database

We provide preprocessed building block data. You can download it from [here](#link) and put it in the `data` directory.

However, the data is derived from Enamine's building block catalog, which are **available only upon request**.
Therefore, you should first request the data from Enamine [here](https://enamine.net/building-blocks/building-blocks-catalog) and download the <ins>US Stock</ins> catalog into the `data` directory.
Then run the following script which will check whether you have a copy of the Enamine's catalog and unarchive the preprocessed data for you:
```bash
python scripts/unarchive_wizard.py
```

### Trained Models

You can download the trained weights from [here](#link) and put them in the `data/trained_weights` directory.


## Usage

### Bottom-Up Synthesis Planning

You can create a list of SMILES strings in CSV format (example: `data/example.csv`) and run the following command to project them into the synthesizable chemical space.
```bash
python sample.py \
    --model-path data/trained_weights/original_default.ckpt \
    --input data/example.csv \
    --output results/example.csv
```


## Reference


