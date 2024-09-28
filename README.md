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
pip install --no-deps -e .
```

### Building Block Database

We provide preprocessed building block data. You can download it from [here](https://huggingface.co/whgao/synformer) and put it in the `data` directory.

However, the data is derived from Enamine's building block catalog, which are **available only upon request**.
Therefore, you should first request the data from Enamine [here](https://enamine.net/building-blocks/building-blocks-catalog) and download the <ins>US Stock</ins> catalog into the `data/building_blocks` directory.
Then run the following script which will check whether you have a copy of the Enamine's catalog and unarchive the preprocessed data for you:
```bash
python scripts/unarchive_wizard.py
```

### Trained Models

You can download the trained weights from [here](https://huggingface.co/whgao/synformer) and put them in the `data/trained_weights` directory.


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

If you find our code or model useful, we kindly ask that you consider citing our work in your papers:

```bibtex
@article{gao2024synformer,
  title={Generative artificial intelligence for navigating synthesizable chemical space},
  author={Wenhao Gao and Shitong Luo and Connor W. Coley},
  year={2024}
}

@inproceedings{luo2024chemprojector,
  title={Projecting Molecules into Synthesizable Chemical Spaces},
  author={Shitong Luo and Wenhao Gao and Zuofan Wu and Jian Peng and Connor W. Coley and Jianzhu Ma},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}

@inproceedings{gao2021amortized,
  title={Amortized Tree Generation for Bottom-up Synthesis Planning and Synthesizable Molecular Design},
  author={Gao, Wenhao and Mercado, Roc{\'\i}o and Coley, Connor W},
  booktitle={International Conference on Learning Representations},
  year={2021}
}

@article{gao2020synthesizability,
  title={The synthesizability of molecules proposed by generative models},
  author={Gao, Wenhao and Coley, Connor W},
  journal={Journal of chemical information and modeling},
  volume={60},
  number={12},
  pages={5714--5723},
  year={2020},
  publisher={ACS Publications}
}
```
