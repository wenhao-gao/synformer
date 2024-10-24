# SynFormer

SynFormer is a generative modeling framework designed to efficiently explore and navigate synthesizable chemical space.

Please see the model detail in our [paper](https://arxiv.org/abs/2410.03494).

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

We provide preprocessed building block data (fpindex.pkl and matrix.pkl). You can download it from [here](https://huggingface.co/whgao/synformer) and put it in the `data` directory. Note that the location of the files should match what is described in the config file. 

However, the data is derived from Enamine's building block catalog, which are **available only upon request**.
Therefore, you should first request the data from Enamine [here](https://enamine.net/building-blocks/building-blocks-catalog) and download the <ins>US Stock</ins> catalog into the `data/building_blocks` directory. The preprocessed data is for research purposes only, and any commercial use requires appropriate permissions. We do not take responsibility for any consequences arising from the use of this data.

### Trained Models

You can download the trained weights from [here](https://huggingface.co/whgao/synformer) and put them in the `data/trained_weights` directory.

### Train with Your Set of Templates and Building Blocks

If you have a set of reaction templates and a list of building block you want to use, first change the file path in 6th and 7th line of the config files, and then run
```bash
python scripts/preprocess.py --model-config your_config_file.yml
```
to obtain the processed data file (fpindex.pkl and matrix.pkl). Then one can train the model with:
```bash
python scripts/train.py configs/dev_smiles_diffusion.yml
```
Please adjust the batch size and number of trainig epoches according to your computational resources.

## Usage

### Inference

You can create a list of SMILES strings in CSV format (example: `data/example.csv`) and run the following command to project them into the synthesizable chemical space.
```bash
python sample.py \
    --model-path data/trained_weights/original_default.ckpt \
    --input data/example.csv \
    --output results/example.csv
```
Note that one can run our model in either CPU or GPU, but due to the wall time limit we set, using a CPU not only results in slower performance but may also affect the model's outcomes. The results reported in the paper were obtained using an NVIDIA RTX 4090. Based on our experience, running a single molecule on a CPU takes approximately 30 seconds to 2 minutes, whereas on an NVIDIA RTX 4090, it takes a few seconds to 30 seconds. Additionally, if the process freezes without any error messages, it could be due to a memory issue causing the subprocess to hang. In this case, you may want to try increasing the available RAM.

### GraphGA-SF

You can run the GraphGA-SF model with following command:
```bash
python experiments/graphga_sf_opt.py --oracle QED --name QED
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
