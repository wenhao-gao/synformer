# synformer_dev

This repo contaisn the main code for SynFormer model. **This repo is for development purposes [DO NOT PUBLIC THIS]**

# TO-DOs

- [ ] Clean the repo and prepare for the release
- [ ] Make a demo on Huggingface

# Instruction

0. Start (for development)

```bash
pip install -e .
```

1. Preprocess data

Download building block data to `data/building_blocks/` in sdf format .

```bash
python scripts/preprocess.py --model-config configs/dev_smiles_diffusion.yml
```
2. Train the model

```bash
python scripts/train.py configs/dev_smiles_diffusion.yml --debug --devices 1
# python scripts/train_ed.py configs/dev_ed.yml --debug --devices 1
```

3. Model inference

Work in progress

4. Use for molecular optimization

```bash
python scripts/molopt.py \
    --model <decoder-only-model-checkpoint> \
    --use-replay-buffer \
    --use-prior \
    --oracle <domain>:<task>  # Example: "tdc:osimertinib_mpo"
```
