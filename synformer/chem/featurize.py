import re
from pathlib import Path

from rdkit import Chem


def atom_features_simple(atom: Chem.rdchem.Atom | None) -> int:
    if atom is None:
        return 0
    return min(atom.GetAtomicNum(), 100)


def bond_features_simple(bond: Chem.rdchem.Bond | None) -> int:
    if bond is None:
        return 0
    bt = bond.GetBondType()
    if bt == Chem.rdchem.BondType.SINGLE:
        return 1
    elif bt == Chem.rdchem.BondType.DOUBLE:
        return 2
    elif bt == Chem.rdchem.BondType.TRIPLE:
        return 3
    elif bt == Chem.rdchem.BondType.AROMATIC:
        return 4
    return 5


_smiles_vocab_path = Path(__file__).parent / "smiles_vocab.txt"
with open(_smiles_vocab_path) as f:
    _smiles_vocab = f.read().splitlines()
    _smiles_token_to_id = {token: i for i, token in enumerate(_smiles_vocab, start=1)}
    _smiles_token_max = max(_smiles_token_to_id.values())
    _smiles_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted(_smiles_vocab, reverse=True))) + ")")


def tokenize_smiles(s_in: str):
    tok: list[int] = []
    for token in _smiles_token_pattern.findall(s_in):
        tok.append(_smiles_token_to_id.get(token, _smiles_token_max + 1))
    return tok
