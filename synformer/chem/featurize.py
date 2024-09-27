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


_smiles_vocab = """H
He
Li
Be
B
C
N
O
F
Ne
Na
Mg
Al
Si
P
S
Cl
Ar
K
Ca
Sc
Ti
V
Cr
Mn
Fe
Co
Ni
Cu
Zn
Ga
Ge
As
Se
Br
Kr
Rb
Sr
Y
Zr
Nb
Mo
Tc
Ru
Rh
Pd
Ag
Cd
In
Sn
Sb
Te
I
Xe
Cs
Ba
La
Ce
Pr
Nd
Pm
Sm
Eu
Gd
Tb
Dy
Ho
Er
Tm
Yb
Lu
Hf
Ta
W
Re
Os
Ir
Pt
Au
Hg
Tl
Pb
Bi
Po
At
Rn
Fr
Ra
Ac
Th
Pa
U
Np
Pu
Am
Cm
Bk
Cf
Es
Fm
Md
No
Lr
Rf
Db
Sg
Bh
Hs
Mt
Ds
Rg
Cn
Nh
Fl
Mc
Lv
Ts
Og
b
c
n
o
s
p
0
1
2
3
4
5
6
7
8
9
[
]
(
)
.
=
#
-
+
\\
/
:
~
@
?
>
*
$
%""".splitlines()
_smiles_token_to_id = {token: i for i, token in enumerate(_smiles_vocab, start=1)}
_smiles_token_max = max(_smiles_token_to_id.values())
_smiles_token_pattern = re.compile("(" + "|".join(map(re.escape, sorted(_smiles_vocab, reverse=True))) + ")")


def tokenize_smiles(s_in: str):
    tok: list[int] = []
    for token in _smiles_token_pattern.findall(s_in):
        tok.append(_smiles_token_to_id.get(token, _smiles_token_max + 1))
    return tok
