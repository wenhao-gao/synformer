import os
from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import overload

from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions

from .base import Drawable
from .mol import Molecule


class Template(Drawable):
    def __init__(self, smarts: str) -> None:
        super().__init__()
        self._smarts = smarts.strip()

    def __getstate__(self):
        return self._smarts

    def __setstate__(self, state):
        self._smarts = state

    @property
    def smarts(self) -> str:
        return self._smarts

    @cached_property
    def _rdmol(self):
        return AllChem.MolFromSmarts(self._smarts)

    def draw(self, size: int = 100, svg: bool = False):
        if svg:
            return Draw._moltoSVG(self._rdmol, sz=(size, size), highlights=[], legend=[], kekulize=True)
        else:
            return Draw.MolToImage(self._rdmol, size=(size, size), kekulize=True)

    def match(self, mol: Molecule) -> bool:
        return mol._rdmol.HasSubstructMatch(self._rdmol)

    def __hash__(self) -> int:
        return hash(self._smarts)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Reaction) and self.smarts == __value.smarts


class Reaction(Drawable):
    def __init__(self, smarts: str) -> None:
        super().__init__()
        self._smarts = smarts.strip()

    def __getstate__(self):
        return self._smarts

    def __setstate__(self, state):
        self._smarts = state

    @property
    def smarts(self) -> str:
        return self._smarts

    @cached_property
    def _reaction(self):
        r = AllChem.ReactionFromSmarts(self._smarts)
        rdChemReactions.ChemicalReaction.Initialize(r)
        return r

    def draw(self, size: int = 100, svg: bool = False):
        return Draw.ReactionToImage(self._reaction, subImgSize=(size, size), useSVG=svg)

    @cached_property
    def num_reactants(self) -> int:
        return self._reaction.GetNumReactantTemplates()

    @cached_property
    def num_agents(self) -> int:
        return self._reaction.GetNumAgentTemplates()

    @cached_property
    def num_products(self) -> int:
        return self._reaction.GetNumProductTemplates()

    @cached_property
    def reactant_templates(self) -> tuple[Template, ...]:
        # reactant_smarts = self.smarts.split(">")[0].split(".")
        reactant_smarts = [Chem.MolToSmarts(self._reaction.GetReactantTemplate(i)) for i in range(self.num_reactants)]
        return tuple(Template(s) for s in reactant_smarts)

    def match_reactant_templates(self, mol: Molecule) -> tuple[int, ...]:
        matched: list[int] = []
        for i, template in enumerate(self.reactant_templates):
            if template.match(mol):
                matched.append(i)
        return tuple(matched)

    @cached_property
    def product_templates(self) -> tuple[Template, ...]:
        product_smarts = self.smarts.split(">")[2].split(".")
        return tuple(Template(s) for s in product_smarts)

    def is_reactant(self, mol: Molecule) -> bool:
        return self._reaction.IsMoleculeReactant(mol._rdmol)

    def is_agent(self, mol: Molecule) -> bool:
        return self._reaction.IsMoleculeAgent(mol._rdmol)

    def is_product(self, mol: Molecule) -> bool:
        return self._reaction.IsMoleculeProduct(mol._rdmol)

    def __call__(self, reactants: Sequence[Molecule]) -> list[Molecule]:
        products = [Molecule.from_rdmol(p[0]) for p in self._reaction.RunReactants([m._rdmol for m in reactants])]
        products = [p for p in products if p.is_valid]
        return products

    def __hash__(self) -> int:
        return hash(self._smarts)

    def __eq__(self, __value: object) -> bool:
        return isinstance(__value, Reaction) and self.smarts == __value.smarts


class ReactionContainer(Sequence[Reaction]):
    def __init__(self, reactions: Iterable[Reaction]) -> None:
        super().__init__()
        self._reactions = tuple(reactions)

    @overload
    def __getitem__(self, index: int) -> Reaction:
        ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Reaction, ...]:
        ...

    def __getitem__(self, index: int | slice):
        return self._reactions[index]

    def __len__(self) -> int:
        return len(self._reactions)

    def match_reactions(self, mol: Molecule) -> dict[int, tuple[int, ...]]:
        matched: dict[int, tuple[int, ...]] = {}
        for i, rxn in enumerate(self._reactions):
            m = rxn.match_reactant_templates(mol)
            if len(m) > 0:
                matched[i] = m
        return matched


def read_reaction_file(path: os.PathLike) -> list[Reaction]:
    reactions: list[Reaction] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            reactions.append(Reaction(line))
    return reactions
