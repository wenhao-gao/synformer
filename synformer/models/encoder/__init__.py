from .base import BaseEncoder, NoEncoder
from .discrete import DiscreteEncoder
from .fixed import FixedSizeEncoder
from .graph import GraphEncoder
from .smiles import SMILESEncoder


def get_encoder(t: str, cfg) -> BaseEncoder:
    if t == "smiles":
        return SMILESEncoder(**cfg)
    elif t == "graph":
        return GraphEncoder(**cfg)
    elif t == "discrete":
        return DiscreteEncoder(**cfg)
    elif t == "fixed":
        return FixedSizeEncoder(**cfg)
    elif t == "none":
        return NoEncoder(**cfg)
    else:
        raise ValueError(f"Unknown encoder type: {t}")
