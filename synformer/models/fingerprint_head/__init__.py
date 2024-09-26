from .ar import AutoRegressiveFingerprintHead
from .base import BaseFingerprintHead, ReactantRetrievalResult
from .diffusion import MlpDiffusionFingerprintHead, TransformerDiffusionFingerprintHead
from .multi import MultiFingerprintHead


def get_fingerprint_head(t: str, cfg):
    if t == "multi":
        return MultiFingerprintHead(**cfg)
    elif t == "diffusion":
        return MlpDiffusionFingerprintHead(**cfg)
    elif t == "diffusion_transformer":
        return TransformerDiffusionFingerprintHead(**cfg)
    elif t == "ar":
        return AutoRegressiveFingerprintHead(**cfg)
    else:
        raise ValueError(f"Invalid fingerprint head type: {t}")
