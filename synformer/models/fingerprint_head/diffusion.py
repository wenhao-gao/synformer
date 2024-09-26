import abc

import torch
from torch import nn
from torch.nn import functional as F

from synformer.models.diffusion import BernoulliDiffusion, NoiseConditionedMLP
from synformer.models.transformer.gradient_checkpointing import (
    GradientCheckpointingTransformerDecoder,
)
from synformer.models.transformer.positional_encoding import PositionalEncoding

from .base import AuxDict, BaseFingerprintHead, LossDict


class BaseDiffusionFingerprintHead(BaseFingerprintHead, abc.ABC):
    def __init__(self, fingerprint_dim: int, diffusion_steps: int, diffusion_s: float):
        super().__init__(fingerprint_dim=fingerprint_dim)
        self.var_sched = BernoulliDiffusion(num_steps=diffusion_steps, s=diffusion_s)

    @abc.abstractmethod
    def predict_fingerprint_logit(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var_sched: BernoulliDiffusion | None = None,
    ) -> torch.Tensor:
        ...

    def predict(
        self,
        h: torch.Tensor,
        *,
        x_T: torch.Tensor | None = None,
        var_sched: BernoulliDiffusion | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        var_sched = var_sched or self.var_sched
        if num_samples is not None:
            h = h[..., None].repeat([1] * h.dim() + [num_samples])
        fp_shape = [*h.shape[:-1], self.fingerprint_dim]
        if x_T is None:
            x_T = torch.randint(
                low=0,
                high=2,
                size=fp_shape,
                dtype=h.dtype,
                device=h.device,
            )

        x_t = x_T
        for t_index in range(var_sched.num_steps, 0, -1):
            t_bitwise = torch.full(fp_shape, fill_value=t_index, dtype=torch.long, device=h.device)
            t_fp = t_bitwise[..., 0]
            p_0_pred = self.predict_fingerprint_logit(h=h, x_t=x_t, t=t_fp, var_sched=var_sched).sigmoid()
            x_t = var_sched.denoise(x_t=x_t, p_0_pred=p_0_pred, t=t_bitwise).value
        return x_t

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        num_timestep_samples: int = 32,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        fp_gt = fp_target[None].repeat(num_timestep_samples, 1, 1, 1)  # (n_smp, bsz, len, fp_dim)
        timestep = torch.randint(
            low=1,
            high=self.var_sched.num_steps + 1,
            size=fp_gt.shape[:-1],
            device=fp_gt.device,
        )  # (n_smp, bsz, len)
        timestamp_bitwise = timestep[..., None].expand_as(fp_gt)

        noisy_fp = self.var_sched.add_noise(x_0=fp_gt, t=timestamp_bitwise).value
        p_denoised = self.predict_fingerprint_logit(
            h=h[None].repeat(num_timestep_samples, 1, 1, 1),
            x_t=noisy_fp,
            t=timestep,
        ).sigmoid()
        posterior_fp_gt = self.var_sched.posterior(p_t=noisy_fp, p_0=fp_gt, t=timestamp_bitwise)
        posterior_fp_pred = self.var_sched.posterior(p_t=noisy_fp, p_0=p_denoised, t=timestamp_bitwise)
        bce_diffusion = F.binary_cross_entropy(input=posterior_fp_pred, target=posterior_fp_gt, reduction="none")
        loss_diffusion = (bce_diffusion.mean(0).sum(-1) * fp_mask).sum() / (fp_mask.sum() + 1e-6)

        bce_direct = F.binary_cross_entropy(input=p_denoised, target=fp_gt, reduction="none")
        loss_bce = (bce_direct.mean(0).sum(-1) * fp_mask).sum() / (fp_mask.sum() + 1e-6)

        return {"fingerprint_diffusion": loss_diffusion, "fingerprint_bce": loss_bce}, {}

    def get_log_likelihood(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        num_timestep_samples: int = 100,
        **kwargs,
    ):
        fp_gt = fp_target[None].repeat(num_timestep_samples, 1, 1, 1)  # (n_smp, bsz, len, fp_dim)
        timestep = torch.randint(
            low=1,
            high=self.var_sched.num_steps + 1,
            size=fp_gt.shape[:-1],
            device=fp_gt.device,
        )  # (n_smp, bsz, len)
        timestamp_bitwise = timestep[..., None].expand_as(fp_gt)

        noisy_fp = self.var_sched.add_noise(x_0=fp_gt, t=timestamp_bitwise).value
        p_denoised = self.predict_fingerprint_logit(
            h=h[None].repeat(num_timestep_samples, 1, 1, 1),
            x_t=noisy_fp,
            t=timestep,
        ).sigmoid()

        # It is an empirical likelihood value
        likelihood = p_denoised * fp_gt + (1 - p_denoised) * (1 - fp_gt)
        ll = torch.log(likelihood + 1e-8).sum(-1).mean(0) * fp_mask
        return ll


class MlpDiffusionFingerprintHead(BaseDiffusionFingerprintHead):
    def __init__(self, d_model: int, fingerprint_dim: int, hidden_dim: int, diffusion_steps: int, diffusion_s: float):
        super().__init__(
            fingerprint_dim=fingerprint_dim,
            diffusion_steps=diffusion_steps,
            diffusion_s=diffusion_s,
        )
        self.denoiser = NoiseConditionedMLP(
            dim_in=d_model + fingerprint_dim,
            dim_out=fingerprint_dim,
            dim_hidden=hidden_dim,
        )

    def predict_fingerprint_logit(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var_sched: BernoulliDiffusion | None = None,
    ) -> torch.Tensor:
        return self.denoiser(
            torch.cat([h, x_t], dim=-1),
            t=t,
            var_sched=var_sched or self.var_sched,
        )


class TransformerDiffusionFingerprintHead(BaseDiffusionFingerprintHead):
    def __init__(
        self,
        d_model: int,
        fingerprint_dim: int,
        diffusion_steps: int,
        diffusion_s: float,
        n_bits_per_word: int,
        nhead: int,
        dim_feedforward: int,
        num_layers: int,
        norm: int,
        num_training_fingerprints_per_sample: int | None = 1,
        enable_gradient_checkpointing: bool = False,
    ):
        super().__init__(
            fingerprint_dim=fingerprint_dim,
            diffusion_steps=diffusion_steps,
            diffusion_s=diffusion_s,
        )
        self.num_training_fingerprints_per_sample = num_training_fingerprints_per_sample

        self.t_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.n_bits_per_word = n_bits_per_word
        self.num_words = fingerprint_dim // n_bits_per_word
        if fingerprint_dim % n_bits_per_word != 0:
            raise ValueError("fingerprint_dim must be divisible by n_bits_per_word")

        self.word_embed = nn.Sequential(
            nn.Linear(n_bits_per_word, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.pe = PositionalEncoding(
            d_model=d_model,
            max_len=self.num_words + 1,
        )

        self.attn = (
            GradientCheckpointingTransformerDecoder if enable_gradient_checkpointing else nn.TransformerDecoder
        )(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
                norm_first=True,
            ),
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model) if norm else None,
        )
        self.out = nn.Linear(d_model, self.n_bits_per_word)

    def predict(
        self,
        h: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        x_T: torch.Tensor | None = None,
        var_sched: BernoulliDiffusion | None = None,
        num_samples: int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if mask is not None:
            idx = mask.flatten().nonzero(as_tuple=True)[0]
            h = h.view(-1, 1, h.size(-1))[idx]
            flat_shape = [mask.numel(), self.fingerprint_dim]
            out_shape = mask.shape + (self.fingerprint_dim,)
            if idx.numel() == 0:
                return torch.zeros(out_shape, dtype=h.dtype, device=h.device)

        out = super().predict(h=h, x_T=x_T, var_sched=var_sched, num_samples=num_samples, **kwargs)

        if mask is not None:
            out_flat = torch.zeros(flat_shape, dtype=out.dtype, device=out.device)
            out_flat = torch.scatter(
                input=out_flat,
                dim=0,
                index=idx[:, None].expand(idx.size(0), self.fingerprint_dim),
                src=out[:, 0],
            )
            out = out_flat.reshape(out_shape)
        return out

    def predict_fingerprint_logit(
        self,
        h: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        var_sched: BernoulliDiffusion | None = None,
    ) -> torch.Tensor:
        """
        Args:
            h:  (*batch_shape, seqlen, d_model)
            x_t:  (*batch_shape, seqlen, fp_dim)
            t:  (*batch_shape, seqlen)
        """
        var_sched = var_sched or self.var_sched
        fp_shape = x_t.shape

        h = h.reshape(-1, h.size(-1))  # (bsz*seqlen, d_model)
        t = t.reshape(-1, 1).float() / self.var_sched.num_steps  # (bsz*seqlen, 1)
        t_feat = self.t_embed(t)  # (bsz*seqlen, d_model)

        mem = torch.stack([h, t_feat], dim=-2)  # (bsz*seqlen, 2, d_model)

        x_t = x_t.reshape(-1, self.num_words, self.n_bits_per_word)  # (bsz*seqlen, n_words, nbpw)
        x_feat = self.pe(self.word_embed(x_t))  # (bsz*seqlen, n_words, d_model)

        y = self.attn(tgt=x_feat, memory=mem)  # (bsz*seqlen, n_words, d_model)
        out: torch.Tensor = self.out(y)  # (bsz*seqlen, n_words, nbpw)
        return out.reshape(fp_shape)

    def get_loss(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[LossDict, AuxDict]:
        """
        Args:
            h:  (bsz, seqlen, d_model)
            fp_target:  (bsz, seqlen, fp_dim)
            fp_mask:  (bsz, seqlen)
        """
        # To reduce memory usage, we only use 1 sample for training
        if self.num_training_fingerprints_per_sample is not None:
            bsz = h.size(0)
            h = h.reshape(-1, h.size(-1))
            fp_target = fp_target.reshape(-1, fp_target.size(-1))
            fp_mask = fp_mask.reshape(-1)
            order_number = torch.rand_like(fp_mask, dtype=torch.float) * fp_mask
            n_fps = bsz * self.num_training_fingerprints_per_sample
            selected_idx = torch.argsort(order_number, descending=True)[:n_fps]

            h = h[selected_idx][:, None]
            fp_target = fp_target[selected_idx][:, None]
            fp_mask = fp_mask[selected_idx][:, None]

        return super().get_loss(h=h, fp_target=fp_target, fp_mask=fp_mask, **kwargs)

    def get_log_likelihood(
        self,
        h: torch.Tensor,
        fp_target: torch.Tensor,
        fp_mask: torch.Tensor,
        *,
        num_timestep_samples: int = 1,
        **kwargs,
    ):
        numel = fp_mask.numel()
        out_shape = fp_mask.size()
        fp_idx = fp_mask.flatten().nonzero(as_tuple=True)[0]
        h = h.reshape(-1, 1, h.size(-1))[fp_idx]
        fp_target = fp_target.reshape(-1, 1, fp_target.size(-1))[fp_idx]
        fp_mask = fp_mask.reshape(-1, 1)[fp_idx]
        ll = super().get_log_likelihood(
            h=h,
            fp_target=fp_target,
            fp_mask=fp_mask,
            num_timestep_samples=num_timestep_samples,
            **kwargs,
        )  # (n_fps, 1)
        ll_out = torch.scatter(
            input=torch.zeros([numel], dtype=ll.dtype, device=fp_mask.device),
            dim=0,
            index=fp_idx,
            src=ll[:, 0],
        ).reshape(out_shape)
        return ll_out
