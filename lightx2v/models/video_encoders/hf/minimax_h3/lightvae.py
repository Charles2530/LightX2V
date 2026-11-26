"""LightVAE decoder replacement for the MiniMax-H3 video VAE.

A LightVAE checkpoint keeps a subset of the original ViT decoder layers. Its
safetensors metadata stores ``architecture = {"teacher_config", "kept_layers"}``.
Only ``decoder`` and ``post_quant_conv`` are replaced; the encoder, latent
normalization, tiling and parallel decode of the original VAE are kept.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from loguru import logger
from safetensors import safe_open
from safetensors.torch import load_file

_LATENT_STAT_KEYS = ("latents_mean", "latents_std")


def _read_architecture(checkpoint: Path) -> dict:
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    if "architecture" not in metadata:
        raise ValueError(f"LightVAE checkpoint {checkpoint} has no 'architecture' metadata")
    architecture = json.loads(metadata["architecture"])
    config = architecture["teacher_config"]
    kept = architecture["kept_layers"]
    if not kept or kept != sorted(set(kept)) or kept[0] < 0 or kept[-1] >= config["decoder_num_layers"]:
        raise ValueError(f"Invalid LightVAE kept_layers {kept} for a {config['decoder_num_layers']}-layer decoder")
    return architecture


def load_lightvae_decoder(
    checkpoint: str | Path,
    *,
    patch_size: int,
    patch_size_t: int,
    infer_dtype: torch.dtype,
    sensitive_layer_dtype: torch.dtype,
    attn_type: str,
    use_compile: bool,
):
    """Build and strictly load a LightVAE decoder on CPU.

    Returns ``(decoder, post_quant_conv, architecture, latent_stats)``.
    """
    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoRotaryPosEmbed, MiniMaxH3VideoViTDecoder3d

    checkpoint = Path(checkpoint)
    architecture = _read_architecture(checkpoint)
    config = architecture["teacher_config"]
    with torch.device("meta"):
        decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=config["latent_channels"],
            out_channels=config["out_channels"],
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            num_layers=len(architecture["kept_layers"]),
            num_attention_heads=config["decoder_num_attention_heads"],
            attention_head_dim=config["decoder_attention_head_dim"],
            num_register_tokens=config["decoder_num_register_tokens"],
            ffn_mult=config["decoder_ffn_mult"],
            rope_theta=config["decoder_rope_theta"],
            rope_dim_ratio=config["decoder_rope_dim_ratio"],
            norm_eps=config["decoder_norm_eps"],
            infer_dtype=infer_dtype,
            sensitive_layer_dtype=sensitive_layer_dtype,
            attn_type=attn_type,
            use_compile=use_compile,
        )
        post_quant_conv = nn.Conv3d(config["latent_channels"], config["latent_channels"], kernel_size=1)

    state = load_file(str(checkpoint), device="cpu")
    expected = {f"decoder.{k}" for k in decoder.state_dict()} | {f"post_quant_conv.{k}" for k in post_quant_conv.state_dict()} | set(_LATENT_STAT_KEYS)
    if set(state) != expected:
        raise ValueError(f"LightVAE checkpoint key mismatch: missing={sorted(expected - set(state))}, unexpected={sorted(set(state) - expected)}")
    decoder.load_state_dict({k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}, strict=True, assign=True)
    post_quant_conv.load_state_dict({k[len("post_quant_conv.") :]: v for k, v in state.items() if k.startswith("post_quant_conv.")}, strict=True, assign=True)
    # The rotary embedding holds non-persistent buffers created on meta.
    decoder.rope = MiniMaxH3VideoRotaryPosEmbed(int(config["decoder_attention_head_dim"] * config["decoder_rope_dim_ratio"]), theta=config["decoder_rope_theta"])

    # Same precision policy as the original decoder: FP16 Linear, everything
    # else (norms, residual path) in the sensitive dtype.
    for module in decoder.modules():
        if isinstance(module, nn.Linear):
            module.to(dtype=infer_dtype)
    post_quant_conv.to(dtype=infer_dtype)
    decoder.eval().requires_grad_(False)
    post_quant_conv.eval().requires_grad_(False)
    return decoder, post_quant_conv, architecture, {k: state[k] for k in _LATENT_STAT_KEYS}


def replace_with_lightvae_decoder(model, checkpoint: str | Path, *, attn_type: str, use_compile: bool) -> None:
    """Swap ``model.decoder``/``model.post_quant_conv`` for LightVAE weights in place."""
    decoder, post_quant_conv, architecture, stats = load_lightvae_decoder(
        checkpoint,
        patch_size=model.spatial_compression_ratio,
        patch_size_t=model.temporal_compression_ratio,
        infer_dtype=model.infer_dtype,
        sensitive_layer_dtype=model.sensitive_layer_dtype,
        attn_type=attn_type,
        use_compile=use_compile,
    )
    # The decoder was distilled against these latent statistics.
    for key, value in stats.items():
        if not torch.equal(getattr(model, key).detach().cpu().float(), value.float()):
            raise ValueError(f"LightVAE checkpoint {key} does not match the MiniMax-H3 VAE config")
    model.decoder = decoder
    model.post_quant_conv = post_quant_conv
    logger.info(
        "Loaded LightVAE decoder {}: {}/{} layers kept {}",
        checkpoint,
        len(architecture["kept_layers"]),
        architecture["teacher_config"]["decoder_num_layers"],
        architecture["kept_layers"],
    )


__all__ = ["load_lightvae_decoder", "replace_with_lightvae_decoder"]
