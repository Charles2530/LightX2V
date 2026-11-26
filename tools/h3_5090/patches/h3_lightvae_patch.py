"""Opt-in official H3 LightVAE decoder in the existing tiled H3 runtime.

Original encoder, latent normalization, tiling and audio are retained.
Weights are loaded strictly; this does not quantize or train the decoder.
"""

import json
from pathlib import Path


def make_decoder(checkpoint, *, attn_type, use_compile, infer_dtype, sensitive_dtype):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file

    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoViTDecoder3d

    with safe_open(str(checkpoint), framework="pt", device="cpu") as f:
        architecture = json.loads(f.metadata()["architecture"])
    cfg = architecture["teacher_config"]
    kept = architecture["kept_layers"]
    assert kept == sorted(set(kept)) and 0 <= kept[0] < kept[-1] < cfg["decoder_num_layers"]
    with torch.device("meta"):
        decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=cfg["latent_channels"],
            out_channels=cfg["out_channels"],
            patch_size=16,
            patch_size_t=4,
            num_layers=len(kept),
            num_attention_heads=cfg["decoder_num_attention_heads"],
            attention_head_dim=cfg["decoder_attention_head_dim"],
            num_register_tokens=cfg["decoder_num_register_tokens"],
            ffn_mult=cfg["decoder_ffn_mult"],
            rope_theta=cfg["decoder_rope_theta"],
            rope_dim_ratio=cfg["decoder_rope_dim_ratio"],
            norm_eps=cfg["decoder_norm_eps"],
            infer_dtype=infer_dtype,
            sensitive_layer_dtype=sensitive_dtype,
            attn_type=attn_type,
            use_compile=use_compile,
        )
        post = torch.nn.Conv3d(cfg["latent_channels"], cfg["latent_channels"], 1)
    state = load_file(str(checkpoint), device="cpu")
    expected = {"decoder." + k for k in decoder.state_dict()} | {"post_quant_conv." + k for k in post.state_dict()} | {"latents_mean", "latents_std"}
    assert set(state) == expected, (set(state) - expected, expected - set(state))
    decoder.load_state_dict({k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}, strict=True, assign=True)
    post.load_state_dict({k[len("post_quant_conv.") :]: v for k, v in state.items() if k.startswith("post_quant_conv.")}, strict=True, assign=True)
    # This nonpersistent buffer is created on meta together with the module.
    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoRotaryPosEmbed

    decoder.rope = MiniMaxH3VideoRotaryPosEmbed(int(cfg["decoder_attention_head_dim"] * cfg["decoder_rope_dim_ratio"]), theta=cfg["decoder_rope_theta"])
    for module in decoder.modules():
        if isinstance(module, torch.nn.Linear):
            module.to(dtype=infer_dtype)
    post.to(dtype=infer_dtype)
    decoder.eval().requires_grad_(False)
    post.eval().requires_grad_(False)
    return decoder, post, architecture, {k: state[k] for k in ("latents_mean", "latents_std")}


def install(config, output):
    import torch

    from lightx2v.models.video_encoders.hf.minimax_h3.video_vae import MiniMaxH3VideoVAE

    original = MiniMaxH3VideoVAE.from_pretrained.__func__
    checkpoint = Path(config["research_lightvae_decoder"])
    output = Path(output)

    def patched(cls, *args, **kwargs):
        model = original(cls, *args, **kwargs)
        assert model._cpu_weight_sources is None, "This hook does not support shared/block VAE offload"
        decoder, post, architecture, stats = make_decoder(
            checkpoint, attn_type=kwargs.get("attn_type", "torch_sdpa"), use_compile=kwargs.get("use_compile", False), infer_dtype=model.infer_dtype, sensitive_dtype=model.sensitive_layer_dtype
        )
        for key, value in stats.items():
            torch.testing.assert_close(getattr(model, key).cpu().float(), value.float(), rtol=0, atol=0)
        model.decoder = decoder
        model.post_quant_conv = post
        if not model.cpu_offload:
            model.decoder.to(model.execution_device)
            model.post_quant_conv.to(model.execution_device)
        torch.cuda.empty_cache()
        details = {
            "checkpoint": str(checkpoint),
            "kept_layers": architecture["kept_layers"],
            "original_depth": architecture["teacher_config"]["decoder_num_layers"],
            "decoder_depth": len(architecture["kept_layers"]),
            "decoder_linear_dtype": str(model.infer_dtype),
            "sensitive_dtype": str(model.sensitive_layer_dtype),
            "attention": kwargs.get("attn_type"),
            "original_encoder_retained": True,
            "original_tiling_retained": True,
            "strict_state_dict": True,
            "note": "Decoder/post_quant_conv use official LightVAE weights. Encoder retains original FP8 checkpoint. No training.",
        }
        (output / "lightvae_integration.json").write_text(json.dumps(details, indent=2) + "\n")
        print("LIGHTVAE_LOADED " + json.dumps(details), flush=True)
        return model

    MiniMaxH3VideoVAE.from_pretrained = classmethod(patched)
