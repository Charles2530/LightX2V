# H3 progressive-resolution experiments

These opt-in configurations run the first two of four Turbo evaluations at a
smaller spatial latent grid, then finish at the requested canvas. They are
experimental; they do not establish quality equivalence or a speed guarantee.
Keep the matching Turbo/SLA LoRA, model, attention backend and quantization
checkpoint from the corresponding `distill/4steps` configuration. Replace the
placeholder checkpoint paths before use. The supplied examples use one GPU;
set `parallel.seq_p_size` for your available hardware.

The interface follows Wan's `changing_resolution`, `resolution_rate` and
`changing_resolution_steps`. A switch value of `2` means two completed
low-resolution evaluations. Only one transition is supported. Spatial sizes
round to the nearest patch multiple; a 768×1344 canvas with rate 0.75 uses a
36×64 latent grid, then returns to 48×84. Time and duration remain unchanged.

H3 predicts data-ward velocity: the transition estimates
`x0 = x_sigma + sigma * velocity` **before** the scheduler update. It resizes
that clean estimate independently per frame, then forms
`x_next = (1-sigma_next)*upsample(x0) + sigma_next*noise`. The noisy latent is
never directly upsampled. Video uses its original shifted sigma grid. Audio
receives the ordinary Euler update and retains its initial RNG stream and
full-canvas RoPE positions. Its predictions can still change because joint
attention sees different video tokens; this does not guarantee audio quality
or synchronization equivalence.

| Option | Default | Alternatives |
| --- | --- | --- |
| `resolution_rate` | `[0.75]` | One rate strictly between 0 and 1 |
| `changing_resolution_steps` | `[1]` | One completed-step count leaving a high-resolution step |
| `cr_upsample_mode` | `bicubic` | `bilinear`, `nearest` |
| `cr_lowres_noise` | `pooled` | `fresh` |
| `cr_renoise_noise` | `fresh` | `initial` |
| `cr_condition_downsample` | `area` | Re-encoding is not implemented |
| `cr_keep_audio_rope` | `true` | `false` for ablation |
| `cr_sigma_bump` | `0.0` | Nonnegative bump, requires `use_adaln_cache=false` |

`pooled` area-resizes the original full-resolution noise and normalizes its
empirical mean/standard deviation. It correlates low-resolution initialization
with the baseline. Nonintegral area reductions can correlate neighboring noise
samples; unit variance does not make them independent. Use `fresh` to measure
this approximation separately. Reusing `initial` noise at the lift also couples
noise to the generated clean estimate; it is an ablation, not a quality claim.
VAE keyframe condition rows are area-resized for the low stage and restored
exactly for the high stage. The Qwen conditioner prefix, including its image
features, stays unchanged; its rows must not all be counted as pure text. Spatial RoPE preserves the normalized full-canvas footprint, with
an aspect-ratio correction for patch rounding.

Supported: base-transformer T2AV/I2AV/L2AV/FL2AV, Euler, one spatial lift.
Rejected: Ref2AV, source regeneration, draft prefixes, LCM, Veda and feature
caching. Quantized projections and the native SP packer consume the current
grid on each call. Compatibility still requires a real run on the selected
backend/hardware. No cached hidden state may cross the transition.

With progressive resolution enabled, built-in warmup evaluates the full
trajectory so both grids execute. `warmup_shapes` overrides the existing
synthetic warmup canvases. Include the real production height, width and frame
count; different prompts can still produce different sequence lengths. Exclude
loading, compilation and warmup from reported request latency. CUDA Graph
capture is not added by this change.

Build AdaLN tables with the repository's existing
`tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py` using the final model,
LoRA and configuration. Schedule-only cache entries remain valid across spatial
grids; a changed sigma requires new entries or online modulation.
