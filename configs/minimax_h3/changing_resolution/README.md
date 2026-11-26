# H3 progressive-resolution experiments

These opt-in configurations run the first two of four Turbo evaluations at a
smaller spatial latent grid, then finish at the requested canvas. They are
experimental; they do not establish quality equivalence or a speed guarantee.
Keep the matching Turbo/SLA LoRA, model, attention backend and quantization
checkpoint from the corresponding `distill/4steps` configuration. Replace the
placeholder checkpoint paths before use. The supplied examples use one GPU;
set `parallel.seq_p_size` for your available hardware.

The single-GPU baseline
`configs/minimax_h3/distill/4steps/minimax_h3_fp8_4step_5090_sla90_lightvae.json`
carries the same keys with `changing_resolution=false`; set it to `true` (or run
`scripts/minimax_h3/run_minimax_h3_4step_sla90_lightvae.sh` with
`CHANGING_RESOLUTION=1`) to stack progressive resolution on that baseline.

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
trajectory so both grids execute. Exclude loading, compilation and warmup
from reported request latency. CUDA Graph capture is not added by this change.

Build AdaLN tables with the repository's existing
`tools/cache_minimax_h3_adaln/cache_minimax_h3_adaln.py` using the final model,
LoRA and configuration. Schedule-only cache entries remain valid across spatial
grids; a changed sigma requires new entries or online modulation.

## Stage-three quality ablations

The following options select an experimental scheduler subclass when
`changing_resolution=true`; all default to disabled. Ordinary HHHH still uses
`MiniMaxH3Scheduler`, and the existing single-lift configuration keeps its
original implementation.

- `cr_intermediate_rate`: add a second low grid after step 0, followed by the
  final lift after step 1. For example rate 0.375 and intermediate 0.5 evaluates
  `[0.375, 0.5, 1, 1]`. Both lifts predict the clean endpoint before updating
  and add fresh noise at the next original video sigma. Only this two-low-step
  arrangement is supported for the intermediate grid.
- `cr_keep_full_condition=true`: retain full-canvas VAE keyframe condition rows
  while target-video rows use the low grid. Their original positions and
  timesteps remain separate from the target grid; Qwen reference features
  remain unchanged, as before. This increases token count and latency.
- `cr_upsample_blur=true`: use bilinear clean-latent upsampling followed by a
  per-frame 3×3 separable `[1,2,1]/4` Gaussian filter. It does not blur noise.
- `cr_sigma_bump`: the staged path rescales the entire remaining video sigma
  suffix by the lift-sigma ratio. Off-grid timesteps require exact newly built
  AdaLN entries (`cr_extended_adaln_cache=true`) or online modulation. Merely
  setting the flag does not build or validate cache contents; the inference
  cache lookup must still find every exact timestep.
- `cr_audio_mode=renoise`: estimate clean audio at the video lift, then apply
  independent fresh audio noise at the unchanged next audio sigma.
- `cr_audio_mode=freeze`: leave initial audio noise unchanged throughout low
  video steps, then apply a separately shifted audio schedule over remaining
  steps. This changes the trained joint time pairs and requires new exact
  AdaLN entries or online modulation.
- `cr_audio_mode=replay`: diagnostic only. The caller must supply
  `scheduler.cr_audio_replay`, the HHHH audio states including initial and final
  states, before every request. Each low-step update is replaced with the
  matching teacher state. It requires a teacher rollout and is not a deployable
  acceleration method or a quality guarantee.

These options are for controlled comparisons. They do not establish quality
parity, audiovisual synchronization, or cross-prompt robustness. Do not combine
unmeasured changes and attribute their cumulative result to one option.
