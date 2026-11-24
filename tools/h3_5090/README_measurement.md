# Opt-in H3 request observation

`request_measurement.py` wraps an already loaded H3 runner. It does not select a
new scheduler, modify model weights, or change the default HHHH path.

```python
from tools.h3_5090.request_measurement import H3RequestObservation

observer = H3RequestObservation(pipe.runner)
try:
    observer.reset(audio=True)
    pipe.generate(**request)
    steps = observer.export()
    final_video, final_audio = observer.finals
finally:
    observer.detach()
```

Each step reports latent grid size, Qwen prefix row counts by modality tag,
condition/target video rows, audio rows, shifted sigmas and CUDA event latency.
The event spans the production inference call and waits for its weight-load
and compute streams before recording completion. It includes host launch gaps
and stream waits; it is not a sum of kernel durations or a hardware occupancy
measurement. The existing runner's profiling synchronizations remain in place.

Final latent clones are made inside the request after the denoising segment,
so their cost belongs to the measured request. Copy or serialize them after
timing. `audio_trajectory` contains the pre-update audio state of each evaluated
step followed by the final state, when `reset(audio=True)` is used. Reading
metadata and recording events introduce overhead: verify same-seed final-latent
identity and report the observation configuration with timings.

`query_gpu_boundary(timeout=5)` performs one bounded `nvidia-smi` call. A timeout
returns `resource_query_timeout`; a command/parser failure returns
`resource_query_error`. Neither implies a foreign process. `boundary_verdict`
checks the observed processes against the supplied task PIDs and reports these
conditions separately. It never claims continuous GPU exclusivity based only
on two observations. For a multi-rank job include every rank PID. Do not launch
formal single-card measurements while another card is running an independent
block-offload workload.

These helpers observe Linux/NVIDIA execution. They do not start, stop, preempt,
retry, or recover jobs. Model loading, same-process warmup, deadlines, artifact
identity and metric interpretation remain the caller's responsibility.
