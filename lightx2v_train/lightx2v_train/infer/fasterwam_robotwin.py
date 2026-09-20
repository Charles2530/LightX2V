"""FasterWAM-only RoboTwin inference bridge (not the FastWAM first-frame path)."""
import logging


def create_fasterwam(*, compile_training_denoise=False, **model_kwargs):
    """Expose upstream one-pass future-cache inference through the policy API.

    The shared RoboTwin policy calls ``infer_action`` and inspects its signature
    to supply ``num_video_frames``. Bind the original bound method directly so
    that signature, video noise, timestep, masks and KV fusion remain upstream's.
    Merely removing the old FastWAM binding would select joint rollout instead.
    """
    if compile_training_denoise:
        raise ValueError("FasterWAM RoboTwin evaluation does not support compile_training_denoise.")

    from fasterwam.runtime import create_fasterwam as create_upstream

    model = create_upstream(**model_kwargs)
    model.infer_action = model.infer_action_one_pass_future_cache
    logging.getLogger(__name__).info(
        "FasterWAM RoboTwin action inference: one_pass_future_cache (upstream implementation)"
    )
    return model
