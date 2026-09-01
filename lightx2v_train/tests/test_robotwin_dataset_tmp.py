import torch

from lightx2v_train.data.robotwin_dataset import (
    RobotWinFastWAMProcessor,
    compose_robotwin_video,
)


def test_robotwin_camera_layout_matches_released_three_camera_checkpoint():
    cameras = torch.stack(
        [
            torch.ones(1, 3, 480, 640),
            torch.full((1, 3, 480, 640), 2.0),
            torch.full((1, 3, 480, 640), 3.0),
        ]
    )

    video = compose_robotwin_video(cameras)

    assert tuple(video.shape) == (1, 3, 384, 320)
    assert torch.allclose(video[:, :, :256], torch.ones_like(video[:, :, :256]))
    assert torch.allclose(video[:, :, 256:, :160], torch.full_like(video[:, :, 256:, :160], 2.0))
    assert torch.allclose(video[:, :, 256:, 160:], torch.full_like(video[:, :, 256:, 160:], 3.0))


def test_robotwin_processor_supports_fourteen_dimensional_zscore_actions():
    shape_meta = {
        "images": [
            {"key": "cam_high", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_left_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
            {"key": "cam_right_wrist", "raw_shape": [3, 480, 640], "shape": [3, 240, 320]},
        ],
        "action": [{"key": "default", "raw_shape": 14, "shape": 14}],
        "state": [{"key": "default", "raw_shape": 14, "shape": 14}],
    }
    processor = RobotWinFastWAMProcessor(shape_meta, num_obs_steps=1, image_size=(240, 320))
    stats = {
        "action": {"default": {"global_mean": torch.ones(14), "global_std": torch.full((14,), 2.0)}},
        "state": {"default": {"global_mean": torch.ones(14), "global_std": torch.full((14,), 2.0)}},
    }
    processor.set_normalizer_from_stats(stats)
    sample = {
        "idx": 0,
        "task": "pick up the block",
        "images": {
            "cam_high": torch.zeros(1, 3, 480, 640),
            "cam_left_wrist": torch.zeros(1, 3, 480, 640),
            "cam_right_wrist": torch.zeros(1, 3, 480, 640),
        },
        "action": {"default": torch.full((32, 14), 3.0)},
        "state": {"default": torch.full((33, 14), 3.0)},
        "action_is_pad": torch.zeros(32, dtype=torch.bool),
        "state_is_pad": torch.zeros(33, dtype=torch.bool),
        "image_is_pad": torch.zeros(1, dtype=torch.bool),
    }

    processed = processor.preprocess(sample)

    assert tuple(processed["pixel_values"].shape) == (3, 1, 3, 240, 320)
    assert tuple(processed["action"].shape) == (32, 14)
    assert torch.allclose(processed["action"], torch.ones(32, 14))
    assert torch.allclose(processed["proprio"], torch.ones(33, 14))
