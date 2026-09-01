"""RobotWin-specific FastWAM data adapters."""

import torch
import torchvision.transforms.functional as vision
from loguru import logger
from torch.utils.data import DataLoader, DistributedSampler

from lightx2v_train.data.libero.dataset import (
    DatasetSliceRepeat,
    _dataset_roots,
    _path,
    _resolve_shape_meta,
)
from lightx2v_train.data.libero.normalizer import LinearNormalizer
from lightx2v_train.data.libero.processor import FastWAMProcessor
from lightx2v_train.data.libero.robot_video_dataset import PROMPT_TEMPLATE, RobotVideoDataset
from lightx2v_train.runtime.distributed import get_data_parallel_rank, get_data_parallel_world_size
from lightx2v_train.utils.registry import DATA_REGISTER


ROBOTWIN_CAMERA_COUNT = 3
ROBOTWIN_ACTION_DIM = 14
ROBOTWIN_CAMERA_IMAGE_SIZE = (240, 320)
ROBOTWIN_VIDEO_SIZE = (384, 320)
ROBOTWIN_DELTA_ACTION_DIM_MASK = (
    True,
    True,
    True,
    True,
    True,
    True,
    False,
    True,
    True,
    True,
    True,
    True,
    True,
    False,
)


def compose_robotwin_video(cameras, video_size=ROBOTWIN_VIDEO_SIZE):
    """Tile three camera streams into the layout used by the RobotWin model."""
    if not isinstance(cameras, torch.Tensor) or cameras.ndim != 5:
        raise ValueError(f"RobotWin cameras must be a tensor [3,T,C,H,W], got {type(cameras).__name__} {getattr(cameras, 'shape', None)}")
    if cameras.shape[0] != ROBOTWIN_CAMERA_COUNT:
        raise ValueError(f"RobotWin composite requires exactly 3 cameras, got {cameras.shape[0]}")
    if cameras.shape[1] < 1:
        raise ValueError("RobotWin camera streams must contain at least one frame")
    if not isinstance(video_size, (tuple, list)) or len(video_size) != 2:
        raise ValueError(f"RobotWin video_size must be [height, width], got {video_size}")
    output_height, output_width = (int(video_size[0]), int(video_size[1]))
    if (output_height, output_width) != ROBOTWIN_VIDEO_SIZE:
        raise ValueError(
            "The released RobotWin FastWAM checkpoint requires video_size "
            f"{list(ROBOTWIN_VIDEO_SIZE)}, got {[output_height, output_width]}"
        )

    cam_top = vision.resize(
        cameras[0],
        size=[256, 320],
        interpolation=vision.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_left = vision.resize(
        cameras[1],
        size=[128, 160],
        interpolation=vision.InterpolationMode.BILINEAR,
        antialias=True,
    )
    cam_right = vision.resize(
        cameras[2],
        size=[128, 160],
        interpolation=vision.InterpolationMode.BILINEAR,
        antialias=True,
    )
    bottom = torch.cat((cam_left, cam_right), dim=-1)
    return torch.cat((cam_top, bottom), dim=-2).contiguous()


class RobotWinFastWAMProcessor(FastWAMProcessor):
    """FastWAM processor matching the released RobotWin recipe."""

    def __init__(
        self,
        shape_meta,
        num_obs_steps,
        image_size=ROBOTWIN_CAMERA_IMAGE_SIZE,
        delta_action_dim_mask=ROBOTWIN_DELTA_ACTION_DIM_MASK,
    ):
        if len(shape_meta.get("images", [])) != ROBOTWIN_CAMERA_COUNT:
            raise ValueError(
                "RobotWin FastWAM expects exactly three cameras, got "
                f"{len(shape_meta.get('images', []))}"
            )
        for group in ("action", "state"):
            if len(shape_meta.get(group, [])) != 1:
                raise ValueError(f"RobotWin FastWAM expects one {group} field")
            width = int(shape_meta[group][0]["raw_shape"])
            if width != ROBOTWIN_ACTION_DIM:
                raise ValueError(
                    f"RobotWin FastWAM expects 14-dimensional {group}, got {width}"
                )
        super().__init__(
            shape_meta,
            num_obs_steps,
            image_size=image_size,
            delta_action_dim_mask=delta_action_dim_mask,
        )
        if tuple(self.image_size) != ROBOTWIN_CAMERA_IMAGE_SIZE:
            raise ValueError(
                "RobotWin camera preprocessing must produce "
                f"{list(ROBOTWIN_CAMERA_IMAGE_SIZE)}, got {list(self.image_size)}"
            )
        if self.delta_action_dim_mask.numel() != ROBOTWIN_ACTION_DIM:
            raise ValueError(
                f"RobotWin delta_action_dim_mask must have 14 entries, got {self.delta_action_dim_mask.numel()}"
            )

    def set_normalizer_from_stats(self, stats):
        self._normalizer = LinearNormalizer(self.shape_meta, stats, mode="z-score")


class RobotWinVideoDataset(RobotVideoDataset):
    """Three-camera RobotWin dataset with released-checkpoint tiling."""

    def __init__(self, *args, video_size=ROBOTWIN_VIDEO_SIZE, **kwargs):
        if tuple(video_size) != ROBOTWIN_VIDEO_SIZE:
            raise ValueError(
                "RobotWin FastWAM currently supports only video_size "
                f"{list(ROBOTWIN_VIDEO_SIZE)}, got {list(video_size)}"
            )
        self.video_size = tuple(video_size)
        super().__init__(*args, **kwargs)

    def _get(self, index):
        sample = self._sample_without_padding(index)
        video = sample["pixel_values"][:, self.video_sample_indices]
        video = compose_robotwin_video(video, self.video_size)
        video = video.mul(2.0).sub(1.0).permute(1, 0, 2, 3).contiguous()

        action = sample["action"]
        proprio = sample["proprio"][:-1]
        prompt = PROMPT_TEMPLATE.format(task=sample["instruction"])
        context, context_mask = self._cached_context(prompt)
        return {
            "video": video,
            "action": action,
            "proprio": proprio,
            "prompt": prompt,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": sample["image_is_pad"][self.video_sample_indices],
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }


def _build_robotwin_dataset(config, split):
    shape_meta = _resolve_shape_meta(config)
    num_frames = int(config.get("num_frames", 33))
    observation_only_video = bool(config.get("observation_only_video", False))
    image_size = tuple(int(value) for value in config.get("image_size", (240, 320)))
    action_dim = int(shape_meta["action"][0]["shape"])
    configured_mask = config.get("delta_action_dim_mask")
    if configured_mask is None:
        configured_mask = ROBOTWIN_DELTA_ACTION_DIM_MASK
    if len(configured_mask) != action_dim:
        raise ValueError(
            "data.delta_action_dim_mask must match RobotWin action width: "
            f"expected {action_dim}, got {len(configured_mask)}"
        )
    processor = RobotWinFastWAMProcessor(
        shape_meta,
        1 if observation_only_video else num_frames,
        image_size=image_size,
        delta_action_dim_mask=configured_mask,
    )
    dataset_dirs = _dataset_roots(config)
    dataset = RobotWinVideoDataset(
        dataset_dirs=[str(item) for item in dataset_dirs],
        shape_meta=shape_meta,
        processor=processor,
        text_embedding_cache_dir=_path(config["text_embedding_cache_dir"]),
        pretrained_norm_stats=_path(config["pretrained_norm_stats"]),
        num_frames=num_frames,
        context_len=int(config.get("context_len", 128)),
        val_set_proportion=float(config.get("val_set_proportion", 0.0)),
        is_training_set=bool(config.get("is_training_set", split == "train")),
        global_sample_stride=int(config.get("global_sample_stride", 1)),
        action_video_freq_ratio=int(config.get("action_video_freq_ratio", 4)),
        skip_padding_as_possible=bool(config.get("skip_padding_as_possible", False)),
        max_padding_retry=int(config.get("max_padding_retry", 3)),
        video_backend=config.get("video_backend"),
        observation_only_video=observation_only_video,
        video_size=config.get("video_size", (384, 320)),
    )
    logger.info("[data] built RobotWin FastWAM {} dataset size={}", split, len(dataset))
    return DatasetSliceRepeat(
        dataset,
        max_samples=config.get("max_samples"),
        dataset_repeat=config.get("dataset_repeat", 1),
    )


@DATA_REGISTER("robotwin_fastwam_dataset")
def build_robotwin_fastwam_dataset(data_config, train_or_val="train"):
    dataset = _build_robotwin_dataset(data_config, train_or_val)
    if data_config.get("return_dataset", False):
        return dataset

    sampler = None
    shuffle = bool(data_config.get("shuffle", train_or_val == "train"))
    world_size = get_data_parallel_world_size()
    if train_or_val == "train" and world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=get_data_parallel_rank(),
            shuffle=shuffle,
            drop_last=bool(data_config.get("drop_last", False)),
        )
        shuffle = False
    return DataLoader(
        dataset,
        batch_size=int(data_config.get("batch_size", 1)),
        shuffle=shuffle,
        sampler=sampler,
        num_workers=int(data_config.get("num_workers", 4)),
        pin_memory=bool(data_config.get("pin_memory", torch.cuda.is_available())),
        drop_last=bool(data_config.get("drop_last", False)),
    )
