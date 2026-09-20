import os
import time
from copy import deepcopy

import torch
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.native.wan.fasterwam.action_distill import (
    CachedActionDenoiser,
    build_action_distill_condition,
    sample_action_one_step,
    sample_action_teacher,
)
from lightx2v_train.runtime.distributed import (
    barrier,
    get_data_parallel_group,
    get_world_size,
    is_distributed,
    is_main_process,
    is_sequence_parallel_enabled,
    reduce_mean,
)
from lightx2v_train.runtime.monitor import build_monitor
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .checkpoint import ActionConsistencyCheckpointManager
from .config import FastWAMActionConsistencyConfig
from .roles import ActionConsistencyRoles


def shifted_consistency_pair(base_sigma, shift, target_steps):
    """Apply FastWAM's shift to a pair separated by one distilled step."""
    sigma_end = (base_sigma - 1.0 / target_steps).clamp(min=0.0)

    def shift_sigma(value):
        return shift * value / (1.0 + (shift - 1.0) * value)

    return shift_sigma(base_sigma), shift_sigma(sigma_end)


def _expand_sigma(sigma, value):
    return sigma.reshape(sigma.shape[0], *([1] * (value.ndim - 1)))


def _masked_mean(error, valid_mask):
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    return (error * mask).sum() / mask.sum().clamp(min=1.0)


def _masked_pseudo_huber(prediction, target, valid_mask, c):
    difference = prediction.float() - target.float()
    return _masked_mean(torch.sqrt(difference.square() + c**2) - c, valid_mask)


def _masked_mse(prediction, target, valid_mask):
    return _masked_mean(F.mse_loss(prediction.float(), target.float(), reduction="none"), valid_mask)


def _masked_l1_per_sample(prediction, target, valid_mask):
    error = (prediction.float() - target.float()).abs()
    reduce_dims = tuple(range(1, error.ndim))
    if valid_mask is None:
        return error.mean(dim=reduce_dims)
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    return (error * mask).sum(dim=reduce_dims) / mask.sum(dim=reduce_dims).clamp(min=1.0)


def _slice_batch(value, size):
    if isinstance(value, torch.Tensor):
        return value[:size]
    if isinstance(value, dict):
        return {key: _slice_batch(item, size) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return value[:size]
    return value


@TRAINER_REGISTER("fasterwam_action_consistency")
class FasterWAMActionConsistencyTrainer:
    config_class = FastWAMActionConsistencyConfig

    def __init__(self, config):
        self.config = config
        self.runtime_config = deepcopy(config)
        self.training_config = config["training"]
        self.inference_config = config.get("inference", {})
        self.logging_config = config.get("logging", {})
        self.parsed = self.config_class.from_mapping(config)
        self.output_dir = self.training_config["output_dir"]
        self.max_train_iters = int(self.training_config["max_train_iters"])
        self.gradient_accumulation_iters = max(1, int(self.training_config.get("gradient_accumulation_iters", 1)))
        self.max_grad_norm = float(self.training_config.get("max_grad_norm", 1.0))
        self.save_every_iters = int(self.training_config.get("save_every_iters", 0) or 0)
        self.save_total_limit = int(self.training_config.get("save_total_limit", 3))
        self.save_final = bool(self.training_config.get("save_final", True))
        self.log_every_iters = max(1, int(self.logging_config.get("train_log_every_iters", 10)))
        self.eval_every_iters = int(self.inference_config.get("infer_every_iters", 0) or 0)
        self.eval_num_samples = max(1, int(self.inference_config.get("num_samples", 1)))
        self.eval_seed = int(self.inference_config.get("seed", 42))
        self.checkpoints = ActionConsistencyCheckpointManager(self)
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
        self.monitor = build_monitor(config)

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def setup(self):
        sequence_parallel = self.config.get("distributed", {}).get("sequence_parallel", {})
        sequence_parallel_enabled = sequence_parallel.get("enabled", False) if isinstance(sequence_parallel, dict) else bool(sequence_parallel)
        if sequence_parallel_enabled or is_sequence_parallel_enabled():
            raise ValueError(f"{self.training_config['method']} does not support sequence parallelism.")

        module = self.model.unwrap_module()
        module.eval().requires_grad_(False)
        self.roles = ActionConsistencyRoles.build(module.action_expert, self.parsed.student)
        self.student_denoiser = CachedActionDenoiser(self.roles.student, module.mot)
        self.target_denoiser = CachedActionDenoiser(self.roles.target, module.mot).eval()
        self.teacher_denoiser = CachedActionDenoiser(self.roles.teacher, module.mot).eval()
        if self.training_config.get("gradient_checkpointing", False):
            self.student_denoiser.action_module().use_gradient_checkpointing = True

        self.student_params = self.roles.trainable_parameters
        if not self.student_params:
            raise RuntimeError("FastWAM action training has no trainable student parameters.")
        optimizer_config = self.parsed.student.optimizer
        self.optimizer = torch.optim.AdamW(
            self.student_params,
            lr=float(optimizer_config.get("learning_rate", 1e-4)),
            betas=(float(optimizer_config.get("adam_beta1", 0.9)), float(optimizer_config.get("adam_beta2", 0.95))),
            weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
            eps=float(optimizer_config.get("adam_epsilon", 1e-8)),
        )
        self.scheduler = get_scheduler(
            self.training_config.get("lr_scheduler", "constant"),
            optimizer=self.optimizer,
            num_warmup_steps=int(self.training_config.get("lr_warmup_iters", 0)),
            num_training_steps=self.max_train_iters,
        )
        if is_distributed():
            self.student_denoiser = DistributedDataParallel(
                self.student_denoiser,
                device_ids=[torch.cuda.current_device()] if torch.cuda.is_available() else None,
                process_group=get_data_parallel_group(),
                find_unused_parameters=False,
            )
        # DDP broadcasts the online student; initialize every rank's EMA from it.
        self.roles.copy_student_to_target()

        resume_path, current_iter = self.checkpoints.resolve_resume()
        if resume_path is not None:
            current_iter = self.checkpoints.load(resume_path)
            logger.info("[resume] restored {} from {} at iteration {}", self.training_config["method"], resume_path, current_iter)
        return current_iter

    def _prepare_batch(self, sample, *, video_generator=None):
        module = self.model.unwrap_module()
        # The released inference path explicitly manages BF16/FP32 inside VAE
        # and VideoDiT. An outer autocast changes the frozen KV numerically.
        precision_context = (
            torch.autocast(device_type=self.model.device.type, enabled=False)
            if self.parsed.video_conditioning == "one_pass_future_cache"
            else self.model.autocast_context()
        )
        with torch.no_grad(), precision_context:
            inputs = module.build_inputs(sample)
            condition = build_action_distill_condition(
                module, inputs, video_conditioning=self.parsed.video_conditioning, generator=video_generator
            )
        valid_mask = None if inputs["action_is_pad"] is None else ~inputs["action_is_pad"]
        return inputs, condition, valid_mask

    def _sigma_pair(self, action):
        scheduler = self.model.unwrap_module().train_action_scheduler
        base_sigma = torch.rand(action.shape[0], device=action.device, dtype=torch.float32)
        sigma_start, sigma_end = shifted_consistency_pair(base_sigma, scheduler.shift, self.parsed.target_steps)
        return sigma_start.to(action.dtype), sigma_end.to(action.dtype)

    @torch.no_grad()
    def _teacher_rollout_inputs(self, noise, noisy_action, sigma_start, condition, teacher_velocity):
        """Select the teacher interval without changing the student/consistency inputs."""
        if self.parsed.teacher_start == "1":
            noisy_action = noise
            sigma_start = torch.ones_like(sigma_start)
            num_timesteps = self.model.unwrap_module().train_action_scheduler.num_train_timesteps
            teacher_velocity = self.teacher_denoiser(noisy_action, sigma_start * num_timesteps, condition)
        sigma_end = torch.zeros_like(sigma_start, dtype=torch.float32)
        if self.parsed.teacher_end == "r":
            sigma_end = torch.rand_like(sigma_end) * sigma_start.float()
        return noisy_action, sigma_start, sigma_end, teacher_velocity

    @torch.no_grad()
    def _teacher_targets(self, noisy_action, sigma_start, condition, teacher_velocity, *, sigma_end=None):
        """Return interval mean velocity and x0, extrapolating if the rollout stops above zero."""
        steps = self.parsed.teacher_reference_steps
        num_timesteps = self.model.unwrap_module().train_action_scheduler.num_train_timesteps
        action = noisy_action.float()
        sigma = sigma_start.float()
        sigma_end = torch.zeros_like(sigma) if sigma_end is None else sigma_end.float()
        delta = _expand_sigma((sigma - sigma_end) / steps, action)
        velocity = teacher_velocity.float()
        velocity_sum = torch.zeros_like(action)
        for index in range(steps):
            if index:
                timestep = ((sigma * (1.0 - index / steps) + sigma_end * (index / steps)) * num_timesteps).to(sigma_start.dtype)
                velocity = self.teacher_denoiser(action.to(noisy_action.dtype), timestep, condition).float()
            velocity_sum += velocity
            action = action - delta * velocity
        # Average directly to avoid division by zero for vanishing rollout intervals.
        mean_velocity = velocity_sum / steps
        return mean_velocity, action - _expand_sigma(sigma_end, action) * mean_velocity

    def _loss(self, inputs, condition, valid_mask):
        action = inputs["action"]
        noise = torch.randn_like(action)
        sigma_start, sigma_end = self._sigma_pair(action)
        sigma_start_expanded = _expand_sigma(sigma_start, action)
        sigma_end_expanded = _expand_sigma(sigma_end, action)
        noisy_action = (1.0 - sigma_start_expanded) * action + sigma_start_expanded * noise
        num_timesteps = float(self.model.unwrap_module().train_action_scheduler.num_train_timesteps)
        timestep_start = sigma_start * num_timesteps
        timestep_end = sigma_end * num_timesteps
        flow_target = noise - action
        x0_target = action

        with torch.no_grad():
            teacher_velocity = self.teacher_denoiser(noisy_action, timestep_start, condition)
            endpoint_action = noisy_action + (sigma_end_expanded - sigma_start_expanded) * teacher_velocity
            if self.parsed.flow_target == "teacher":
                teacher_action, teacher_start, teacher_end, rollout_velocity = self._teacher_rollout_inputs(noise, noisy_action, sigma_start, condition, teacher_velocity)
                flow_target, x0_target = self._teacher_targets(teacher_action, teacher_start, condition, rollout_velocity, sigma_end=teacher_end)

        student_velocity = self.student_denoiser(noisy_action, timestep_start, condition)
        student_x0 = noisy_action - sigma_start_expanded * student_velocity
        with torch.no_grad():
            target_velocity = self.target_denoiser(endpoint_action, timestep_end, condition)
            target_x0 = endpoint_action - sigma_end_expanded * target_velocity

        consistency_loss = _masked_pseudo_huber(student_x0, target_x0, valid_mask, self.parsed.huber_c)
        flow_loss = _masked_mse(student_velocity, flow_target, valid_mask)
        x0_loss = _masked_mse(student_x0, x0_target, valid_mask)
        supervision_loss = flow_loss if self.parsed.supervision_type == "flow" else x0_loss
        loss = self.parsed.consistency_loss_weight * consistency_loss + self.parsed.flow_loss_weight * supervision_loss
        return loss, {"consistency": consistency_loss.detach(), "flow": flow_loss.detach(), "x0": x0_loss.detach()}

    def _iter_train_samples(self):
        epoch = 0
        while True:
            sampler = getattr(self.dataloader_train, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            yield from self.dataloader_train
            epoch += 1

    @torch.no_grad()
    def evaluate(self, current_iter):
        if self.dataloader_eval is None:
            return
        totals = {"ema_teacher_l1": 0.0, "ema_gt_l1": 0.0}
        count = 0
        generator = torch.Generator(device=self.model.unwrap_module().device).manual_seed(self.eval_seed)
        video_generator = torch.Generator(device="cpu").manual_seed(self.eval_seed)
        for sample in self.dataloader_eval:
            batch_size = int(sample["video"].shape[0])
            remaining = self.eval_num_samples - count
            if batch_size > remaining:
                sample = _slice_batch(sample, remaining)
                batch_size = remaining
            inputs, condition, valid_mask = self._prepare_batch(sample, video_generator=video_generator)
            noise = torch.randn(inputs["action"].shape, generator=generator, device=inputs["action"].device, dtype=inputs["action"].dtype)
            module = self.model.unwrap_module()
            with self.model.autocast_context():
                ema_action = sample_action_one_step(self.target_denoiser, noise, condition, module.train_action_scheduler.num_train_timesteps)
                teacher_action = sample_action_teacher(self.teacher_denoiser, noise, condition, module.infer_action_scheduler, self.parsed.teacher_reference_steps)
            totals["ema_teacher_l1"] += float(_masked_l1_per_sample(ema_action, teacher_action, valid_mask).sum().item())
            totals["ema_gt_l1"] += float(_masked_l1_per_sample(ema_action, inputs["action"], valid_mask).sum().item())
            count += batch_size
            if count >= self.eval_num_samples:
                break
        metrics = {f"eval/{name}": reduce_mean(total / count) for name, total in totals.items()} if count else {}
        if count and is_main_process():
            logger.info("[eval] iter={} ema_teacher_l1={:.6f} ema_gt_l1={:.6f}", current_iter, metrics["eval/ema_teacher_l1"], metrics["eval/ema_gt_l1"])
            self.monitor.log_metrics(metrics, step=current_iter)

    def _training_details(self):
        return (
            f"target_steps={self.parsed.target_steps} ema_decay={self.parsed.ema_decay} "
            f"flow_target={self.parsed.flow_target} supervision_type={self.parsed.supervision_type} "
            f"teacher_start={self.parsed.teacher_start} teacher_end={self.parsed.teacher_end} "
            f"teacher_reference_steps={self.parsed.teacher_reference_steps} "
            f"video_conditioning={self.parsed.video_conditioning}"
        )

    def train(self):
        current_iter = self.setup()
        start_iter = current_iter
        barrier()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        samples = self._iter_train_samples()
        started_at = time.perf_counter()
        logger.info(
            "[train] start method={} iter={}/{} world_size={} global_batch={} {}",
            self.training_config["method"],
            current_iter,
            self.max_train_iters,
            get_world_size(),
            int(self.config["data"]["train"]["batch_size"]) * get_world_size() * self.gradient_accumulation_iters,
            self._training_details(),
        )
        while current_iter < self.max_train_iters:
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = {}
            for _ in range(self.gradient_accumulation_iters):
                inputs, condition, valid_mask = self._prepare_batch(next(samples))
                with self.model.autocast_context():
                    loss, metrics = self._loss(inputs, condition, valid_mask)
                (loss / self.gradient_accumulation_iters).backward()
                for name, value in metrics.items():
                    accumulated[name] = accumulated.get(name, 0.0) + float(value.item()) / self.gradient_accumulation_iters

            grad_norm = torch.nn.utils.clip_grad_norm_(self.student_params, self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.roles.update_target(self.parsed.ema_decay)
            current_iter += 1

            if current_iter == 1 or current_iter % self.log_every_iters == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                metrics = {
                    **{f"train/{name}_loss": reduce_mean(value) for name, value in accumulated.items()},
                    "train/grad_norm": reduce_mean(float(grad_norm)),
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/iters_per_second": (current_iter - start_iter) / elapsed,
                }
                if torch.cuda.is_available():
                    metrics["system/gpu_max_memory_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
                logger.info(
                    "[train] iter={}/{} {} grad={:.4f} speed={:.3f} it/s max_mem={:.2f}GiB",
                    current_iter,
                    self.max_train_iters,
                    " ".join(f"{name}={metrics[f'train/{name}_loss']:.6f}" for name in accumulated),
                    metrics["train/grad_norm"],
                    metrics["train/iters_per_second"],
                    metrics.get("system/gpu_max_memory_allocated_gib", 0.0),
                )
                self.monitor.log_metrics(metrics, step=current_iter)
            if self.eval_every_iters and current_iter % self.eval_every_iters == 0:
                self.evaluate(current_iter)
            if self.save_every_iters and current_iter % self.save_every_iters == 0:
                self.checkpoints.save(current_iter)

        if self.save_final and (not self.save_every_iters or current_iter % self.save_every_iters):
            self.checkpoints.save(current_iter)
        logger.info("[train] finished {} iter={}", self.training_config["method"], current_iter)
        self.monitor.finish()
