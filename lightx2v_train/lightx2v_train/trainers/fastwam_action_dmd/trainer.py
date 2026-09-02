import os
import time
from copy import deepcopy

import torch
import torch.distributed as dist
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import (
    CachedActionDenoiser,
    build_action_distill_condition,
    sample_action_one_step,
    sample_action_teacher,
)
from lightx2v_train.runtime.distributed import (
    barrier,
    get_data_parallel_group,
    get_data_parallel_world_size,
    get_world_size,
    is_distributed,
    is_main_process,
    is_sequence_parallel_enabled,
    reduce_mean,
)
from lightx2v_train.runtime.monitor import build_monitor
from lightx2v_train.trainers.dmd.math import dmd_loss
from lightx2v_train.utils.registry import TRAINER_REGISTER

from .checkpoint import ActionDmdCheckpointManager
from .config import FastWAMActionDmdConfig
from .roles import ActionDmdRoles, attach_video_role


def _masked_mse(prediction, target, valid_mask):
    error = F.mse_loss(prediction.float(), target.float(), reduction="none")
    if valid_mask is None:
        return error.mean()
    mask = valid_mask.to(device=error.device, dtype=error.dtype).unsqueeze(-1).expand_as(error)
    reduce_dims = tuple(range(1, error.ndim))
    valid_per_sample = mask.sum(dim=reduce_dims).clamp(min=1.0)
    return ((error * mask).sum(dim=reduce_dims) / valid_per_sample).mean()


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


def _optimizer(parameters, config):
    return torch.optim.AdamW(
        parameters,
        lr=float(config.get("learning_rate", 1e-4)),
        betas=(float(config.get("adam_beta1", 0.9)), float(config.get("adam_beta2", 0.95))),
        weight_decay=float(config.get("weight_decay", 0.0)),
        eps=float(config.get("adam_epsilon", 1e-8)),
    )


def _broadcast_parameters(parameters):
    """Synchronize parameters that are outside the action DDP wrappers."""
    if not parameters or not is_distributed():
        return
    group = get_data_parallel_group()
    for parameter in parameters:
        dist.broadcast(parameter.data, src=0, group=group)


def _all_reduce_gradients(parameters):
    """Average gradients for the shared video role across data-parallel ranks."""
    if not parameters or not is_distributed():
        return
    group = get_data_parallel_group()
    world_size = get_data_parallel_world_size()

    # The video LoRA role contains hundreds of small matrices.  Coalesce them
    # into one collective per device/dtype bucket instead of launching one NCCL
    # operation per parameter on every training iteration.
    buckets = {}
    for parameter in parameters:
        buckets.setdefault((parameter.device, parameter.dtype), []).append(parameter)
    for bucket in buckets.values():
        flat = torch.cat(
            [
                (parameter.grad.detach() if parameter.grad is not None else torch.zeros_like(parameter)).reshape(-1)
                for parameter in bucket
            ]
        )
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        flat.div_(world_size)
        offset = 0
        for parameter in bucket:
            length = parameter.numel()
            averaged = flat[offset : offset + length].view_as(parameter)
            if parameter.grad is None:
                parameter.grad = averaged.clone()
            else:
                parameter.grad.copy_(averaged)
            offset += length


@TRAINER_REGISTER("fastwam_action_dmd")
class FastWAMActionDmdTrainer:
    def __init__(self, config):
        self.config = config
        # Freeze the exact launch configuration. The source YAML can be edited
        # while a long-running job is active, so checkpoint metadata must not
        # be reconstructed from that mutable file at save time.
        self.runtime_config = deepcopy(config)
        self.training_config = config["training"]
        self.inference_config = config.get("inference", {})
        self.logging_config = config.get("logging", {})
        self.parsed = FastWAMActionDmdConfig.from_mapping(config)
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
        self.video_expert = None
        self.video_params = []
        self.video_optimizer = None
        self.video_scheduler = None
        self.video_anchor_state = {}
        self.video_anchor_device_state = {}
        self.checkpoints = ActionDmdCheckpointManager(self)
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
        self.monitor = build_monitor(self.config)

    def set_model(self, model):
        self.model = model

    def set_data(self, dataloader_train, dataloader_eval=None):
        self.dataloader_train = dataloader_train
        self.dataloader_eval = dataloader_eval

    def _build_scheduler(self, optimizer, *, num_training_steps=None, num_warmup_steps=None):
        return get_scheduler(
            self.training_config.get("lr_scheduler", "constant"),
            optimizer=optimizer,
            num_warmup_steps=(int(self.training_config.get("lr_warmup_iters", 0)) if num_warmup_steps is None else int(num_warmup_steps)),
            num_training_steps=self.max_train_iters if num_training_steps is None else int(num_training_steps),
        )

    def setup(self):
        sequence_parallel = self.config.get("distributed", {}).get("sequence_parallel", {})
        sequence_parallel_enabled = sequence_parallel.get("enabled", False) if isinstance(sequence_parallel, dict) else bool(sequence_parallel)
        if sequence_parallel_enabled:
            raise ValueError("fastwam_action_dmd does not support sequence parallelism.")
        if is_sequence_parallel_enabled():
            raise ValueError("fastwam_action_dmd does not support sequence parallelism.")
        module = self.model.unwrap_module()
        module.eval()
        module.requires_grad_(False)
        self.roles = ActionDmdRoles.build(module.action_expert, self.parsed)
        if self.parsed.unfreeze_video:
            if self.parsed.video is None:
                raise RuntimeError("Video DMD is enabled but no video role configuration was parsed.")
            self.video_expert = attach_video_role(module, self.parsed.video)
            self.video_expert.train()
            module.mot.train()
            if self.training_config.get("gradient_checkpointing", False):
                video_module = self.video_expert.get_base_model() if hasattr(self.video_expert, "get_base_model") else self.video_expert
                video_module.use_gradient_checkpointing = True
            self.video_params = ActionDmdRoles.trainable_parameters(self.video_expert)
            if not self.video_params:
                raise RuntimeError("Video DMD is enabled but has no trainable video parameters.")
            # PEFT creates adapters independently in each process. Broadcast
            # before taking the anchor snapshot so every rank starts identically.
            _broadcast_parameters(self.video_params)
            self.video_anchor_state = self._capture_video_anchor()
        self.student_denoiser = CachedActionDenoiser(self.roles.student, module.mot)
        self.fake_denoiser = CachedActionDenoiser(self.roles.fake, module.mot)
        self.teacher_denoiser = CachedActionDenoiser(self.roles.teacher, module.mot).eval()
        if self.training_config.get("gradient_checkpointing", False):
            self.student_denoiser.action_module().use_gradient_checkpointing = True
            self.fake_denoiser.action_module().use_gradient_checkpointing = True
            module.mot.train()
            if self.video_expert is None:
                module.video_expert.eval()
            else:
                self.video_expert.train()
        self.student_params = ActionDmdRoles.trainable_parameters(self.roles.student)
        self.fake_params = ActionDmdRoles.trainable_parameters(self.roles.fake)
        if not self.student_params or not self.fake_params:
            raise RuntimeError("FastWAM action DMD has no trainable student or fake parameters.")

        self.student_optimizer = _optimizer(self.student_params, self.parsed.student.optimizer)
        self.fake_optimizer = _optimizer(self.fake_params, self.parsed.fake.optimizer)
        self.student_scheduler = self._build_scheduler(self.student_optimizer)
        fake_training_iters = max(0, self.max_train_iters - self.parsed.endpoint_warmup_iters)
        self.fake_scheduler = self._build_scheduler(
            self.fake_optimizer,
            num_training_steps=max(1, fake_training_iters * self.parsed.fake_update_ratio),
            num_warmup_steps=0,
        )
        if self.video_params:
            self.video_optimizer = _optimizer(self.video_params, self.parsed.video.optimizer)
            self.video_scheduler = self._build_scheduler(self.video_optimizer)
        if is_distributed():
            device_ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
            group = get_data_parallel_group()
            self.student_denoiser = DistributedDataParallel(
                self.student_denoiser,
                device_ids=device_ids,
                process_group=group,
                find_unused_parameters=False,
            )
            self.fake_denoiser = DistributedDataParallel(
                self.fake_denoiser,
                device_ids=device_ids,
                process_group=group,
                find_unused_parameters=False,
            )

        resume_path, current_iter = self.checkpoints.resolve_resume()
        if resume_path is not None:
            current_iter = self.checkpoints.load(resume_path)
            logger.info("[resume] restored FastWAM action DMD from {} at iteration {}", resume_path, current_iter)
        # Ensure a resumed adapter is identical on every rank as well. The
        # action roles are synchronized by their DDP wrappers; video is not.
        _broadcast_parameters(self.video_params)
        return current_iter

    def _sample_sigma(self, batch_size, device, dtype):
        scheduler = self.model.unwrap_module().train_action_scheduler
        timestep = scheduler.sample_training_t(
            batch_size=batch_size,
            device=device,
            dtype=torch.float32,
        )
        sigma = timestep / float(scheduler.num_train_timesteps)
        sigma = sigma.clamp(min=self.parsed.sigma_min, max=self.parsed.sigma_max)
        return sigma.to(dtype=dtype)

    @staticmethod
    def _expand_sigma(sigma, value):
        return sigma.reshape(sigma.shape[0], *([1] * (value.ndim - 1)))

    def _capture_video_anchor(self):
        if self.video_expert is None:
            return {}
        self.video_anchor_device_state = {}
        return {
            name: parameter.detach().cpu().clone()
            for name, parameter in self.video_expert.named_parameters()
            if parameter.requires_grad
        }

    def _video_anchor_weight(self):
        # ``getattr`` keeps lightweight legacy test/migration trainer objects
        # usable even when they predate the optional video fields.
        return float(getattr(self.parsed, "video_anchor_weight", 0.0))

    def _video_anchor_loss(self, device):
        if not self.video_params or self._video_anchor_weight() <= 0.0:
            return torch.zeros((), device=device, dtype=torch.float32)
        penalties = []
        for name, parameter in self.video_expert.named_parameters():
            if not parameter.requires_grad:
                continue
            anchor = self.video_anchor_device_state.get(name)
            if anchor is None:
                cpu_anchor = self.video_anchor_state.get(name)
                if cpu_anchor is None:
                    continue
                anchor = cpu_anchor.to(device=parameter.device, dtype=torch.float32)
                self.video_anchor_device_state[name] = anchor
            if anchor.device != parameter.device:
                anchor = anchor.to(device=parameter.device, dtype=torch.float32)
                self.video_anchor_device_state[name] = anchor
            penalties.append(F.mse_loss(parameter.float(), anchor, reduction="mean"))
        if not penalties:
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.stack(penalties).mean()

    def _prepare_batch(self, sample, *, track_video_grad=False):
        module = self.model.unwrap_module()
        if track_video_grad:
            # ``build_action_distill_inputs`` keeps VAE/text/proprio encoding
            # detached; only the video expert prefill is intentionally kept in
            # the graph for the opt-in video role.
            with self.model.autocast_context():
                inputs = module.build_action_distill_inputs(sample)
                condition = build_action_distill_condition(module, inputs, requires_grad=True)
        else:
            with torch.no_grad(), self.model.autocast_context():
                inputs = module.build_action_distill_inputs(sample)
                condition = build_action_distill_condition(module, inputs)
        valid_mask = None if inputs["action_is_pad"] is None else ~inputs["action_is_pad"]
        return inputs, condition, valid_mask

    def _student_loss(self, inputs, condition, valid_mask, current_iter):
        target_action = inputs["action"]
        noise = torch.randn_like(target_action)
        module = self.model.unwrap_module()
        generated = sample_action_one_step(
            self.student_denoiser,
            noise,
            condition,
            module.train_action_scheduler.num_train_timesteps,
        )
        video_anchor = self._video_anchor_loss(generated.device)
        video_anchor_weight = self._video_anchor_weight()
        warmup = current_iter < self.parsed.endpoint_warmup_iters
        endpoint_weight = 1.0 if warmup else self.parsed.endpoint_loss_weight
        endpoint_loss = generated.new_zeros((), dtype=torch.float32)
        if endpoint_weight > 0:
            teacher_action = sample_action_teacher(
                self.teacher_denoiser,
                noise,
                condition,
                module.infer_action_scheduler,
                self.parsed.teacher_steps,
            )
            endpoint_loss = _masked_mse(generated, teacher_action, valid_mask)
        if warmup:
            return (
                endpoint_weight * endpoint_loss + video_anchor_weight * video_anchor,
                generated.detach(),
                {
                    "endpoint": endpoint_loss.detach(),
                    "dmd": generated.new_zeros((), dtype=torch.float32),
                    "video_anchor": video_anchor.detach(),
                },
            )

        sigma = self._sample_sigma(generated.shape[0], generated.device, generated.dtype)
        expanded_sigma = self._expand_sigma(sigma, generated)
        renoise = torch.randn_like(generated)
        noisy_action = (1.0 - expanded_sigma) * generated.detach() + expanded_sigma * renoise
        timestep = sigma * float(module.train_action_scheduler.num_train_timesteps)
        fake_was_training = self.fake_denoiser.training
        self.fake_denoiser.eval()
        with torch.no_grad():
            fake_velocity = self.fake_denoiser(noisy_action, timestep, condition)
            teacher_velocity = self.teacher_denoiser(noisy_action, timestep, condition)
            fake_x0 = noisy_action - expanded_sigma * fake_velocity
            teacher_x0 = noisy_action - expanded_sigma * teacher_velocity
        if fake_was_training:
            self.fake_denoiser.train()
        loss_dmd = dmd_loss(
            generated,
            fake_x0,
            teacher_x0,
            norm_clip_min=self.parsed.norm_clip_min,
            mask=valid_mask,
        )
        loss = (
            self.parsed.dmd_loss_weight * loss_dmd
            + endpoint_weight * endpoint_loss
            + video_anchor_weight * video_anchor
        )
        return loss, generated.detach(), {
            "endpoint": endpoint_loss.detach(),
            "dmd": loss_dmd.detach(),
            "video_anchor": video_anchor.detach(),
        }

    def _fake_loss(self, generated, condition, valid_mask):
        module = self.model.unwrap_module()
        self.fake_denoiser.train()
        sigma = self._sample_sigma(generated.shape[0], generated.device, generated.dtype)
        expanded_sigma = self._expand_sigma(sigma, generated)
        noise = torch.randn_like(generated)
        noisy_action = (1.0 - expanded_sigma) * generated + expanded_sigma * noise
        timestep = sigma * float(module.train_action_scheduler.num_train_timesteps)
        velocity = self.fake_denoiser(noisy_action, timestep, condition)
        return _masked_mse(velocity, noise - generated, valid_mask) * self.parsed.fake_loss_weight

    @torch.no_grad()
    def _generate_with_current_student(self, inputs, condition):
        module = self.model.unwrap_module()
        noise = torch.randn_like(inputs["action"])
        return sample_action_one_step(
            self.student_denoiser,
            noise,
            condition,
            module.train_action_scheduler.num_train_timesteps,
        ).detach()

    def _train_fake_updates(self, samples):
        loss_total = 0.0
        grad_total = 0.0
        accumulation = self.gradient_accumulation_iters
        for _ in range(self.parsed.fake_update_ratio):
            self.fake_optimizer.zero_grad(set_to_none=True)
            update_loss = 0.0
            for _ in range(accumulation):
                sample = next(samples)
                inputs, condition, valid_mask = self._prepare_batch(sample)
                condition = condition.detach()
                with self.model.autocast_context():
                    generated = self._generate_with_current_student(inputs, condition)
                    fake_loss = self._fake_loss(generated, condition, valid_mask)
                (fake_loss / accumulation).backward()
                update_loss += float(fake_loss.detach().item()) / accumulation
            fake_grad_norm = torch.nn.utils.clip_grad_norm_(self.fake_params, self.max_grad_norm)
            self.fake_optimizer.step()
            self.fake_scheduler.step()
            self.fake_optimizer.zero_grad(set_to_none=True)
            loss_total += update_loss / self.parsed.fake_update_ratio
            grad_total += float(fake_grad_norm) / self.parsed.fake_update_ratio
        return loss_total, grad_total

    def _step_video_optimizer(self):
        if self.video_optimizer is None:
            return 0.0
        _all_reduce_gradients(self.video_params)
        video_grad_norm = torch.nn.utils.clip_grad_norm_(self.video_params, self.max_grad_norm)
        self.video_optimizer.step()
        self.video_scheduler.step()
        self.video_optimizer.zero_grad(set_to_none=True)
        return float(video_grad_norm)

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
        module = self.model.unwrap_module()
        mot_was_training = module.mot.training
        video_was_training = module.video_expert.training
        module.mot.eval()
        module.video_expert.eval()
        self.roles.student.eval()
        totals = {"student_teacher_l1": 0.0, "student_gt_l1": 0.0}
        count = 0
        generator = torch.Generator(device=module.device).manual_seed(self.eval_seed)
        for sample in self.dataloader_eval:
            batch_size = int(sample["video"].shape[0])
            remaining = self.eval_num_samples - count
            if batch_size > remaining:
                sample = _slice_batch(sample, remaining)
                batch_size = remaining
            inputs, condition, valid_mask = self._prepare_batch(sample)
            noise = torch.randn(
                inputs["action"].shape,
                generator=generator,
                device=inputs["action"].device,
                dtype=inputs["action"].dtype,
            )
            steps = self.model.unwrap_module().train_action_scheduler.num_train_timesteps
            with self.model.autocast_context():
                student = sample_action_one_step(self.student_denoiser, noise, condition, steps)
                teacher = sample_action_teacher(
                    self.teacher_denoiser,
                    noise,
                    condition,
                    self.model.unwrap_module().infer_action_scheduler,
                    self.parsed.teacher_steps,
                )
            totals["student_teacher_l1"] += float(_masked_l1_per_sample(student, teacher, valid_mask).sum().item())
            totals["student_gt_l1"] += float(_masked_l1_per_sample(student, inputs["action"], valid_mask).sum().item())
            count += batch_size
            if count >= self.eval_num_samples:
                break
        if count:
            student_teacher_l1 = reduce_mean(totals["student_teacher_l1"] / count)
            student_gt_l1 = reduce_mean(totals["student_gt_l1"] / count)
        if count and is_main_process():
            logger.info(
                "[eval] iter={} student_teacher_l1={:.6f} student_gt_l1={:.6f}",
                current_iter,
                student_teacher_l1,
                student_gt_l1,
            )
            self.monitor.log_metrics(
                {
                    "eval/student_teacher_l1": student_teacher_l1,
                    "eval/student_gt_l1": student_gt_l1,
                },
                step=current_iter,
            )
        self.roles.student.train()
        module.mot.train(mot_was_training)
        module.video_expert.train(video_was_training)

    def train(self):
        current_iter = self.setup()
        if is_main_process():
            os.makedirs(self.output_dir, exist_ok=True)
        barrier()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.student_optimizer.zero_grad(set_to_none=True)
        self.fake_optimizer.zero_grad(set_to_none=True)
        if self.video_optimizer is not None:
            self.video_optimizer.zero_grad(set_to_none=True)
        samples = self._iter_train_samples()
        started_at = time.perf_counter()

        logger.info(
            "[train] start method=fastwam_action_dmd iter={}/{} world_size={} mbs={} global_batch={} teacher_steps={} warmup_iters={} video_unfreeze={} video_trainable={}",
            current_iter,
            self.max_train_iters,
            get_world_size(),
            int(self.config["data"]["train"]["batch_size"]),
            int(self.config["data"]["train"]["batch_size"]) * get_world_size() * self.gradient_accumulation_iters,
            self.parsed.teacher_steps,
            self.parsed.endpoint_warmup_iters,
            self.video_expert is not None,
            sum(parameter.numel() for parameter in self.video_params),
        )
        while current_iter < self.max_train_iters:
            self.student_optimizer.zero_grad(set_to_none=True)
            for _ in range(self.gradient_accumulation_iters):
                sample = next(samples)
                inputs, condition, valid_mask = self._prepare_batch(
                    sample,
                    track_video_grad=self.video_expert is not None,
                )
                with self.model.autocast_context():
                    student_loss, _, student_metrics = self._student_loss(inputs, condition, valid_mask, current_iter)
                (student_loss / self.gradient_accumulation_iters).backward()

            student_grad_norm = torch.nn.utils.clip_grad_norm_(self.student_params, self.max_grad_norm)
            self.student_optimizer.step()
            self.student_scheduler.step()
            self.student_optimizer.zero_grad(set_to_none=True)
            video_grad_norm = self._step_video_optimizer()

            fake_loss = 0.0
            fake_grad_norm = 0.0
            if current_iter >= self.parsed.endpoint_warmup_iters:
                fake_loss, fake_grad_norm = self._train_fake_updates(samples)

            current_iter += 1
            if current_iter == 1 or current_iter % self.log_every_iters == 0:
                elapsed = max(time.perf_counter() - started_at, 1e-6)
                metrics = {
                    "train/endpoint_loss": reduce_mean(float(student_metrics["endpoint"].item())),
                    "train/dmd_loss": reduce_mean(float(student_metrics["dmd"].item())),
                    "train/fake_loss": reduce_mean(float(fake_loss)),
                    "train/student_grad_norm": reduce_mean(float(student_grad_norm)),
                    "train/fake_grad_norm": reduce_mean(float(fake_grad_norm)),
                    "train/video_anchor_loss": reduce_mean(float(student_metrics["video_anchor"].item())),
                    "train/video_grad_norm": reduce_mean(float(video_grad_norm)),
                    "train/student_lr": self.student_scheduler.get_last_lr()[0],
                    "train/fake_lr": self.fake_scheduler.get_last_lr()[0],
                    "train/video_lr": (
                        self.video_scheduler.get_last_lr()[0] if self.video_scheduler is not None else 0.0
                    ),
                    "train/iters_per_second": current_iter / elapsed,
                    "train/is_dmd_stage": int(current_iter > self.parsed.endpoint_warmup_iters),
                    "train/world_size": get_world_size(),
                    "train/micro_batch_size": int(self.config["data"]["train"]["batch_size"]),
                    "train/global_batch_size": int(self.config["data"]["train"]["batch_size"]) * get_world_size() * self.gradient_accumulation_iters,
                }
                if torch.cuda.is_available():
                    metrics.update(
                        {
                            "system/gpu_memory_allocated_gib": torch.cuda.memory_allocated() / 1024**3,
                            "system/gpu_memory_reserved_gib": torch.cuda.memory_reserved() / 1024**3,
                            "system/gpu_max_memory_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
                        }
                    )
                logger.info(
                    "[train] iter={}/{} stage={} endpoint={:.6f} dmd={:.6f} fake={:.6f} student_grad={:.4f} fake_grad={:.4f} video_grad={:.4f} speed={:.3f} it/s max_mem={:.2f}GiB",
                    current_iter,
                    self.max_train_iters,
                    "warmup" if current_iter <= self.parsed.endpoint_warmup_iters else "dmd",
                    metrics["train/endpoint_loss"],
                    metrics["train/dmd_loss"],
                    metrics["train/fake_loss"],
                    metrics["train/student_grad_norm"],
                    metrics["train/fake_grad_norm"],
                    metrics["train/video_grad_norm"],
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
        logger.info("[train] finished FastWAM action DMD iter={}", current_iter)
        self.monitor.finish()
