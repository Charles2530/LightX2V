import atexit
import os
from copy import deepcopy

from loguru import logger

from lightx2v_train.runtime.distributed import is_main_process


def build_monitor(config):
    wandb_config = config.get("logging", {}).get("wandb", {})
    if wandb_config.get("enable", False):
        return WandbMonitor(config, wandb_config)
    swanlab_config = config.get("logging", {}).get("swanlab", {})
    if not swanlab_config.get("enable", False):
        return NoopMonitor()
    return SwanLabMonitor(config, swanlab_config)


class NoopMonitor:
    def log_metrics(self, metrics, step=None):
        return

    def finish(self):
        return


class WandbMonitor:
    def __init__(self, config, wandb_config):
        self._run = None
        self._enabled = is_main_process()
        if not self._enabled:
            return

        import wandb

        if not os.environ.get("WANDB_API_KEY"):
            raise ValueError("WANDB_API_KEY must be set when logging.wandb.enable=true.")

        init_kwargs = {
            "project": wandb_config.get("project", "lightx2v-train"),
            "config": self._config_without_secrets(config),
            "dir": os.path.expanduser(str(config.get("training", {}).get("output_dir", "."))),
        }
        for key in ("entity", "group", "job_type", "name", "notes", "tags"):
            if wandb_config.get(key) is not None:
                init_kwargs[key] = wandb_config[key]
        run_id = os.environ.get("WANDB_RUN_ID") or wandb_config.get("id")
        if run_id:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = wandb_config.get("resume", "allow")

        self._run = wandb.init(**init_kwargs)
        atexit.register(self.finish)
        logger.info("[monitor] Weights & Biases enabled project={} run={}", init_kwargs["project"], self._run.name)

    @staticmethod
    def _config_without_secrets(config):
        safe_config = deepcopy(config)
        safe_config.get("logging", {}).get("wandb", {}).pop("api_key", None)
        return safe_config

    def log_metrics(self, metrics, step=None):
        if not self._enabled or self._run is None:
            return
        values = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            values[key] = value
        if values:
            self._run.log(values, step=step)

    def finish(self):
        if not self._enabled or self._run is None:
            return
        self._run.finish()
        self._run = None
        self._enabled = False


class SwanLabMonitor:
    def __init__(self, config, swanlab_config):
        self._swanlab = None
        self._enabled = is_main_process()
        if not self._enabled:
            return

        import swanlab

        api_key = swanlab_config.get("api_key")
        if not api_key:
            raise ValueError("logging.swanlab.api_key must be set when logging.swanlab.enable=true.")
        swanlab.login(api_key=api_key)

        init_kwargs = {}
        if swanlab_config.get("project") is not None:
            init_kwargs["project"] = swanlab_config["project"]
        if swanlab_config.get("name") is not None:
            init_kwargs["experiment_name"] = swanlab_config["name"]
        init_kwargs["config"] = self._config_without_secrets(config)

        self._swanlab = swanlab
        self._swanlab.init(**init_kwargs)
        atexit.register(self.finish)
        logger.info("[monitor] SwanLab enabled")

    @staticmethod
    def _config_without_secrets(config):
        safe_config = deepcopy(config)
        safe_config.get("logging", {}).get("swanlab", {}).pop("api_key", None)
        return safe_config

    def log_metrics(self, metrics, step=None):
        if not self._enabled:
            return
        values = {}
        for key, value in metrics.items():
            if hasattr(value, "item"):
                value = value.item()
            values[key] = value
        if values:
            self._swanlab.log(values, step=step)

    def finish(self):
        if not self._enabled:
            return
        self._swanlab.finish()
        self._enabled = False
