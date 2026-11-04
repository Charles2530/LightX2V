import atexit
from copy import deepcopy

from loguru import logger

from lightx2v_train.runtime.distributed import is_main_process


def build_monitor(config):
    logging_config = config.get("logging", {})
    if logging_config.get("wandb", {}).get("enable", False):
        return WandBMonitor(config)
    swanlab_config = logging_config.get("swanlab", {})
    if not swanlab_config.get("enable", False):
        return NoopMonitor()
    return SwanLabMonitor(config, swanlab_config)


class NoopMonitor:
    def log_metrics(self, metrics, step=None):
        return

    def finish(self):
        return


class WandBMonitor:
    def __init__(self, config):
        self._run = None
        if not is_main_process():
            return

        import wandb

        safe_config = deepcopy(config)
        safe_config.get("logging", {}).get("swanlab", {}).pop("api_key", None)
        safe_config.get("logging", {}).get("wandb", {}).pop("api_key", None)
        self._run = wandb.init(config=safe_config)
        atexit.register(self.finish)
        logger.info("[monitor] Weights & Biases enabled")

    def log_metrics(self, metrics, step=None):
        if self._run is None:
            return
        values = {key: value.item() if hasattr(value, "item") else value for key, value in metrics.items()}
        if values:
            self._run.log(values, step=step)

    def finish(self):
        if self._run is not None:
            self._run.finish()
            self._run = None


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
