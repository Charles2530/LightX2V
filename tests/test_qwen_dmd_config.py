import copy
import unittest

from lightx2v_train.runtime import load_config
from lightx2v_train.schedulers import DMDFlowMatchingScheduler
from lightx2v_train.trainers.dmd.config import DmdConfig
from lightx2v_train.utils.quantization import validate_training_quantization


CONFIG_PATH = "/mnt/lm_data_afs/charles/codes/LightX2V/lightx2v_train/configs/train/dmd/qwen_image_2_1_dmd_mxfp4.yaml"


class QwenDmdConfigTest(unittest.TestCase):
    def test_fixed_sigmas_and_quantization(self):
        config = load_config(CONFIG_PATH)
        parsed = DmdConfig.from_mapping(config)
        self.assertEqual(parsed.sigmas, (1.0, 0.95418, 0.89508, 0.704534))
        self.assertEqual(validate_training_quantization(config)["weight"]["bit"], "mxfp4")

        scheduler = DMDFlowMatchingScheduler(config)
        scheduler.set_timesteps(parsed.num_inference_steps, sigmas=parsed.sigmas)
        for actual, expected in zip(scheduler.sigmas.tolist(), (*parsed.sigmas, 0.0)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_invalid_fixed_sigmas_fail(self):
        for sigmas in ([1.0, 0.8, 0.9, 0.2], [1.0, 0.8], [0.9, 0.8, 0.6, 0.4]):
            with self.subTest(sigmas=sigmas):
                config = copy.deepcopy(load_config(CONFIG_PATH))
                config["training"]["dmd"]["sigmas"] = sigmas
                with self.assertRaisesRegex(ValueError, "training.dmd.sigmas"):
                    DmdConfig.from_mapping(config)


if __name__ == "__main__":
    unittest.main()


"""PYTHONPATH=/mnt/lm_data_afs/charles/codes/LightX2V/lightx2v_train /mnt/lm_data_afs/charles/codes/LightX2V/.venv/bin/python /mnt/lm_data_afs/charles/codes/LightX2V/tests/test_qwen_dmd_config.py"""
