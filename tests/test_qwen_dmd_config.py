import copy
import unittest
from unittest.mock import patch

from lightx2v_train.data import build_data, build_sample_processor
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

    def test_mxfp4_variants(self):
        for mode, bit, backward in (
            ("pure_dmd", None, False),
            ("nvfp4_forward", "nvfp4", False),
            ("nvfp4_forward_backward", "nvfp4", True),
            ("mxfp4_forward", "mxfp4", False),
            ("mxfp4_forward_backward", "mxfp4", True),
            ("mxfp4_auto_forward_backward", "mxfp4_auto", True),
        ):
            with self.subTest(mode=mode), patch.dict("os.environ", {"QWEN_QUANT_MODE": mode}):
                config = load_config(CONFIG_PATH)
                quant = validate_training_quantization(config)
                self.assertEqual(config["training"]["output_dir"].split("/")[-1], mode)
                if bit is None:
                    self.assertIsNone(quant)
                else:
                    self.assertEqual(quant["weight"]["bit"], bit)
                    self.assertEqual(quant["act"]["bit"], bit)
                    self.assertEqual(quant["grad_output"] is not None, backward)

    def test_validation_dataset_supports_image_inference(self):
        config = load_config(CONFIG_PATH)
        processor = build_sample_processor(config)
        dataset = build_data(config, "val", sample_processor=processor).dataset
        self.assertIs(dataset.sample_processor, processor)
        self.assertEqual(len(dataset), 32)
        self.assertEqual(processor.infer_target_size(dataset[0], 1024, 1024), (1024, 1024))


if __name__ == "__main__":
    unittest.main()


"""PYTHONPATH=/mnt/lm_data_afs/charles/codes/LightX2V/lightx2v_train /mnt/lm_data_afs/charles/codes/LightX2V/.venv/bin/python /mnt/lm_data_afs/charles/codes/LightX2V/tests/test_qwen_dmd_config.py"""
