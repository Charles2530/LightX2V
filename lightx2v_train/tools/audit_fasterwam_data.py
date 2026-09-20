"""Compare the RobotWin training processor with the original FasterWAM processor."""
import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf
from fasterwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from lightx2v_train.data.robotwin_dataset import _build_robotwin_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root/'lightx2v_train/runs/fasterwam_robotwin_action_1step_consistency_ts10/checkpoint-000030000/config.yaml').read_text())
    dataset = _build_robotwin_dataset(cfg['data']['val'], 'val').dataset
    data_cfg = OmegaConf.create({'data': OmegaConf.load('/mnt/miaohua/charles/codes/FasterWAM/configs/data/robotwin.yaml')})
    native = instantiate(data_cfg.data.train.processor).train()
    native.set_normalizer_from_stats(load_dataset_stats_from_json(cfg['data']['val']['pretrained_norm_stats']))
    result = []
    for index in (0, dataset.lerobot_dataset.episodes[0].length-3):
        raw = dataset.lerobot_dataset[index]
        native_raw = copy.deepcopy(raw)
        native_raw['images'] = {k: (v*255).round().to(torch.uint8) for k,v in raw['images'].items()}
        actual = dataset.processor.preprocess(copy.deepcopy(raw))
        expected = native.preprocess(native_raw)
        record = {'index': index, 'padded_actions': int(raw['action_is_pad'].sum())}
        for name in ('pixel_values','action','proprio'):
            diff = (actual[name]-expected[name]).abs()
            record[name+'_mae'] = float(diff.mean())
            record[name+'_max_abs'] = float(diff.max())
        valid = ~raw['action_is_pad']
        record['valid_action_max_abs'] = float((actual['action'][valid]-expected['action'][valid]).abs().max())
        assert record['valid_action_max_abs'] < 1e-5
        assert record['proprio_max_abs'] < 1e-5
        assert record['pixel_values_max_abs'] < 1e-5
        result.append(record)
        print(json.dumps(record), flush=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
