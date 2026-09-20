"""Bounded two-step diagnostic of the ORIGINAL consistency trainer.

No checkpoint/LoRA is overwritten, no W&B run is created, and no recipe repair
is applied. Run with torchrun to check real DDP forwards/backwards and syncing.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from lightx2v_train.data import build_data
from lightx2v_train.model_zoo import build_model
from lightx2v_train.runtime import cleanup_distributed, init_distributed
from lightx2v_train.trainers import build_trainer


def main():
    root = Path(__file__).resolve().parents[2]
    cfg = yaml.safe_load((root/'lightx2v_train/runs/fasterwam_robotwin_action_1step_consistency_ts10/checkpoint-000030000/config.yaml').read_text())
    cfg['model']['skip_dit_load_from_pretrain'] = True
    cfg['data']['train'].update(batch_size=1, num_workers=0)
    cfg['training'].update(max_train_iters=2, save_every_iters=0, save_final=False,
                           output_dir=str(root/'alignment_audit_20260916/training_smoke'), lr_warmup_iters=0)
    cfg['logging']['wandb']['enable'] = False
    cfg['logging']['train_log_every_iters'] = 1
    cfg['inference']['infer_every_iters'] = 0
    cfg['resume'] = {'auto_resume': False, 'resume_ckpt_path': None}
    init_distributed(cfg)
    try:
        model = build_model(cfg)
        model.load_components()
        data = build_data(cfg, train_or_val='train')
        trainer = build_trainer(cfg)
        trainer.set_model(model)
        trainer.set_data(data)
        trainer.train()
        digest = hashlib.sha256()
        for parameter in trainer.student_params:
            assert parameter.grad is not None
            assert bool(torch.isfinite(parameter.grad).all())
            digest.update(parameter.detach().cpu().numpy().tobytes())
        hashes = [None] * dist.get_world_size()
        dist.all_gather_object(hashes, digest.hexdigest())
        assert len(set(hashes)) == 1, hashes
        if dist.get_rank() == 0:
            result = dict(world_size=dist.get_world_size(), steps=2, trainable_parameters=len(trainer.student_params),
                          all_gradients_finite=True, all_ranks_weights_identical=True, sha256=hashes[0],
                          scope='mechanical DDP smoke of unchanged observation-only consistency training; not a fix or quality test')
            (root/'alignment_audit_20260916/training_smoke.json').write_text(json.dumps(result,indent=2)+'\n')
            print('TRAINING_SMOKE_DONE',json.dumps(result),flush=True)
    finally:
        cleanup_distributed()


if __name__ == '__main__':
    main()
