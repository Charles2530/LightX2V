"""Read-only checkpoint diagnostics; all artifacts stay under --output."""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch
import yaml


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='/mnt/afs_1/lvchengtao/code/wam/lightx2v_distill/lightx2v_train')
    p.add_argument('--checkpoint')
    p.add_argument('--summarize', nargs='+', default=None, help='Completed diagnostic output directories')
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--audit', nargs='+', help='Check completed output provenance and paired sample manifests')
    p.add_argument('--components', action='store_true', help='EMA/base branch swaps and action decomposition')
    p.add_argument('--check-merged', help='Optional deployed merged checkpoint for numerical fidelity audit')
    p.add_argument('--task-text', help='Filter validation episodes by text in episode instructions')
    p.add_argument('--phases', type=float, nargs='+', default=[.1,.5,.85])
    p.add_argument('--comparison-checkpoint', help='Additional exported FastWAM checkpoint')
    p.add_argument('--comparison-name', default='comparison')
    p.add_argument('--noise-mode', choices=['sampled_gpu','fixed_cpu'], default='sampled_gpu')
    p.add_argument('--microwave-evidence', action='store_true')
    p.add_argument('--microwave-summary', nargs='+')
    p.add_argument('--microwave-response', action='store_true')
    p.add_argument('--response-summary', nargs='+')
    p.add_argument('--response-manifest', default='runs/open_microwave/fixed_joint/manifest.json')
    p.add_argument('--component-summary', nargs='+')
    p.add_argument('--closedloop', action='store_true')
    p.add_argument('--video-evidence', action='store_true')
    p.add_argument('--plot-deps', help='Optional directory containing matplotlib dependencies')
    p.add_argument('--output', required=True)
    p.add_argument('--episodes', type=int, default=48)
    p.add_argument('--seeds', type=int, nargs='+', default=[20260908, 20260909])
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
    p.add_argument('--teacher-steps', type=int, nargs='+', default=[1, 4, 20])
    p.add_argument('--target-steps', type=int, default=None)
    p.add_argument('--offset', type=int, default=0)
    p.add_argument('--limit', type=int, default=None)
    return p.parse_args()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def signature(path):
    st = path.stat()
    return {'path': str(path), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns}


def gradient_stats(left, right, names):
    groups = {'all': [], 'action': [], 'video': []}
    for i, name in enumerate(names):
        groups['all'].append(i)
        groups[name.split('.', 1)[0]].append(i)
    out = {}
    for group, indices in groups.items():
        if not indices:
            continue
        device = next(g.device for g in left if g is not None)
        dot, aa, bb = (torch.zeros((), device=device, dtype=torch.float64) for _ in range(3))
        for i in indices:
            a, b = left[i], right[i]
            if a is not None:
                aa += a.float().square().sum(dtype=torch.float64)
            if b is not None:
                bb += b.float().square().sum(dtype=torch.float64)
            if a is not None and b is not None:
                dot += (a.float() * b.float()).sum(dtype=torch.float64)
        d, na, nb = dot.item(), aa.sqrt().item(), bb.sqrt().item()
        out[group] = {
            'cos': max(-1., min(1., d / (na * nb))) if na * nb > 1e-20 else None,
            'dot': d, 'norm_left': na, 'norm_right': nb,
            'ratio_weight_0.2': 0.2 * nb / na if na > 1e-20 else None,
            'joint_descent_factor_weight_0.2': 1 + 0.2 * d / (na * na) if na > 1e-20 else None,
        }
    return out


def vector_cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    denom = a.norm() * b.norm()
    return float((a @ b / denom).clamp(-1, 1)) if denom > 1e-20 else None


def self_test(output):
    torch.manual_seed(71)
    a, eps = torch.randn(2,3,dtype=torch.float64), torch.randn(2,3,dtype=torch.float64)
    errors = []
    for sigma in [.1,.3,.9,1.]:
        x = (1-sigma)*a+sigma*eps
        v = torch.randn_like(x,requires_grad=True)
        f = x-sigma*v
        endpoint = torch.randn_like(x)
        fm = (v-(eps-a)).square().mean()
        gt_endpoint = (f-a).square().mean()/sigma**2
        tv = (v-(x-endpoint)/sigma).square().mean()
        te = (f-endpoint).square().mean()/sigma**2
        g1 = torch.autograd.grad(tv,v,retain_graph=True)[0]
        g2 = torch.autograd.grad(te,v)[0]
        wrong = x-sigma*(eps-endpoint)
        errors += [float((fm-gt_endpoint).abs().detach()),float((tv-te).abs().detach()),float((g1-g2).abs().max()),
                   float((wrong-((1-sigma)*a+sigma*endpoint)).abs().max())]
    assert max(errors)<1e-11
    # Same endpoint and same coordinate-wise signs need not imply parameter alignment.
    theta = torch.tensor(0.,dtype=torch.float64,requires_grad=True)
    f = torch.tensor([1.,10.],dtype=torch.float64)+torch.tensor([2.,-1.],dtype=torch.float64)*theta
    c = (torch.sqrt(f.square()+.001**2)-.001).mean()
    t = f.square().mean()
    gc=torch.autograd.grad(c,theta,retain_graph=True)[0]
    gt=torch.autograd.grad(t,theta)[0]
    assert gc*gt<0
    result={'float64_identity_max_error':max(errors),'same_endpoint_parameter_counterexample':{'huber_gradient':float(gc),'mse_gradient':float(gt),'dot':float(gc*gt)},
            'passed':True,'kind':'algebra sanity checks, NOT robot measurements'}
    p=Path(output)
    p.mkdir(parents=True,exist_ok=True)
    dump(p/'self_test.json',result)
    print(json.dumps(result,indent=2))


def audit(args):
    manifests, counts, max_error = set(), {}, 0.
    for directory in args.audit:
        path = Path(directory)
        meta=json.loads((path/'metadata.json').read_text())
        assert meta['completed'] and meta['checkpoint_files_unchanged']
        for info in meta['checkpoint_files']:
            assert signature(Path(info['path'])) == info
        for file,expected in meta['source_hashes'].items():
            assert hashlib.sha256(Path(file).read_bytes()).hexdigest()==expected
        manifests.add(hashlib.sha256((path/'manifest.json').read_bytes()).hexdigest())
        rows=[json.loads(line) for line in (path/'measurements.jsonl').read_text().splitlines()]
        identities=[(r['episode'],r['seed'],r['sigma_requested']) for r in rows]
        assert len(set(identities))==len(identities)
        assert all(r['valid_action_steps']>0 for r in rows)
        max_error=max(max_error,max(r['k1_velocity_identity_max_error'] for r in rows))
        counts[path.name]=len(rows)
    assert len(manifests)==1, 'Different sample manifests'
    result={'passed':True,'paired_manifest_sha256':next(iter(manifests)),
            'rows_by_shard':counts,'total_rows':sum(counts.values()),
            'max_k1_velocity_identity_error':max_error,
            'source_code_hashes_and_checkpoint_file_signatures_unchanged':True}
    p=Path(args.output)
    p.mkdir(parents=True,exist_ok=True)
    dump(p/'audit.json',result)
    print(json.dumps(result,indent=2))


def cluster_interval(rows, value, samples=5000):
    grouped = {}
    for row in rows:
        v = value(row)
        if v is not None and np.isfinite(v):
            grouped.setdefault(row['episode'], []).append(v)
    x = np.array([np.mean(v) for v in grouped.values()])
    if not len(x):
        return {'mean': None, 'ci95': None, 'episodes': 0}
    rng = np.random.default_rng(74393)
    means = x[rng.integers(0, len(x), size=(samples, len(x)))].mean(axis=1)
    return {'mean': float(x.mean()), 'ci95': np.quantile(means,[.025,.975]).tolist(), 'episodes': len(x)}


def cluster_ratio(rows, numerator, denominator, samples=5000):
    grouped={}
    for row in rows:grouped.setdefault(row['episode'],[]).append((numerator(row),denominator(row)))
    x=np.array([np.mean(v,axis=0) for v in grouped.values()])
    rng=np.random.default_rng(74393)
    means=x[rng.integers(0,len(x),size=(samples,len(x)))].mean(axis=1)
    assert np.all(means[:,1]>0)
    return {'ratio_of_means':float(x[:,0].mean()/x[:,1].mean()),
            'ci95':np.quantile(means[:,0]/means[:,1],[.025,.975]).tolist(),'episodes':len(x)}


def summarize(args):
    from collections import defaultdict
    all_rows, metas = defaultdict(list), {}
    for directory in args.summarize:
        p = Path(directory)
        meta = json.loads((p/'metadata.json').read_text())
        if not meta.get('completed'):
            raise ValueError(f'Incomplete experiment: {p}')
        label = Path(meta['args']['checkpoint']).parent.name
        metas[label] = meta
        all_rows[label].extend(json.loads(line) for line in (p/'measurements.jsonl').read_text().splitlines())
    results, table = {}, ['| Checkpoint | Sigma group | Branch | N | C vs GT FM cosine | C vs T20 cosine | FM conflict | T20 conflict | Paired cosine gain [95% CI] |',
                          '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for label, rows in all_rows.items():
        identities = [(r['episode'],r['seed'],r['sigma_requested']) for r in rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f'Duplicate measurements: {label}')
        expected = metas[label]['args']['episodes']*len(metas[label]['args']['seeds'])*len(metas[label]['args']['sigmas'])
        if len(rows) != expected:
            raise ValueError(f'Expected {expected} rows, got {len(rows)} for {label}')
        results[label] = {'rows':len(rows), 'episodes':len(set(r['episode'] for r in rows)), 'groups': {}}
        groups = {'all':rows, 'high_0.9_1.0':[r for r in rows if r['sigma']>=.89]}
        groups.update({str(s):[r for r in rows if r['sigma_requested']==s] for s in sorted(set(r['sigma_requested'] for r in rows))})
        for group, selected in groups.items():
            if not selected:
                continue
            entry = {'n':len(selected), 'parameter':{}, 'output':{}, 'errors':{}}
            for branch in ['all','action','video']:
                if branch not in rows[0]['parameter_pairs']['consistency__gt_fm']:
                    continue
                b = {}
                for pair in rows[0]['parameter_pairs']:
                    stats = {}
                    for key in ['cos','ratio_weight_0.2','joint_descent_factor_weight_0.2']:
                        stats[key] = cluster_interval(selected, lambda r, p=pair, k=key: r['parameter_pairs'][p][branch][k])
                    stats['conflict_rate'] = cluster_interval(selected, lambda r,p=pair: float(r['parameter_pairs'][p][branch]['cos']<0))
                    stats['strong_conflict_rate'] = cluster_interval(selected, lambda r,p=pair: float(r['parameter_pairs'][p][branch]['cos']<-.1))
                    ratios=[r['parameter_pairs'][pair][branch]['ratio_weight_0.2'] for r in selected]
                    stats['ratio_median'] = float(np.median(ratios))
                    stats['joint_ascent_rate_weight_0.2'] = cluster_interval(selected, lambda r,p=pair: float(r['parameter_pairs'][p][branch]['joint_descent_factor_weight_0.2']<0))
                    b[pair] = stats
                b['teacher20_minus_fm_cos'] = cluster_interval(selected,lambda r:r['parameter_pairs']['consistency__teacher_20'][branch]['cos']-r['parameter_pairs']['consistency__gt_fm'][branch]['cos'])
                b['fm_minus_teacher20_conflict_rate'] = cluster_interval(selected,lambda r:float(r['parameter_pairs']['consistency__gt_fm'][branch]['cos']<0)-float(r['parameter_pairs']['consistency__teacher_20'][branch]['cos']<0))
                b['teacher4_minus_fm_cos'] = cluster_interval(selected,lambda r:r['parameter_pairs']['consistency__teacher_4'][branch]['cos']-r['parameter_pairs']['consistency__gt_fm'][branch]['cos'])
                entry['parameter'][branch] = b
                fm, tea, gain = b['consistency__gt_fm'],b['consistency__teacher_20'],b['teacher20_minus_fm_cos']
                table.append(f"| {label} | {group} | {branch} | {len(selected)} | {fm['cos']['mean']:.4f} | {tea['cos']['mean']:.4f} | {fm['conflict_rate']['mean']:.1%} | {tea['conflict_rate']['mean']:.1%} | {gain['mean']:.4f} [{gain['ci95'][0]:.4f}, {gain['ci95'][1]:.4f}] |")
            for pair in rows[0]['output_cos']:
                entry['output'][pair] = {'cos':cluster_interval(selected,lambda r,p=pair:r['output_cos'][p]),
                                         'conflict_rate':cluster_interval(selected,lambda r,p=pair:float(r['output_cos'][p]<0))}
            for key in ['target_mse_gt','student_mse_gt']:
                entry['errors'][key] = cluster_interval(selected,lambda r,k=key:r[k])
            for key in ['target_mse_teacher','teacher_mse_gt','student_mse_teacher','teacher_solver_mse_vs_max']:
                entry['errors'][key] = {k:cluster_interval(selected,lambda r,k1=key,k2=k:r[k1][k2]) for k in rows[0][key]}
            results[label]['groups'][group] = entry
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    dump(output/'summary.json',results)
    (output/'tables.md').write_text('\n'.join(table)+'\n')
    if args.plot_deps:
        sys.path.insert(0,args.plot_deps)
        plot_results(results,output)
    print('Wrote',output/'summary.json',output/'tables.md')


def plot_results(results,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = {
        'fastwam_robotwin_action_1step_consistency_joint_ts10':'Joint consistency + GT FM',
        'fastwam_robotwin_action_1step_consistency_joint_ts10_noflow':'Joint consistency, no FM',
        'fastwam_robotwin_action_1step_consistency_ts10':'Action-only consistency + GT FM',
        'fastwam_robotwin_action_1step_tbsm_teacher':'TBSM teacher (counterfactual C loss)',
    }
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
    pairs=[('gt_fm','GT FM','#b3333d'),('teacher_4','Teacher 4-step','#21805e'),('teacher_20','Teacher 20-step','#2866aa')]
    for ax,(label,model) in zip(axes.flat,results.items()):
        points=[(float(k),v) for k,v in model['groups'].items() if k not in ['all','high_0.9_1.0'] and float(k)>0]
        points.sort(key=lambda v:v[0])
        for key,title,color in pairs:
            stats=[v['parameter']['all']['consistency__'+key]['cos'] for _,v in points]
            means=np.array([v['mean'] for v in stats])
            lo=np.array([v['ci95'][0] for v in stats]); hi=np.array([v['ci95'][1] for v in stats])
            ax.errorbar([s for s,_ in points],means,yerr=[means-lo,hi-means],label=title,color=color,marker='o',capsize=3,linewidth=1.5)
        ax.axhline(0,color='#555555',linewidth=.8,linestyle='--')
        ax.set_title(labels.get(label,label),fontsize=11)
        ax.set_xlabel('Noise level sigma')
        ax.set_ylabel('Parameter gradient cosine with consistency')
        ax.set_ylim(-1.05,1.05)
        ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('48 validation episodes, 2 noise seeds; episode-bootstrap 95% CI',fontsize=12)
    fig.tight_layout()
    fig.savefig(output/'gradient_cosines.png',dpi=180)
    fig.savefig(output/'gradient_cosines.pdf')
    plt.close(fig)


def main():
    args = arguments()
    if args.microwave_response:
        microwave_response(args)
        return
    if args.response_summary:
        response_summary(args)
        return
    if args.components:
        components(args)
        return
    if args.component_summary:
        component_summary(args)
        return
    if args.closedloop:
        closedloop(args)
        return
    if args.video_evidence:
        video_evidence(args)
        return
    if args.microwave_evidence:
        microwave_evidence(args)
        return
    if args.microwave_summary:
        microwave_summary(args)
        return
    if args.audit:
        audit(args)
        return
    if args.self_test:
        self_test(args.output)
        return
    if args.summarize:
        summarize(args)
        return
    if not args.checkpoint:
        raise ValueError('--checkpoint is required for measurement')
    sys.path.insert(0, args.source)
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.data.robotwin_dataset import _build_robotwin_dataset
    from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import CachedActionDenoiser, build_action_distill_condition
    from lightx2v_train.trainers.fastwam_action_consistency.config import ActionStudentConfig
    from lightx2v_train.trainers.fastwam_joint_consistency.roles import ConsistencyRoles, JointConsistencyDenoiser, load_role_state_dict
    from lightx2v_train.trainers.fastwam_joint_consistency.trainer import _masked_mse, _masked_pseudo_huber, shifted_consistency_pair
    from torch.utils.data import default_collate

    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    out = Path(args.output).resolve()
    source = Path(args.source).resolve()
    if out.is_relative_to(source.parent):
        raise ValueError('Output must not be in the read-only auxiliary repository')
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    config = yaml.safe_load((checkpoint / 'config.yaml').read_text())
    ac = config['training'].get('action_consistency', {})
    target_steps = args.target_steps or int(ac.get('target_steps', 10))
    huber_c = float(ac.get('huber_c', .001))
    joint = bool(config['training'].get('train_video', False))

    # Select episode quantiles, not adjacent windows of the first validation episode.
    dataset = _build_robotwin_dataset(config['data']['val'], 'val').dataset
    lr = dataset.lerobot_dataset
    ordered = sorted(range(len(lr.episodes)), key=lambda i: lr.episodes[i].index)
    positions = [ordered[i] for i in np.linspace(0, len(ordered)-1, min(args.episodes, len(ordered)), dtype=int)]
    manifest = []
    for pos in positions:
        ep = lr.episodes[pos]
        rng = np.random.default_rng(92371 + ep.index)
        frame = int(rng.integers(0, max(1, ep.length - dataset.num_frames + 1)))
        start = 0 if pos == 0 else lr._episode_ends[pos-1]
        data = lr._load_episode(pos)
        task_index = int(data['task_index'][frame])
        manifest.append({'episode': ep.index, 'frame': frame, 'index': start+frame,
                         'task_index': task_index, 'prompt_task': ep.tasks[task_index], 'episode_length': ep.length})
    dump(out / 'manifest.json', manifest)
    files = [checkpoint / 'config.yaml', checkpoint / 'student_action.pt', checkpoint / 'ema_action.pt']
    if joint:
        files += [checkpoint / 'student_video.pt', checkpoint / 'ema_video.pt']
    before = [signature(p) for p in files]
    source_files = [source / 'lightx2v_train/trainers/fastwam_joint_consistency/trainer.py',
                    source / 'lightx2v_train/model_zoo/native/wan/fastwam/action_distill.py']
    metadata = {'args': vars(args), 'checkpoint_method': config['training']['method'],
                'joint': joint, 'diagnostic_target_steps': target_steps, 'huber_c': huber_c,
                'original_flow_weight': ac.get('flow_loss_weight'), 'checkpoint_files': before,
                'source_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(),
                'batch_size': 1, 'student_eval_mode': True, 'autocast': 'bfloat16, cache disabled',
                'teacher_solver': 'Euler, uniform in inverse-shift coordinate from given sigma to 0; fp32 accumulation',
                'sigma_grid_measure': ('original shifted-uniform training distribution' if args.sigmas == [-1.0]
                                       else 'equal-weight diagnostic grid; NOT original training distribution')}
    dump(out / 'metadata.json', metadata)
    print('LOADING', checkpoint, 'joint=', joint, flush=True)
    model = build_model(config)
    model.load_components()
    module = model.unwrap_module()
    module.eval().requires_grad_(False)
    student_config = ActionStudentConfig.from_mapping(config['training']['student'])
    roles = ConsistencyRoles.build(module.action_expert, student_config)
    video_roles = ConsistencyRoles.build(module.video_expert, student_config) if joint else None
    for component, rr in [('action', roles), ('video', video_roles)]:
        if rr is None:
            continue
        for role, file_role in [('student', 'student'), ('target', 'ema')]:
            payload = torch.load(checkpoint / f'{file_role}_{component}.pt', map_location='cpu', weights_only=True)
            load_role_state_dict(getattr(rr, role), student_config.train_type, payload)
            del payload
    denoisers = {}
    for role in ['student', 'target', 'teacher']:
        denoisers[role] = (JointConsistencyDenoiser(getattr(roles, role), getattr(video_roles, role), module)
                           if joint else CachedActionDenoiser(getattr(roles, role), module.mot))
        denoisers[role].eval()
    params, names = [], []
    for component, rr in [('action', roles), ('video', video_roles)]:
        if rr:
            for name, parameter in rr.student.named_parameters():
                if parameter.requires_grad:
                    params.append(parameter)
                    names.append(component + '.' + name)
    metadata['trainable_parameters'] = {g: sum(p.numel() for n,p in zip(names,params) if n.startswith(g+'.')) for g in ['action','video']}
    dump(out / 'metadata.json', metadata)
    shift = module.train_action_scheduler.shift
    nt = float(module.train_action_scheduler.num_train_timesteps)
    metadata['shift'] = shift
    started = time.time()
    jobs = manifest[args.offset: None if args.limit is None else args.offset+args.limit]
    with (out / 'measurements.jsonl').open('w') as log:
        for mi, item in enumerate(jobs):
            # Bypass dataset's random fallback: a failed sample must not silently change identity.
            sample = default_collate([dataset._get(item['index'])])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                inputs = module.build_action_distill_inputs(sample)
                if joint:
                    teacher_condition = denoisers['teacher'].build_condition(inputs)
                    target_condition = denoisers['target'].build_condition(inputs)
                else:
                    teacher_condition = target_condition = build_action_distill_condition(module, inputs)
            valid = None if inputs['action_is_pad'] is None else ~inputs['action_is_pad']
            a = inputs['action'].float()
            for seed in args.seeds:
                generator = torch.Generator(device=a.device).manual_seed(seed + item['episode']*1009 + item['frame'])
                eps = torch.randn(a.shape, generator=generator, device=a.device, dtype=torch.bfloat16).float()
                for requested_sigma in args.sigmas:
                    sigma = requested_sigma
                    if sigma == -1:
                        rng = np.random.default_rng(seed + item['episode']*1009 + item['frame'] + 89173)
                        u = float(rng.uniform())
                        sigma = shift*u/(1+(shift-1)*u)
                    base = sigma / (shift - (shift-1)*sigma)
                    s, e = shifted_consistency_pair(torch.tensor([base], device=a.device), shift, target_steps)
                    s, e = s.to(torch.bfloat16), e.to(torch.bfloat16)
                    sb, eb = s[:,None,None], e[:,None,None]
                    s3, e3 = sb.float(), eb.float()
                    x = (1-sb)*a.to(torch.bfloat16) + sb*eps.to(torch.bfloat16)
                    t = s*nt
                    rollout_base = float(s.item()) / (shift - (shift-1)*float(s.item()))
                    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        vt = denoisers['teacher'](x, t, teacher_condition)
                        # Match the source training target's bf16 arithmetic.
                        bridge = x + (eb-sb)*vt
                        ve = denoisers['target'](bridge, e*nt, target_condition)
                        ctarget = (bridge-eb*ve).float()
                        endpoints = {}
                        for steps in args.teacher_steps:
                            trajectory = x.float().clone()
                            grid = torch.linspace(rollout_base, 0., steps+1, device=a.device)
                            grid = shift*grid/(1+(shift-1)*grid)
                            for j in range(steps):
                                v = vt if j == 0 else denoisers['teacher'](trajectory.to(torch.bfloat16), (grid[j:j+1]*nt).to(torch.bfloat16), teacher_condition)
                                trajectory = trajectory + (grid[j+1]-grid[j])*v.float()
                            endpoints[steps] = trajectory.detach()
                    with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
                        condition = denoisers['student'].build_condition(inputs) if joint else teacher_condition
                        velocity = denoisers['student'](x, t, condition)
                        x0 = (x-sb*velocity).float()
                        losses = {'consistency': _masked_pseudo_huber(x0, ctarget, valid, huber_c),
                                  'gt_fm': _masked_mse(velocity, eps.to(torch.bfloat16)-a.to(torch.bfloat16), valid)}
                        for steps, endpoint in endpoints.items():
                            losses[f'teacher_{steps}'] = _masked_mse(velocity, (x.float()-endpoint)/s3, valid)
                    # Same forward graph and same inputs for every objective; no DDP or optimizer.
                    outgrads = {k: torch.autograd.grad(loss, velocity, retain_graph=True)[0].detach().float() for k, loss in losses.items()}
                    gc = torch.autograd.grad(losses['consistency'], params, retain_graph=True, allow_unused=True)
                    pair_stats = {}
                    gf = None
                    keys = list(losses)[1:]
                    for ki, key in enumerate(keys):
                        grad = torch.autograd.grad(losses[key], params, retain_graph=ki != len(keys)-1, allow_unused=True)
                        pair_stats['consistency__'+key] = gradient_stats(gc, grad, names)
                        if key == 'gt_fm':
                            gf = grad
                        else:
                            pair_stats['gt_fm__'+key] = gradient_stats(gf, grad, names)
                        del grad
                    output_cos = {a1+'__'+b1: vector_cos(outgrads[a1],outgrads[b1]) for a1,b1 in itertools.combinations(losses,2)}
                    row = {**item, 'seed': seed, 'sigma': float(s.item()), 'sigma_requested': requested_sigma,
                           'sigma_end': float(e.item()), 'base_sigma': base,
                           'losses': {k: float(v.detach()) for k,v in losses.items()},
                           'parameter_pairs': pair_stats, 'output_cos': output_cos,
                           'target_mse_gt': float(_masked_mse(ctarget,a,valid)),
                           'target_mse_teacher': {str(k): float(_masked_mse(ctarget,v,valid)) for k,v in endpoints.items()},
                           'teacher_mse_gt': {str(k): float(_masked_mse(v,a,valid)) for k,v in endpoints.items()},
                           'student_mse_teacher': {str(k): float(_masked_mse(x0.detach(),v,valid)) for k,v in endpoints.items()},
                           'student_mse_gt': float(_masked_mse(x0.detach(),a,valid)),
                           'teacher_solver_mse_vs_max': {str(k): float(_masked_mse(v,endpoints[max(endpoints)],valid)) for k,v in endpoints.items()},
                           'valid_action_steps': int(valid.sum()) if valid is not None else int(a.shape[1])}
                    if 1 in endpoints:
                        err = ((x.float()-endpoints[1])/s3-vt.float()).abs().max().item()
                        row['k1_velocity_identity_max_error'] = err
                        assert err < 5e-5, err
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    log.flush()
                    print('MEASURE', mi+1, '/', len(jobs), item['episode'], seed, sigma,
                          'cosC_F=', pair_stats['consistency__gt_fm']['all']['cos'],
                          'cosC_T=', pair_stats[f'consistency__teacher_{max(endpoints)}']['all']['cos'],
                          'seconds=',round(time.time()-started,1), flush=True)
                    del gc, gf, losses, outgrads, velocity, x0, condition
            del teacher_condition, target_condition, inputs
        torch.cuda.synchronize()
    after = [signature(p) for p in files]
    assert before == after, 'Source checkpoint changed during diagnostics'
    metadata.update({'elapsed_seconds': time.time()-started, 'max_memory_gib': torch.cuda.max_memory_allocated()/2**30,
                     'checkpoint_files_unchanged': True, 'completed': True})
    dump(out / 'metadata.json', metadata)
    print('DONE', out, flush=True)


def action_metrics(prediction, reference, scale, offset):
    p, r = prediction.float(), reference.float()
    error = p-r
    raw = error/scale
    joints = [0,1,2,3,4,5,7,8,9,10,11,12]
    grips = [6,13]
    n = p.shape[0]
    t = torch.arange(n,device=p.device,dtype=torch.float32)
    k = t[:,None]
    basis = torch.cos(torch.pi/n*(t[None,:]+.5)*k)*(2/n)**.5
    basis[0] /= 2**.5
    spectrum = (basis@error).square()
    raw_p,raw_r=(p-offset)/scale,(r-offset)/scale
    return {'normalized_mse':float(error.square().mean()),
            'per_dim_mse':error.square().mean(0).tolist(),
            'per_step_mse':error.square().mean(1).tolist(),
            'first8_mse':float(error[:8].square().mean()),
            'first24_mse':float(error[:24].square().mean()),
            'last8_mse':float(error[24:].square().mean()),
            'joint_mae_rad':float(raw[:,joints].abs().mean()),
            'gripper_mae':float(raw[:,grips].abs().mean()),
            'gripper_midpoint_disagreement':float(((raw_p[:,grips]>.5)!=(raw_r[:,grips]>.5)).float().mean()),
            'joint_first_difference_mse':float((raw[1:,joints]-raw[:-1,joints]).square().mean()),
            'gripper_first_difference_mse':float((raw[1:,grips]-raw[:-1,grips]).square().mean()),
            'dct_error_energy':{'dc':float(spectrum[0].sum()),'low_1_3':float(spectrum[1:4].sum()),
                                'mid_4_7':float(spectrum[4:8].sum()),'high_8_31':float(spectrum[8:].sum())}}


@torch.no_grad()
def components(args):
    import copy
    import re
    from dataclasses import replace
    sys.path.insert(0,args.source)
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.data.robotwin_dataset import _build_robotwin_dataset
    from lightx2v_train.model_zoo.native.wan.fastwam.action_distill import CachedActionDenoiser
    from lightx2v_train.trainers.fastwam_action_consistency.config import ActionStudentConfig
    from lightx2v_train.trainers.fastwam_joint_consistency.roles import configure_student, load_role_state_dict, JointConsistencyDenoiser
    from torch.utils.data import default_collate
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    out=Path(args.output).resolve()
    assert not out.is_relative_to(Path(args.source).resolve().parent)
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=Path(args.checkpoint)
    cfg=yaml.safe_load((checkpoint/'config.yaml').read_text())
    assert cfg['training']['train_video'] and cfg['training']['action_consistency']['flow_loss_weight']==0
    signatures=[signature(checkpoint/name) for name in ['config.yaml','ema_action.pt','ema_video.pt']]
    dataset=_build_robotwin_dataset(cfg['data']['val'],'val').dataset
    lr=dataset.lerobot_dataset
    order=sorted(range(len(lr.episodes)),key=lambda i:lr.episodes[i].index)
    if args.task_text:
        allowed=set()
        for root in {ep.root for ep in lr.episodes}:
            for line in (root/'meta/episodes.jsonl').read_text().splitlines():
                entry=json.loads(line)
                if any(args.task_text.lower() in text.lower() for text in entry['tasks']):allowed.add((root,entry['episode_index']))
        order=[i for i in order if (lr.episodes[i].root,lr.episodes[i].index) in allowed]
        assert order,'No matching validation episodes'
    positions=[order[i] for i in np.linspace(0,len(order)-1,min(args.episodes,len(order)),dtype=int)]
    manifest=[]
    for pos in positions:
        ep=lr.episodes[pos]
        start=0 if pos==0 else lr._episode_ends[pos-1]
        for phase in args.phases:
            frame=int(max(0,ep.length-dataset.num_frames)*phase)
            manifest.append({'episode':ep.index,'frame':frame,'phase':phase,'index':start+frame,'position':pos})
    dump(out/'manifest.json',manifest)
    print('COMPONENTS loading',checkpoint,flush=True)
    model=build_model(cfg); model.load_components()
    module=model.unwrap_module();module.eval().requires_grad_(False)
    original_action=copy.deepcopy(module.action_expert).eval().requires_grad_(False)
    original_video=copy.deepcopy(module.video_expert).eval().requires_grad_(False)
    scfg=ActionStudentConfig.from_mapping(cfg['training']['student'])
    action=configure_student(module.action_expert,scfg)
    video=configure_student(module.video_expert,scfg)
    for expert,kind in [(action,'action'),(video,'video')]:
        load_role_state_dict(expert,scfg.train_type,torch.load(checkpoint/f'ema_{kind}.pt',map_location='cpu',weights_only=True))
        expert.eval().requires_grad_(False)
    teacher=JointConsistencyDenoiser(original_action,original_video,module).eval()
    ema=JointConsistencyDenoiser(action,video,module).eval()
    merged=None
    if args.check_merged:
        merged=JointConsistencyDenoiser(copy.deepcopy(original_action),copy.deepcopy(original_video),module).eval().requires_grad_(False)
        saved=torch.load(args.check_merged,map_location='cpu',weights_only=True,mmap=True)
        merged.mot.load_state_dict(saved['mot'],strict=True)
        assert saved['step']==30000
        for k,v in module.proprio_encoder.state_dict().items():
            assert torch.equal(v.cpu(),saved['proprio_encoder'][k])
        del saved
    comparison=None
    if args.comparison_checkpoint:
        comparison=JointConsistencyDenoiser(copy.deepcopy(original_action),copy.deepcopy(original_video),module).eval().requires_grad_(False)
        saved=torch.load(args.comparison_checkpoint,map_location='cpu',weights_only=True,mmap=True)
        comparison.mot.load_state_dict(saved['mot'],strict=True)
        for k,v in module.proprio_encoder.state_dict().items():assert torch.equal(v.cpu(),saved['proprio_encoder'][k])
        del saved
    norm=dataset.processor.normalizer.normalizers['action']['default']
    scale=norm.scale.to(module.device);offset=norm.offset.to(module.device)
    adapter_layers={kind:[(n,m) for n,m in expert.named_modules() if hasattr(m,'lora_A') and 'default' in m.lora_A]
                    for kind,expert in [('action',action),('video',video)]}
    def adapter_gate(kind,group=None):
        for name,m in adapter_layers[kind]:
            block=int(re.search(r'blocks\.(\d+)',name).group(1))
            selected=(group=='all' or group=='early' and block<10 or group=='middle' and 10<=block<20 or
                      group=='late' and block>=20 or group in ['self_attn','cross_attn','ffn'] and group in name)
            m.scaling['default']=0. if selected else float(scfg.lora['alpha'])/int(scfg.lora['rank'])
    if args.offset==0:
        weights=[]
        for kind,layers in adapter_layers.items():
            for name,m in layers:
                a=m.lora_A['default'].weight.float();b=m.lora_B['default'].weight.float()
                delta=b@a*m.scaling['default'];base=m.base_layer.weight.float()
                weights.append({'branch':kind,'name':name,'block':int(re.search(r'blocks\.(\d+)',name).group(1)),
                                'module':'self_attn' if 'self_attn' in name else 'cross_attn' if 'cross_attn' in name else 'ffn',
                                'base_norm':float(base.norm()),'update_norm':float(delta.norm()),
                                'relative_update':float(delta.norm()/base.norm()),'base_update_cos':vector_cos(base,delta)})
        dump(out/'weight_updates.json',weights)
    def expand_condition(c,batch):
        def ex(t):return t.expand(batch,*t.shape[1:])
        return replace(c,context=ex(c.context),context_mask=ex(c.context_mask),
                       video_kv_cache=[{k:ex(v) for k,v in item.items()} for item in c.video_kv_cache])
    def sample(denoiser,condition,noise,steps,save_states=False):
        times,deltas=module.infer_action_scheduler.build_inference_schedule(steps,noise.device,noise.dtype)
        x=noise.clone();states=[]
        for i,(t,d) in enumerate(zip(times,deltas)):
            if save_states:states.append((float(t)/1000,x.clone()))
            x=module.infer_action_scheduler.step(denoiser(x,t.expand(x.shape[0]),condition),d,x)
        return (x,states) if save_states else x
    def sample_precise_head(denoiser,condition,noise,steps):
        def fp32_head(layer,inputs,output):
            tf32=torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32=False
                with torch.autocast('cuda',enabled=False):
                    return torch.nn.functional.linear(inputs[0].float(),layer.weight.float(),None if layer.bias is None else layer.bias.float())
            finally:
                torch.backends.cuda.matmul.allow_tf32=tf32
        handle=denoiser.action_module().head.register_forward_hook(fp32_head)
        try:return sample(denoiser,condition,noise,steps)
        finally:handle.remove()
    metadata={'args':vars(args),'weight_role':'EMA','config':cfg,'checkpoint_signatures':signatures,
              'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'completed':False,
              'sampling':'original scheduler, bf16 accumulation, common initial noise',
              'interpretation':'Branch swaps and adapter ablations are offline interventions, not closed-loop success tests.'}
    dump(out/'metadata.json',metadata)
    jobs=manifest[args.offset*len(args.phases):None if args.limit is None else (args.offset+args.limit)*len(args.phases)]
    started=time.time()
    with (out/'components.jsonl').open('w') as handle:
        for wi,item in enumerate(jobs):
            sample_data=default_collate([dataset._get(item['index'])])
            with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
                inputs=module.build_action_distill_inputs(sample_data)
                assert inputs['action_is_pad'] is None or not bool(inputs['action_is_pad'].any())
                ct=teacher.build_condition(inputs);ce=ema.build_condition(inputs)
                drift=[]
                for j,(a,b) in enumerate(zip(ct.video_kv_cache,ce.video_kv_cache)):
                    drift.append({'block':j,**{k:{'cos':vector_cos(a[k],b[k]),'relative_l2':float((a[k].float()-b[k].float()).norm()/a[k].float().norm())} for k in a}})
                batch=len(args.seeds)
                ct,ce=expand_condition(ct,batch),expand_condition(ce,batch)
                if args.noise_mode=='fixed_cpu':
                    noise=torch.cat([torch.randn((1,*inputs['action'].shape[1:]),generator=torch.Generator(device='cpu').manual_seed(seed),device='cpu',dtype=torch.float32).to(device=module.device,dtype=torch.bfloat16) for seed in args.seeds])
                else:
                    noise=torch.cat([torch.randn((1,*inputs['action'].shape[1:]),generator=torch.Generator(device=module.device).manual_seed(seed+item['episode']*1009+item['frame']),device=module.device,dtype=torch.bfloat16) for seed in args.seeds])
                pred={}
                pred['original_10'],states=sample(teacher,ct,noise,10,True)
                pred['original_20']=sample(teacher,ct,noise,20)
                pred['original_1']=sample(teacher,ct,noise,1)
                pred['noflow_1']=sample(ema,ce,noise,1)
                if comparison is not None:
                    cc=expand_condition(comparison.build_condition(inputs),batch)
                    pred[args.comparison_name]=sample(comparison,cc,noise,1)
                    del cc
                if merged is not None:
                    cm=expand_condition(merged.build_condition(inputs),batch)
                    pred['deployed_merged_1']=sample(merged,cm,noise,1)
                    pred['deployed_original_video_1']=sample(merged,ct,noise,1)
                    for group,low,high in [('early',0,10),('middle',10,20),('late',20,30)]:
                        backups=[]
                        for name,layer in adapter_layers['video']:
                            block=int(re.search(r'blocks\.(\d+)',name).group(1))
                            if low<=block<high:
                                plain=name.removeprefix('base_model.model.')
                                weight=merged.video_expert.get_submodule(plain).weight
                                backups.append((weight,weight.clone()))
                                weight.copy_(original_video.get_submodule(plain).weight)
                        cg=expand_condition(merged.build_condition(inputs),batch)
                        pred['deployed_restore_video_'+group]=sample(merged,cg,noise,1)
                        for weight,backup in backups:weight.copy_(backup)
                        del cg,backups
                    pred['deployed_fp32_subtract_1']=noise.float()-merged(noise,torch.full((batch,),1000,device=noise.device,dtype=noise.dtype),cm).float()
                    pred['deployed_fp32_head_1']=sample_precise_head(merged,cm,noise,1)
                    pred['noflow_fp32_head_1']=sample_precise_head(ema,ce,noise,1)
                    pred['original_fp32_head_10']=sample_precise_head(teacher,ct,noise,10)
                    pred['oracle_endpoint_bf16_roundtrip']=noise-(noise.float()-pred['original_10'].float()).to(noise.dtype)
                    del cm
                pred['original_video_noflow_action_1']=sample(ema,ct,noise,1)
                pred['noflow_video_original_action_10']=sample(teacher,ce,noise,10)
                # This probes reuse as the original Euler velocity field, not an approved CM sampler.
                pred['noflow_as_euler_10']=sample(ema,ce,noise,10)
                for group in ['early','middle','late','self_attn','cross_attn','ffn']:
                    adapter_gate('action',group)
                    pred['restore_action_'+group]=sample(ema,ce,noise,1)
                adapter_gate('action')
                for group in ['early','middle','late']:
                    adapter_gate('video',group)
                    cv=expand_condition(ema.build_condition(inputs),batch)
                    pred['restore_video_'+group]=sample(ema,cv,noise,1)
                adapter_gate('video')
                probes={}
                for index in [0,3,6,9]:
                    sigma,x=states[index]
                    t=torch.full((batch,),sigma*1000,device=x.device,dtype=x.dtype)
                    v=ema(x,t,ce)
                    vt=teacher(x,t,ct)
                    endpoint=x-sigma*v
                    probes[str(sigma)]={'endpoint':endpoint.float().cpu(),'original_endpoint':(x-sigma*vt).float().cpu(),
                                        'velocity_cos':[vector_cos(v[i],vt[i]) for i in range(batch)]}
            for i,seed in enumerate(args.seeds):
                ref=pred['original_10'][i]
                metrics={name:action_metrics(value[i],ref,scale,offset) for name,value in pred.items()}
                gt_metrics={name:action_metrics(value[i],inputs['action'][0],scale,offset) for name,value in pred.items()}
                probe_rows={k:{'endpoint_mse':float((v['endpoint'][i]-ref.float().cpu()).square().mean()),
                               'original_tangent_endpoint_mse':float((v['original_endpoint'][i]-ref.float().cpu()).square().mean()),
                               'self_endpoint_drift_mse':float((v['endpoint'][i]-pred['noflow_1'][i].float().cpu()).square().mean()),
                               'velocity_cos':v['velocity_cos'][i]} for k,v in probes.items()}
                row={**item,'seed':seed,'prompt':sample_data['prompt'][0],'metrics_vs_original10':metrics,'metrics_vs_gt':gt_metrics,'video_cache_drift':drift,'teacher_trajectory_probes':probe_rows}
                handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush()
            # Small action tensors permit independent later verification and decomposition.
            torch.save({'item':item,'seeds':args.seeds,'gt':inputs['action'].float().cpu(),
                        'observed_state_normalized':sample_data['proprio'].float().cpu(),
                        'action_scale':scale.cpu(),'action_offset':offset.cpu(),
                        'state_scale':dataset.processor.normalizer.normalizers['state']['default'].scale.cpu(),
                        'state_offset':dataset.processor.normalizer.normalizers['state']['default'].offset.cpu(),
                        'predictions':{k:v.float().cpu() for k,v in pred.items()}},out/f"actions_ep{item['episode']}_f{item['frame']}.pt")
            print('COMPONENT',wi+1,'/',len(jobs),item['episode'],item['phase'],round(time.time()-started,1),flush=True)
            del inputs,ct,ce,cv,pred,states,probes
    assert signatures==[signature(checkpoint/name) for name in ['config.yaml','ema_action.pt','ema_video.pt']]
    metadata.update(completed=True,elapsed_seconds=time.time()-started,max_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    dump(out/'metadata.json',metadata)
    print('DONE COMPONENTS',out,flush=True)


def closedloop(args):
    import re
    root=Path('/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM/evaluate_results/robotwin')
    folders={'original':root/'robotwin_uncond_3cam_384/robotwin_original_fastwam_10step_20260905_041955',
             'noflow':root/'robotwin_consistency_joint_ts10_noflow_ema_step30000/robotwin_consistency_joint_ts10_noflow_ema_step30000_1step_20260907_130853'}
    summaries={k:json.loads((p/'summary.json').read_text()) for k,p in folders.items()}
    by={k:{t['task_name']:t for t in v['per_task']} for k,v in summaries.items()}
    tasks=[]
    for name in by['original']:
        a,b=by['original'][name],by['noflow'][name]
        c=b['clean_success_rate']-a['clean_success_rate'];r=b['random_success_rate']-a['random_success_rate']
        tasks.append({'task':name,'original':a,'noflow':b,'clean_delta':c,'random_delta':r,'overall_delta':(c+r)/2})
    outcomes={}
    ansi=re.compile(r'\x1b\[[0-9;]*m')
    for label,folder in folders.items():
        results={}
        for task in by[label]:
            for p in sorted(folder.glob('eval_'+task+'_*.log')):
                if not re.fullmatch('eval_'+re.escape(task)+r'_\d{8}_\d{6}\.log',p.name):continue
                phase=None;previous=0
                for line in ansi.sub('',p.read_text(errors='replace')).splitlines():
                    if '| fastwam_policy | demo_' in line:phase='random' if 'demo_randomized' in line else 'clean'
                    m=re.search(r'Success rate: (\d+)/(\d+).*current seed: (\d+)',line)
                    if m and phase:
                        successes,trial,seed=map(int,m.groups())
                        if trial==1:previous=0
                        success=successes-previous;assert success in [0,1]
                        results[(task,phase,seed)]={'success':success,'episode':trial-1,'log':str(p)}
                        previous=successes
        assert len(results)==10000,(label,len(results))
        outcomes[label]=results
    common=outcomes['original'].keys()&outcomes['noflow'].keys()
    paired=[]
    for key in sorted(common):
        a,b=outcomes['original'][key],outcomes['noflow'][key]
        task,phase,seed=key
        row={'task':task,'phase':phase,'seed':seed,'original':a,'noflow':b}
        for label,record in [('original',a),('noflow',b)]:
            video=folders[label]/task/f"episode{record['episode']}_randomized-{'true' if phase=='random' else 'false'}_success-{'true' if record['success'] else 'false'}.mp4"
            row[label]['video']=str(video)
        paired.append(row)
    rng=np.random.default_rng(8193);diff=np.array([x['overall_delta'] for x in tasks])
    interval=np.quantile(diff[rng.integers(0,len(diff),size=(20000,len(diff)))].mean(1),[.025,.975]).tolist()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'closedloop.json',{'sources':{k:str(p) for k,p in folders.items()},'overall':{k:v['overall'] for k,v in summaries.items()},
                               'tasks':sorted(tasks,key=lambda x:x['overall_delta']),'task_bootstrap_delta_ci95':interval,
                               'matched':len(paired),'lost':sum(r['original']['success'] and not r['noflow']['success'] for r in paired),
                               'gained':sum(r['noflow']['success'] and not r['original']['success'] for r in paired)})
    dump(out/'paired_episodes.json',paired)
    print('CLOSEDLOOP',len(paired),interval)


@torch.no_grad()
def microwave_response(args):
    import copy
    from torch.utils.data import default_collate
    sys.path.insert(0,args.source)
    from lightx2v_train.model_zoo import build_model
    from lightx2v_train.data.robotwin_dataset import _build_robotwin_dataset
    from lightx2v_train.trainers.fastwam_joint_consistency.roles import JointConsistencyDenoiser
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=True
    out=Path(args.output).resolve()
    assert not out.is_relative_to(Path(args.source).resolve().parent)
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'metadata.json').exists(),'Use a fresh response output directory'
    cfg=yaml.safe_load((Path(args.checkpoint)/'config.yaml').read_text())
    dataset=_build_robotwin_dataset(cfg['data']['val'],'val').dataset
    assert dataset.lerobot_dataset.global_sample_stride==1
    manifest=json.loads(Path(args.response_manifest).read_text())
    manifest=manifest[args.offset:None if args.limit is None else args.offset+args.limit]
    directions=torch.randn(3,6,generator=torch.Generator().manual_seed(9183),dtype=torch.float32)
    directions=directions/directions.norm(dim=1,keepdim=True)
    perturbations=[(amplitude,direction,sign) for amplitude in [.01,.05] for direction in range(3) for sign in [-1,1]]
    paths={'original_10':cfg['model']['checkpoint_path'],'noflow':args.check_merged,args.comparison_name:args.comparison_checkpoint}
    assert len(paths)==3 and all(paths.values())
    signatures=[signature(Path(path)) for path in paths.values()]
    metadata={'args':vars(args),'config':cfg,'checkpoint_signatures':signatures,'manifest':manifest,
              'models':paths,'directions':directions.tolist(),'amplitudes_rad':[.01,.05],
              'noise':'CPU FP32 seed42 -> BF16, fixed across conditions','completed':False,
              'interpretation':'Demonstration transitions and proprio-only perturbations, not simulator interventions.'}
    dump(out/'metadata.json',metadata)
    print('RESPONSE loading',args.comparison_name,flush=True)
    model=build_model(cfg);model.load_components()
    module=model.unwrap_module().eval().requires_grad_(False)
    denoisers={'original_10':JointConsistencyDenoiser(module.action_expert,module.video_expert,module).eval()}
    for name in ['noflow',args.comparison_name]:
        role=JointConsistencyDenoiser(copy.deepcopy(module.action_expert),copy.deepcopy(module.video_expert),module).eval().requires_grad_(False)
        weights=torch.load(paths[name],map_location='cpu',mmap=True,weights_only=True)
        role.mot.load_state_dict(weights['mot'],strict=True)
        for key,value in module.proprio_encoder.state_dict().items():assert torch.equal(value.cpu(),weights['proprio_encoder'][key])
        denoisers[name]=role
        del weights
    anorm=dataset.processor.normalizer.normalizers['action']['default']
    snorm=dataset.processor.normalizer.normalizers['state']['default']
    scale=anorm.scale.cpu();offset=anorm.offset.cpu()
    def predict(inputs):
        batch=inputs['action'].shape[0]
        noise=torch.randn((1,32,14),generator=torch.Generator(device='cpu').manual_seed(42),dtype=torch.float32).to(module.device,torch.bfloat16).expand(batch,-1,-1)
        predictions={}
        for name,denoiser in denoisers.items():
            condition=denoiser.build_condition(inputs)
            times,deltas=module.infer_action_scheduler.build_inference_schedule(10 if name=='original_10' else 1,noise.device,noise.dtype)
            x=noise.clone()
            for t,d in zip(times,deltas):x=module.infer_action_scheduler.step(denoiser(x,t.expand(batch),condition),d,x)
            predictions[name]=x.float().cpu()
            del condition
        return predictions
    started=time.time()
    for wi,item in enumerate(manifest):
        result={'item':item,'action_scale':scale,'action_offset':offset,'directions':directions,'real':{},'perturbed':[]}
        with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for lag in [0,8,24]:
                ep,frame=dataset.lerobot_dataset._locate(item['index']+lag)
                assert ep==item['position'] and frame==item['frame']+lag
                sample=default_collate([dataset._get(item['index']+lag)])
                assert not sample['action_is_pad'].any()
                inputs=module.build_action_distill_inputs(sample)
                result['real'][str(lag)]={'gt':inputs['action'].float().cpu(),'predictions':predict(inputs),
                                         'proprio':sample['proprio'][:,0].clone(),'prompt':sample['prompt']}
                if lag==0:base_inputs=inputs;base_sample=sample
            for lag in [8,24]:
                assert torch.equal(result['real']['0']['gt'][:,lag:],result['real'][str(lag)]['gt'][:,:32-lag])
            for begin in range(0,len(perturbations),4):
                group=perturbations[begin:begin+4];batch=len(group)
                q=base_sample['proprio'][:,0].float().repeat(batch,1)
                for i,(amplitude,direction,sign) in enumerate(group):q[i,:6]+=amplitude*sign*directions[direction]*snorm.scale[:6].cpu()
                inputs={k:v.expand(batch,*v.shape[1:]) if torch.is_tensor(v) else v for k,v in base_inputs.items()}
                inputs['context'],inputs['context_mask']=module._append_proprio_to_context(
                    base_sample['context'].to(module.device,torch.bfloat16).expand(batch,-1,-1),
                    base_sample['context_mask'].to(module.device,torch.bool).expand(batch,-1),q.to(module.device,torch.bfloat16))
                predictions=predict(inputs)
                for i,(amplitude,direction,sign) in enumerate(group):
                    effective=(q[i].to(torch.bfloat16).float()-base_sample['proprio'][0,0].to(torch.bfloat16).float())/snorm.scale.cpu()
                    result['perturbed'].append({'amplitude':amplitude,'direction':direction,'sign':sign,
                                               'effective_delta_rad':effective,'predictions':{name:p[i] for name,p in predictions.items()}})
            # A matching batch-size control bounds rounding differences in perturbation comparisons.
            inputs={k:v.expand(4,*v.shape[1:]) if torch.is_tensor(v) else v for k,v in base_inputs.items()}
            result['batch4_control']=predict(inputs)
        torch.save(result,out/f'response_{wi:03d}.pt')
        print('RESPONSE',args.comparison_name,wi+1,'/',len(manifest),round(time.time()-started,1),flush=True)
    for info in signatures:assert signature(Path(info['path']))==info
    metadata['completed']=True;metadata['elapsed_seconds']=time.time()-started
    dump(out/'metadata.json',metadata)
    print('DONE RESPONSE',args.comparison_name,flush=True)


def response_summary(args):
    combined={};provenance=[]
    for directory in args.response_summary:
        root=Path(directory);meta=json.loads((root/'metadata.json').read_text());assert meta['completed']
        provenance.append(meta)
        for file in sorted(root.glob('response_*.pt')):
            data=torch.load(file,map_location='cpu',weights_only=True)
            key=(data['item']['episode'],data['item']['frame'])
            if key not in combined:combined[key]=data;continue
            old=combined[key]
            assert old['item']==data['item'] and torch.equal(old['directions'],data['directions'])
            assert torch.equal(old['action_scale'],data['action_scale']) and torch.equal(old['action_offset'],data['action_offset'])
            for lag in ['0','8','24']:
                assert old['real'][lag]['prompt']==data['real'][lag]['prompt']
                for field in ['gt','proprio']:assert torch.equal(old['real'][lag][field],data['real'][lag][field])
            assert len(old['perturbed'])==len(data['perturbed'])
            for a,b in zip(old['perturbed'],data['perturbed']):
                assert all(a[k]==b[k] for k in ['amplitude','direction','sign'])
                assert torch.equal(a['effective_delta_rad'],b['effective_delta_rad'])
            pairs=[(old['real'][lag]['predictions'],data['real'][lag]['predictions']) for lag in ['0','8','24']]
            pairs.extend((a['predictions'],b['predictions']) for a,b in zip(old['perturbed'],data['perturbed']))
            pairs.append((old['batch4_control'],data['batch4_control']))
            for dest,source in pairs:
                for name,value in source.items():
                    if name in dest:assert torch.equal(dest[name],value),(key,name)
                    else:dest[name]=value
    names=list(next(iter(combined.values()))['batch4_control'])
    transition=[];response=[];baseline=[]
    for data in combined.values():
        scale=data['action_scale'].numpy().astype(np.float64)
        def raw(pred):return pred.numpy().astype(np.float64)/scale
        base={name:raw(data['real']['0']['predictions'][name][0]) for name in names}
        for name in names:
            baseline.append({**data['item'],'model':name,
                             'mse':float(np.mean((base[name][:24,:6]-base['original_10'][:24,:6])**2)),
                             'batch4_rounding_mse':float(np.mean((raw(data['batch4_control'][name][0])[:24,:6]-base[name][:24,:6])**2))})
        for lag in [8,24]:
            nxt={name:raw(data['real'][str(lag)]['predictions'][name][0]) for name in names}
            teacher_revision=nxt['original_10'][:32-lag,:6]-base['original_10'][lag:,:6]
            teacher_jump=nxt['original_10'][0,:6]-base['original_10'][lag-1,:6]
            for name in names:
                revision=nxt[name][:32-lag,:6]-base[name][lag:,:6]
                jump=nxt[name][0,:6]-base[name][lag-1,:6]
                transition.append({**data['item'],'lag':lag,'model':name,
                                   'overlap_revision_mse':float(np.mean(revision**2)),
                                   'revision_error_mse':float(np.mean((revision-teacher_revision)**2)),
                                   'boundary_jump_mse':float(np.mean(jump**2)),
                                   'boundary_error_mse':float(np.mean((jump-teacher_jump)**2))})
        probes={(p['amplitude'],p['direction'],p['sign']):p for p in data['perturbed']}
        for amplitude in [.01,.05]:
            for direction in range(3):
                plus=probes[(amplitude,direction,1)];minus=probes[(amplitude,direction,-1)]
                teacher_change=(raw(plus['predictions']['original_10'])-raw(minus['predictions']['original_10']))[:24,:6]
                norm=float(np.linalg.norm(teacher_change))
                for name in names:
                    change=(raw(plus['predictions'][name])-raw(minus['predictions'][name]))[:24,:6]
                    cnorm=float(np.linalg.norm(change))
                    cosine=float(np.clip(np.sum(change*teacher_change)/(cnorm*norm),-1,1)) if min(cnorm,norm)>1e-8 else None
                    response.append({**data['item'],'model':name,'amplitude':amplitude,'direction':direction,
                                     'response_error_mse':float(np.mean((change-teacher_change)**2)),
                                     'response_mse':float(np.mean(change**2)),
                                     'teacher_response_mse':float(np.mean(teacher_change**2)),
                                     'response_cosine':cosine,
                                     'response_angle_deg':float(np.degrees(np.arccos(cosine))) if cosine is not None else None,
                                     'response_norm_ratio':cnorm/norm if norm>1e-8 else None})
    summary={'windows':len(combined),'episodes':len({r['episode'] for r in baseline}),'provenance':args.response_summary,
             'baseline':{},'transitions':{},'perturbations':{}}
    for name in names:
        subset=[r for r in baseline if r['model']==name]
        summary['baseline'][name]={k:cluster_interval(subset,lambda r,k=k:r[k]) for k in ['mse','batch4_rounding_mse']}
    for lag in [8,24]:
        summary['transitions'][str(lag)]={}
        for name in names:
            subset=[r for r in transition if r['model']==name and r['lag']==lag]
            reference={(r['episode'],r['frame']):r for r in transition if r['model']=='noflow' and r['lag']==lag}
            result={k:cluster_interval(subset,lambda r,k=k:r[k]) for k in ['overlap_revision_mse','revision_error_mse','boundary_jump_mse','boundary_error_mse']}
            result['revision_error_ratio_vs_noflow']=cluster_ratio(subset,lambda r:r['revision_error_mse'],lambda r:reference[(r['episode'],r['frame'])]['revision_error_mse'])
            summary['transitions'][str(lag)][name]=result
    for amplitude in [.01,.05]:
        summary['perturbations'][str(amplitude)]={}
        for name in names:
            subset=[r for r in response if r['model']==name and r['amplitude']==amplitude]
            reference={(r['episode'],r['frame'],r['direction']):r for r in response if r['model']=='noflow' and r['amplitude']==amplitude}
            result={k:cluster_interval(subset,lambda r,k=k:r[k]) for k in ['response_error_mse','response_mse','teacher_response_mse','response_cosine','response_angle_deg','response_norm_ratio']}
            result['response_error_ratio_vs_noflow']=cluster_ratio(subset,lambda r:r['response_error_mse'],lambda r:reference[(r['episode'],r['frame'],r['direction'])]['response_error_mse'])
            result['angle_delta_vs_noflow']=cluster_interval(subset,lambda r:r['response_angle_deg']-reference[(r['episode'],r['frame'],r['direction'])]['response_angle_deg'] if r['response_angle_deg'] is not None and reference[(r['episode'],r['frame'],r['direction'])]['response_angle_deg'] is not None else None)
            summary['perturbations'][str(amplitude)][name]=result
    summary['phase_breakdown']={}
    for phase in sorted({r['phase'] for r in baseline}):
        summary['phase_breakdown'][str(phase)]={name:{
            'revision24_mse':cluster_interval([r for r in transition if r['model']==name and r['lag']==24 and r['phase']==phase],lambda r:r['revision_error_mse']),
            'response_005_mse':cluster_interval([r for r in response if r['model']==name and r['amplitude']==.05 and r['phase']==phase],lambda r:r['response_error_mse']),
            'response_005_angle':cluster_interval([r for r in response if r['model']==name and r['amplitude']==.05 and r['phase']==phase],lambda r:r['response_angle_deg'])}
            for name in names}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    dump(out/'response_summary.json',summary)
    dump(out/'transition_rows.json',transition);dump(out/'perturbation_rows.json',response);dump(out/'baseline_rows.json',baseline)
    if args.plot_deps:
        sys.path.insert(0,args.plot_deps)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels={'noflow':'Consistency noflow','cm_flow':'Consistency + flow','dmd_joint':'DMD joint','dmd_action':'DMD action'}
        chosen=[name for name in labels if name in names]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for ax,group,key in [(axes[0],summary['transitions']['24'],'revision_error_ratio_vs_noflow'),
                             (axes[1],summary['perturbations']['0.05'],'response_error_ratio_vs_noflow')]:
            values=[group[name][key]['ratio_of_means'] for name in chosen]
            intervals=[group[name][key]['ci95'] for name in chosen]
            ax.bar(range(len(chosen)),values,color=['#16856b','#777777','#c75445','#3b78ad'][:len(chosen)])
            ax.errorbar(range(len(chosen)),values,yerr=np.array([[v-lo,hi-v] for v,(lo,hi) in zip(values,intervals)]).T,fmt='none',color='black',capsize=4)
            ax.set_xticks(range(len(chosen)),[labels[n] for n in chosen],rotation=20,ha='right');ax.axhline(1,color='black',linestyle=':')
            ax.set_ylabel('Error ratio vs Consistency noflow')
        axes[0].set_title('24-step revision error vs original 10-step')
        axes[1].set_title('Proprio response error, perturbation norm 0.05 rad')
        fig.tight_layout();fig.savefig(out/'response_comparison.png',dpi=180);plt.close(fig)
    print('RESPONSE SUMMARY',summary['windows'],summary['episodes'],flush=True)


def microwave_summary(args):
    windows={};provenance=[]
    for directory in args.microwave_summary:
        p=Path(directory);meta=json.loads((p/'metadata.json').read_text());assert meta['completed']
        if provenance:
            assert meta['args'].get('noise_mode','sampled_gpu')==provenance[0]['args'].get('noise_mode','sampled_gpu')
        provenance.append(meta)
        for file in sorted(p.glob('actions_*.pt')):
            data=torch.load(file,map_location='cpu',weights_only=True);key=(data['item']['episode'],data['item']['frame'])
            if key in windows:
                previous=windows[key]
                assert previous['seeds']==data['seeds'] and torch.equal(previous['gt'],data['gt'])
                for name,value in data['predictions'].items():
                    if name in previous['predictions']:assert torch.equal(value,previous['predictions'][name]),name
                    else:previous['predictions'][name]=value
            else:windows[key]=data
    names=['original_10','original_1','deployed_merged_1','dmd_action','dmd_joint','cm_flow','deployed_fp32_head_1']
    rows=[]
    for data in windows.values():
        assert all(n in data['predictions'] for n in names)
        scale=data['action_scale'].numpy().astype(np.float64);offset=data['action_offset'].numpy().astype(np.float64)
        q=(data['observed_state_normalized'].numpy().astype(np.float64).reshape(-1,14)[0]-data['state_offset'].numpy())/data['state_scale'].numpy()
        gt=(data['gt'][0].numpy().astype(np.float64)-offset)/scale
        for i,seed in enumerate(data['seeds']):
            ref=(data['predictions']['original_10'][i].numpy().astype(np.float64)-offset)/scale
            direction=ref[20:24,:6].mean(0)-q[:6];denom=float(np.sum(direction**2))
            row={**data['item'],'seed':seed,'teacher_move_norm':denom**.5,'models':{}}
            for name in names:
                pred=(data['predictions'][name][i].numpy().astype(np.float64)-offset)/scale
                error=pred-ref;move=pred[20:24,:6].mean(0)-q[:6]
                shifts=list(range(-4,5))
                shift_errors=[float(np.mean((pred[4:20,:6]-ref[4+shift:20+shift,:6])**2)) for shift in shifts]
                best=int(np.argmin(shift_errors))
                metrics={'left24_mse_rad2':float(np.mean(error[:24,:6]**2)),
                         'left24_mae_rad':float(np.mean(np.abs(error[:24,:6]))),
                         'left24_gt_mse_rad2':float(np.mean((pred[:24,:6]-gt[:24,:6])**2)),
                         'right24_mse_rad2':float(np.mean(error[:24,7:13]**2)),
                         'left_gripper24_mae':float(np.mean(np.abs(error[:24,6]))),
                         'left_first_difference_mse':float(np.mean(np.diff(error[:24,:6],axis=0)**2)),
                         'move_projection_ratio':float(move@direction/denom) if denom>.05**2 else None,
                         'move_cosine':float(np.clip(move@direction/(np.linalg.norm(move)*denom**.5),-1,1)) if denom>.05**2 and np.linalg.norm(move)>1e-8 else None,
                         'central_unaligned_mse':shift_errors[4],'oracle_shifted_mse':shift_errors[best],'best_shift_steps':shifts[best]}
                row['models'][name]=metrics
            rows.append(row)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    keys=list(rows[0]['models'][names[0]])
    summary={'episodes':len({r['episode'] for r in rows}),'windows':len(windows),'samples':len(rows),
             'input_equality_verified_across_checkpoints':True,'provenance':args.microwave_summary,
             'models':{name:{key:cluster_interval(rows,lambda r,n=name,k=key:r['models'][n][k]) for key in keys} for name in names},
             'phases':{str(phase):{name:{key:cluster_interval([r for r in rows if r['phase']==phase],lambda r,n=name,k=key:r['models'][n][k])
                                        for key in ['left24_mse_rad2','left24_gt_mse_rad2','left_gripper24_mae']} for name in names}
                        for phase in sorted({r['phase'] for r in rows})},
             'paired_vs_noflow':{name:{'left24_mse_ratio':cluster_ratio(rows,lambda r,n=name:r['models'][n]['left24_mse_rad2'],lambda r:r['models']['deployed_merged_1']['left24_mse_rad2']),
                                      'left24_gt_mse_delta':cluster_interval(rows,lambda r,n=name:r['models'][n]['left24_gt_mse_rad2']-r['models']['deployed_merged_1']['left24_gt_mse_rad2'])}
                                  for name in names}}
    summary['noise_mode']=provenance[0]['args'].get('noise_mode','sampled_gpu')
    summary['by_seed']={str(seed):{
        'samples':sum(r['seed']==seed for r in rows),
        'models':{name:{key:cluster_interval([r for r in rows if r['seed']==seed],lambda r,n=name,k=key:r['models'][n][k])
                        for key in ['left24_mse_rad2','left24_mae_rad','left24_gt_mse_rad2','right24_mse_rad2','left_gripper24_mae']}
                  for name in names},
        'mse_ratio_vs_noflow':{name:cluster_ratio([r for r in rows if r['seed']==seed],
                                                lambda r,n=name:r['models'][n]['left24_mse_rad2'],
                                                lambda r:r['models']['deployed_merged_1']['left24_mse_rad2']) for name in names}}
        for seed in sorted({r['seed'] for r in rows})}
    summary['progress_alignment']={}
    for label,subset in [('all',rows),('initial',[r for r in rows if r['phase']==0])]:
        summary['progress_alignment'][label]={name:{
            'samples':len(subset),
            'best_shift_counts':{str(shift):sum(r['models'][name]['best_shift_steps']==shift for r in subset) for shift in range(-4,5)},
            'remaining_error_ratio':cluster_ratio(subset,lambda r,n=name:r['models'][n]['oracle_shifted_mse'],
                                                   lambda r,n=name:r['models'][n]['central_unaligned_mse'])}
            for name in names if name!='original_10'}
    dump(out/'focused_summary.json',summary);dump(out/'focused_rows.json',rows)
    if args.plot_deps:
        sys.path.insert(0,args.plot_deps)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        labels={'original_10':'Original 10','deployed_merged_1':'Consistency noflow','dmd_action':'DMD action','dmd_joint':'DMD joint','cm_flow':'Consistency + flow'}
        fig,ax=plt.subplots(figsize=(8,4))
        for name in ['deployed_merged_1','dmd_action','dmd_joint','cm_flow']:
            ax.plot([float(p) for p in summary['phases']],[d[name]['left24_mse_rad2']['mean'] for d in summary['phases'].values()],marker='o',label=labels[name])
        ax.set_xlabel('Relative position in demonstration (not contact phase)');ax.set_ylabel('Left-arm MSE vs original 10-step (rad squared)')
        ax.set_yscale('log');ax.legend();fig.tight_layout();fig.savefig(out/'left_arm_by_phase.png',dpi=180);plt.close(fig)
        example=max(rows,key=lambda r:r['models']['dmd_joint']['left24_mse_rad2']-r['models']['deployed_merged_1']['left24_mse_rad2'])
        data=windows[(example['episode'],example['frame'])];i=data['seeds'].index(example['seed'])
        scale=data['action_scale'].numpy();offset=data['action_offset'].numpy()
        error=(data['predictions']['dmd_joint'][i]-data['predictions']['original_10'][i]).numpy()/scale
        dims=list(np.argsort(np.mean(error[:24,:6]**2,axis=0))[-3:])+[6]
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for axis,d in zip(axes.flat,dims):
            gt=(data['gt'][0].numpy()-offset)/scale;axis.plot(range(1,33),gt[:,d],color='black',linestyle=':',label='Demonstration')
            for name in ['original_10','deployed_merged_1','dmd_joint','cm_flow']:
                pred=(data['predictions'][name][i].numpy()-offset)/scale;axis.plot(range(1,33),pred[:,d],label=labels[name])
            axis.set_title('Left gripper opening' if d==6 else f'Left joint {d} (rad)');axis.set_xlabel('Action index');axis.axvline(24.5,color='gray',linestyle=':')
        axes[0,0].legend(fontsize=8);fig.suptitle(f'Largest DMD joint excess error example: episode {example["episode"]}, frame {example["frame"]}, seed {example["seed"]}')
        fig.tight_layout();fig.savefig(out/'trajectory_example.png',dpi=180);plt.close(fig);dump(out/'trajectory_example.json',example)
    print('MICROWAVE SUMMARY',len(rows),len(windows),flush=True)


def microwave_evidence(args):
    import io
    import re
    import subprocess
    from PIL import Image,ImageDraw
    root=Path('/mnt/afs_1/lvchengtao/code/wam/MeanFlowWAM/evaluate_results/robotwin')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    rates=[];outcomes={};folders={}
    ansi=re.compile(r'\x1b\[[0-9;]*m')
    for file in sorted(root.glob('*/*/summary.json')):
        doc=json.loads(file.read_text())
        found=[r for r in doc.get('per_task',[]) if r['task_name']=='open_microwave']
        if not found or any(found[0].get(k) is None for k in ['clean_success_rate','random_success_rate']):continue
        folder=file.parent;key=str(folder.relative_to(root));folders[key]=folder
        results={};commands=[]
        for path in sorted(folder.glob('eval_open_microwave_*.log')):
            phase=None;previous=0
            for line in ansi.sub('',path.read_text(errors='replace')).splitlines():
                if '| fastwam_policy | demo_' in line:phase='random' if 'demo_randomized' in line else 'clean'
                match=re.search(r'Success rate: (\d+)/(\d+).*current seed: (\d+)',line)
                if match and phase:
                    cumulative,trial,seed=map(int,match.groups())
                    if trial==1:previous=0
                    success=cumulative-previous;assert success in [0,1]
                    video=folder/'open_microwave'/f"episode{trial-1}_randomized-{'true' if phase=='random' else 'false'}_success-{'true' if success else 'false'}.mp4"
                    results[(phase,seed)]={'episode':trial-1,'success':success,'video':str(video),'log':str(path)}
                    previous=cumulative
        if (folder/'manager.log').exists():
            commands=[line for line in (folder/'manager.log').read_text(errors='replace').splitlines() if 'launch task=open_microwave ' in line]
        for phase in ['clean','random']:
            part=[v['success'] for (ph,seed),v in results.items() if ph==phase]
            if part:assert len(part)==100 and np.isclose(np.mean(part),found[0][phase+'_success_rate']),(key,phase,len(part))
        outcomes[key]=results
        rates.append({'run':key,**found[0],'parsed_episodes':len(results),'commands':commands})
    names={
        'original':'robotwin_uncond_3cam_384/robotwin_original_fastwam_10step_20260905_041955',
        'consistency':'robotwin_consistency_joint_ts10_noflow_ema_step30000/robotwin_consistency_joint_ts10_noflow_ema_step30000_1step_20260907_130853',
        'dmd_joint':'robotwin_dmd_action_video_step30000/robotwin_dmd_action_video_step30000_1step_20260907_024202',
        'dmd_action':'robotwin_dmd_v2_step30000/dmd_20260905_043611',
        'meanflow':'robotwin_uncond_3cam_384_mean_flow_1e-4_2026-09-02_03-35-35/robotwin_mean_flow_step117415_step_20260905_044752',
        'cm_flow':'robotwin_consistency_joint_ts10_ema_step30000/robotwin_consistency_joint_ts10_ema_step30000_1step_20260907_111929',
    }
    comparisons={}
    for name,key in names.items():
        left=outcomes[names['consistency']];right=outcomes[key];shared=left.keys()&right.keys()
        comparisons[name]={}
        for phase in ['clean','random','all']:
            selected=[k for k in sorted(shared) if phase=='all' or k[0]==phase]
            rows=[{'episode':k[1],'difference':left[k]['success']-right[k]['success']} for k in selected]
            comparisons[name][phase]={'matched':len(selected),'consistency_wins':sum(left[k]['success']>right[k]['success'] for k in selected),
                                      'comparison_wins':sum(right[k]['success']>left[k]['success'] for k in selected),
                                      'paired_delta':cluster_interval(rows,lambda r:r['difference'])}
    dump(out/'closedloop.json',{'rates':rates,'selected_runs':names,'paired':comparisons})
    common=set.intersection(*[set(outcomes[key]) for name,key in names.items() if name in ['original','consistency','dmd_joint','meanflow']])
    selected=[]
    for phase in ['clean','random']:
        candidates=[k for k in sorted(common) if k[0]==phase and outcomes[names['consistency']][k]['success'] and not outcomes[names['dmd_joint']][k]['success']]
        selected.extend(candidates[:2])
    evidence=[]
    for key in selected:
        phase,seed=key;tiles=[];records=[]
        for name in ['original','consistency','dmd_joint','meanflow']:
            record=outcomes[names[name]][key];path=record['video']
            info=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_format','-show_streams','-of','json',path]))
            duration=float(info['format']['duration']);frames=[]
            for fraction in [0,.08,.16,.28,.42,.6,.8,.98]:
                seconds=max(0,min(duration-.1,duration*fraction))
                data=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',path,'-frames:v','1','-vf','crop=iw/2:ih:0:0,scale=224:-1','-f','image2pipe','-vcodec','png','-threads','1','pipe:1'])
                im=Image.open(io.BytesIO(data)).convert('RGB');tile=Image.new('RGB',(im.width,im.height+30),'white');tile.paste(im,(0,30))
                ImageDraw.Draw(tile).text((4,4),f'{name} {seconds:.1f}/{duration:.1f}s\n'+('success' if record['success'] else 'failure'),fill='black');frames.append(tile)
            tiles.append(frames);records.append({'method':name,**record,'duration':duration})
        w,h=tiles[0][0].size;sheet=Image.new('RGB',(w*8,h*4+25),'white')
        ImageDraw.Draw(sheet).text((5,5),f'open_microwave {phase} seed={seed}; left half of video; relative progress, not synchronized time',fill='black')
        for y,frames in enumerate(tiles):
            for x,im in enumerate(frames):sheet.paste(im,(x*w,y*h+25))
        file=out/f'paired_{phase}_{seed}.jpg';sheet.save(file,quality=95)
        evidence.append({'phase':phase,'seed':seed,'records':records,'sheet':str(file)})
    dump(out/'video_evidence.json',evidence)
    print('MICROWAVE EVIDENCE',len(rates),'runs',len(evidence),'video pairs',flush=True)


def video_evidence(args):
    import io
    import subprocess
    from PIL import Image,ImageDraw
    out=Path(args.output)
    pairs=json.loads((out/'paired_episodes.json').read_text())
    records=[]
    for task in ['place_fan','stack_blocks_three','place_can_basket','open_microwave']:
        candidates=[r for r in pairs if r['task']==task and r['phase']=='random' and
                    r['original']['success'] and not r['noflow']['success'] and
                    all(Path(r[k]['video']).exists() for k in ['original','noflow'])]
        if not candidates:continue
        row=candidates[0];tiles=[];metadata=[]
        for label in ['original','noflow']:
            path=row[label]['video']
            info=json.loads(subprocess.check_output(['ffprobe','-v','quiet','-show_format','-show_streams','-of','json',path]))
            duration=float(info['format']['duration']);images=[]
            for frac in [0,.2,.4,.6,.8,.98]:
                seconds=max(0,min(duration-.1,duration*frac))
                buf=subprocess.check_output(['ffmpeg','-v','error','-ss',str(seconds),'-i',path,'-frames:v','1','-vf','scale=384:-1','-f','image2pipe','-vcodec','png','-threads','1','pipe:1'])
                im=Image.open(io.BytesIO(buf)).convert('RGB')
                tile=Image.new('RGB',(im.width,im.height+26),'white');tile.paste(im,(0,26))
                ImageDraw.Draw(tile).text((6,6),f'{label} {seconds:.1f}s / {duration:.1f}s',fill='black')
                images.append(tile)
            tiles.append(images);metadata.append({'model':label,'duration':duration,'video':path})
        w,h=tiles[0][0].size
        sheet=Image.new('RGB',(w*6,h*2+30),'white')
        ImageDraw.Draw(sheet).text((8,8),f'{task} randomized seed={row["seed"]}; original success / noflow failure; relative video progress',fill='black')
        for y,images in enumerate(tiles):
            for x,im in enumerate(images):sheet.paste(im,(x*w,30+y*h))
        target=out/f'video_{task}_seed{row["seed"]}.jpg';sheet.save(target,quality=94)
        records.append({'pair':row,'videos':metadata,'sheet':str(target)})
    dump(out/'video_evidence.json',records)
    print('VIDEO EVIDENCE',len(records),flush=True)


def component_summary(args):
    rows=[];metas=[];action_files=[];weights=None
    for name in args.component_summary:
        p=Path(name);meta=json.loads((p/'metadata.json').read_text())
        assert meta['completed'],name
        for info in meta['checkpoint_signatures']:assert signature(Path(info['path']))==info
        metas.append(meta)
        rows.extend(json.loads(line) for line in (p/'components.jsonl').read_text().splitlines())
        action_files.extend(sorted(p.glob('actions_*.pt')))
        if (p/'weight_updates.json').exists():weights=json.loads((p/'weight_updates.json').read_text())
    assert all(m['checkpoint_signatures']==metas[0]['checkpoint_signatures'] for m in metas)
    keys=[(r['episode'],r['frame'],r['seed']) for r in rows]
    assert len(set(keys))==len(keys),'Duplicate samples'
    row_map={k:r for k,r in zip(keys,rows)}
    # Recompute the orthonormal transform in float64, independent of GPU TF32 settings.
    n=32;t=np.arange(n,dtype=np.float64)
    basis=np.cos(np.pi/n*(t[None,:]+.5)*t[:,None])*(2/n)**.5;basis[0]/=2**.5
    for path in action_files:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        for i,seed in enumerate(saved['seeds']):
            row=row_map[(saved['item']['episode'],saved['item']['frame'],seed)]
            for target,reference in [('metrics_vs_original10',saved['predictions']['original_10'][i]),('metrics_vs_gt',saved['gt'][0])]:
                for name,pred in saved['predictions'].items():
                    error=pred[i].numpy().astype(np.float64)-reference.numpy().astype(np.float64)
                    spectrum=(basis@error)**2
                    row[target][name]['dct_error_energy']={'dc':float(spectrum[0].sum()),'low_1_3':float(spectrum[1:4].sum()),
                                                          'mid_4_7':float(spectrum[4:8].sum()),'high_8_31':float(spectrum[8:].sum())}
    for row in rows:
        for metric in row['metrics_vs_original10'].values():
            assert np.isclose(np.mean(metric['per_dim_mse']),metric['normalized_mse'],atol=1e-7,rtol=1e-5)
            assert np.isclose(sum(metric['dct_error_energy'].values())/448,metric['normalized_mse'],atol=1e-7,rtol=1e-5)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary={'samples':len(rows),'episodes':len({r['episode'] for r in rows}),
             'windows':len({(r['episode'],r['frame']) for r in rows}),'sources':args.component_summary,
             'checkpoint_signatures':metas[0]['checkpoint_signatures'],'variants':{},'paired_changes':{}}
    names=list(rows[0]['metrics_vs_original10'])
    scalar_keys=[k for k,v in rows[0]['metrics_vs_original10'][names[0]].items() if isinstance(v,(int,float))]
    for name in names:
        entry={}
        for target in ['metrics_vs_original10','metrics_vs_gt']:
            entry[target]={key:cluster_interval(rows,lambda r,n=name,t=target,k=key:r[t][n][k]) for key in scalar_keys}
            for key in ['per_dim_mse','per_step_mse']:
                entry[target][key]=np.mean([r[target][name][key] for r in rows],axis=0).tolist()
            energy={key:float(np.mean([r[target][name]['dct_error_energy'][key] for r in rows]))
                    for key in ['dc','low_1_3','mid_4_7','high_8_31']}
            entry[target]['mean_dct_error_energy']=energy
            entry[target]['dct_error_energy_fraction']={k:v/sum(energy.values()) if sum(energy.values()) else 0 for k,v in energy.items()}
        summary['variants'][name]=entry
        summary['paired_changes'][name]={target:{key:cluster_interval(rows,lambda r,n=name,t=target,k=key:r[t][n][k]-r[t]['noflow_1'][k])
                                                 for key in ['normalized_mse','first24_mse','joint_mae_rad','gripper_mae']}
                                         for target in ['metrics_vs_original10','metrics_vs_gt']}
    summary['phase_metrics']={str(phase):{n:cluster_interval([r for r in rows if r['phase']==phase],lambda r,n=n:r['metrics_vs_original10'][n]['normalized_mse'])
                                        for n in names} for phase in sorted({r['phase'] for r in rows})}
    summary['video_cache_drift']=[{'block':i,**{k:{metric:float(np.mean([r['video_cache_drift'][i][k][metric] for r in rows]))
                                                   for metric in ['cos','relative_l2']} for k in rows[0]['video_cache_drift'][i] if k!='block'}}
                                  for i in range(len(rows[0]['video_cache_drift']))]
    summary['trajectory_probes']={sigma:{metric:cluster_interval(rows,lambda r,s=sigma,m=metric:r['teacher_trajectory_probes'][s][m])
                                        for metric in rows[0]['teacher_trajectory_probes'][sigma]}
                                  for sigma in rows[0]['teacher_trajectory_probes']}
    if weights:
        groups={}
        for branch in ['action','video']:
            for group in ['all','early','middle','late','self_attn','cross_attn','ffn']:
                selected=[w for w in weights if w['branch']==branch and
                          (group=='all' or group=='early' and w['block']<10 or group=='middle' and 10<=w['block']<20 or
                           group=='late' and w['block']>=20 or w['module']==group)]
                groups[branch+'_'+group]={'relative_update_frobenius':(sum(w['update_norm']**2 for w in selected)/sum(w['base_norm']**2 for w in selected))**.5,
                                         'matrices':len(selected)}
        summary['weight_groups']=groups
        summary['largest_relative_updates']=sorted(weights,key=lambda w:w['relative_update'],reverse=True)[:30]
    distribution=[];factorial=[]
    for p in action_files:
        data=torch.load(p,map_location='cpu',weights_only=True);pred={k:v.numpy().astype(np.float64) for k,v in data['predictions'].items()}
        ref=pred['original_10'];ref_mean=ref.mean(0);ref_centered=ref-ref_mean
        ref_var=float(np.mean(ref_centered**2));item={'episode':data['item']['episode']}
        for name,x in pred.items():
            mean=x.mean(0);centered=x-mean
            item[name]={'mean_component_mse':float(np.mean((mean-ref_mean)**2)),
                        'noise_dependent_component_mse':float(np.mean((centered-ref_centered)**2)),
                        'noise_variance':float(np.mean(centered**2)),
                        'noise_variance_ratio':float(np.mean(centered**2))/ref_var if ref_var>1e-15 else None,
                        'paired_centered_cos':float(np.sum(centered*ref_centered)/(np.linalg.norm(centered)*np.linalg.norm(ref_centered))) if np.linalg.norm(centered)*np.linalg.norm(ref_centered)>1e-15 else None,
                        'temporal_first_difference_energy':float(np.mean(np.diff(x,axis=1)**2))}
            assert np.isclose(item[name]['mean_component_mse']+item[name]['noise_dependent_component_mse'],np.mean((x-ref)**2))
        if 'deployed_merged_1' in pred:
            item['merge_audit']={'mse':float(np.mean((pred['deployed_merged_1']-pred['noflow_1'])**2)),
                                 'max_absolute_difference':float(np.max(np.abs(pred['deployed_merged_1']-pred['noflow_1'])))}
        if 'original_fp32_head_10' in pred:
            item['precision_reference']={name:float(np.mean((x-pred['original_fp32_head_10'])**2)) for name,x in pred.items()}
        distribution.append(item)
        a=pred['original_video_noflow_action_1']-ref
        v=pred['noflow_video_original_action_10']-ref
        interaction=pred['noflow_1']-ref-a-v
        err=pred['noflow_1']-ref
        factorial.append({'episode':item['episode'],'action_plus_nfe_mse':float(np.mean(a*a)),
                          'video_only_mse':float(np.mean(v*v)),'interaction_mse':float(np.mean(interaction**2)),
                          'action_video_inner_product':float(np.mean(a*v)),
                          'interaction_error_inner_product':float(np.mean(interaction*err))})
    summary['noise_decomposition']={name:{k:cluster_interval(distribution,lambda r,n=name,k=k:r[n][k]) for k in distribution[0][name]} for name in names}
    summary['noise_variance_pooled_ratio']={name:cluster_ratio(distribution,lambda r,n=name:r[n]['noise_variance'],lambda r:r['original_10']['noise_variance']) for name in names}
    if 'merge_audit' in distribution[0]:
        summary['merge_audit']={k:cluster_interval(distribution,lambda r,k=k:r['merge_audit'][k]) for k in distribution[0]['merge_audit']}
        summary['paired_vs_deployed']={name:{target:{'mse_delta':cluster_interval(rows,lambda r,n=name,t=target:r[t][n]['normalized_mse']-r[t]['deployed_merged_1']['normalized_mse']),
                                                     'mse_ratio':cluster_ratio(rows,lambda r,n=name,t=target:r[t][n]['normalized_mse'],lambda r,t=target:r[t]['deployed_merged_1']['normalized_mse'])}
                                              for target in ['metrics_vs_original10','metrics_vs_gt']} for name in names}
    if 'precision_reference' in distribution[0]:
        summary['vs_fp32_head_reference']={name:{'mse':cluster_interval(distribution,lambda r,n=name:r['precision_reference'][n]),
                                                'ratio_vs_deployed':cluster_ratio(distribution,lambda r,n=name:r['precision_reference'][n],lambda r:r['precision_reference']['deployed_merged_1'])}
                                           for name in names}
    summary['factorial_decomposition']={k:cluster_interval(factorial,lambda r,k=k:r[k]) for k in factorial[0] if k!='episode'}
    dump(out/'component_summary.json',summary)
    lines=['variant,reference_mse,reference_first24_mse,joint_mae_rad,gripper_mae,gt_mse,delta_reference_mse_vs_noflow,delta_ci_low,delta_ci_high']
    for name in names:
        entry=summary['variants'][name];ref=entry['metrics_vs_original10'];change=summary['paired_changes'][name]['metrics_vs_original10']['normalized_mse']
        values=[name,*[ref[k]['mean'] for k in ['normalized_mse','first24_mse','joint_mae_rad','gripper_mae']],
                entry['metrics_vs_gt']['normalized_mse']['mean'],change['mean'],*change['ci95']]
        lines.append(','.join(map(str,values)))
    (out/'variants.csv').write_text('\n'.join(lines)+'\n')
    if args.plot_deps:
        sys.path.insert(0,args.plot_deps)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(16,4))
        selected=['original_1','noflow_1','original_video_noflow_action_1','noflow_video_original_action_10','original_20']
        labels=['Original 1-step','Noflow 1-step','Restore video','Video swap / base action','Original 20-step']
        values=[summary['variants'][n]['metrics_vs_original10']['normalized_mse'] for n in selected]
        means=np.array([v['mean'] for v in values]);ci=np.array([v['ci95'] for v in values])
        ax[0].barh(labels,means,color=['#777777','#168b80','#df7860','#cbad42','#777777'])
        ax[0].errorbar(means,range(len(means)),xerr=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means])),fmt='none',color='black')
        ax[0].set_xlabel('Normalized MSE vs original 10-step')
        ax[0].ticklabel_format(axis='x',style='sci',scilimits=(0,0))
        for name,label in zip(selected[1:3],labels[1:3]):
            ax[1].plot(range(1,33),summary['variants'][name]['metrics_vs_original10']['per_step_mse'],label=label)
        ax[1].axvline(24.5,color='gray',linestyle=':');ax[1].set_xlabel('Action index');ax[1].set_ylabel('MSE');ax[1].legend()
        for key in ['k','v']:
            if key in summary['video_cache_drift'][0]:
                ax[2].plot([r['block'] for r in summary['video_cache_drift']],[r[key]['cos'] for r in summary['video_cache_drift']],label=key.upper())
        ax[2].set_xlabel('Video block');ax[2].set_ylabel('Original / noflow cache cosine');ax[2].legend()
        fig.tight_layout();fig.savefig(out/'components.png',dpi=180);plt.close(fig)
        if 'paired_vs_deployed' in summary:
            selected=['deployed_original_video_1','deployed_restore_video_early','deployed_restore_video_middle',
                      'deployed_restore_video_late','deployed_fp32_subtract_1','deployed_fp32_head_1']
            labels=['Restore full video','Restore video blocks 0-9','Restore video blocks 10-19',
                    'Restore video blocks 20-29','FP32 subtraction only','FP32 head + subtraction']
            ratios=[summary['paired_vs_deployed'][n]['metrics_vs_original10']['mse_ratio'] for n in selected]
            means=np.array([r['ratio_of_means']-1 for r in ratios])*100
            ci=(np.array([r['ci95'] for r in ratios])-1)*100
            fig,ax=plt.subplots(1,2,figsize=(13,4))
            ax[0].barh(labels,means,color=['#777777']*4+['#168b80']*2)
            ax[0].errorbar(means,range(len(means)),xerr=np.maximum(0,np.stack([means-ci[:,0],ci[:,1]-means])),fmt='none',color='black')
            ax[0].axvline(0,color='black',linewidth=.6);ax[0].set_xlabel('MSE change vs deployed noflow (%) / paired 95% CI')
            for name,label in [('deployed_merged_1','Deployed noflow'),('deployed_fp32_head_1','FP32 head + subtraction')]:
                ax[1].plot(range(1,33),summary['variants'][name]['metrics_vs_original10']['per_step_mse'],label=label)
            ax[1].axvline(24.5,color='gray',linestyle=':');ax[1].set_xlabel('Action index');ax[1].set_ylabel('MSE vs original 10-step')
            ax[1].ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax[1].legend()
            fig.tight_layout();fig.savefig(out/'deployed_interventions.png',dpi=180);plt.close(fig)
    print('COMPONENT SUMMARY',len(rows),len(distribution),out,flush=True)


if __name__ == '__main__':
    main()
