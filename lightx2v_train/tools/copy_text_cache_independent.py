"""Copy a flat .pt cache with parallel rsync, then verify every file by checksum.

Source is read-only. Existing destination files are never replaced. A checksum
mismatch fails the job for manual investigation, rather than overwriting data.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--job-dir', required=True)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    source, destination = Path(args.source), Path(args.destination)
    if source.is_symlink() or destination.is_symlink():
        raise ValueError('Cache directories must not be symlinks.')
    source, destination = source.resolve(strict=True), destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be disjoint directories.')
    if not source.is_dir() or not 1 <= args.workers <= 32:
        raise ValueError('Require a source directory and 1..32 workers.')
    job_dir = Path(args.job_dir).resolve()
    job_dir.mkdir(parents=True, exist_ok=False)
    destination.mkdir(parents=True, exist_ok=True)
    names = []
    with os.scandir(source) as entries:
        for entry in entries:
            if not entry.name.endswith('.pt'):
                continue
            if not entry.is_file(follow_symlinks=False):
                raise ValueError(f'Refusing non-regular source cache: {entry.path}')
            names.append(entry.name)
    names.sort()
    if not names:
        raise ValueError('Source contains no .pt cache files.')
    with os.scandir(destination) as entries:
        for entry in entries:
            if entry.is_symlink():
                raise ValueError(f'Refusing destination symlink: {entry.path}')
    shards = [names[index::args.workers] for index in range(args.workers)]
    manifests = []
    for index, shard in enumerate(shards):
        if shard:
            manifest = job_dir / f'files-{index:02d}.nul'
            manifest.write_bytes(b'\0'.join(os.fsencode(name) for name in shard) + b'\0')
            manifests.append((index, manifest, len(shard)))
    state = dict(pid=os.getpid(), source=str(source), destination=str(destination),
                 files=len(names), workers=len(manifests), started_at=time.time(), phase='copying')

    def report():
        state['updated_at'] = time.time()
        temporary = job_dir / 'status.json.tmp'
        temporary.write_text(json.dumps(state, indent=2) + '\n')
        temporary.replace(job_dir / 'status.json')
        print(json.dumps(state), flush=True)

    def run_shard(item, verify):
        index, manifest, count = item
        log = job_dir / f'{"verify" if verify else "copy"}-{index:02d}.log'
        flags = ['-rnc', '--itemize-changes'] if verify else ['-rt', '--ignore-existing', '--info=progress2,stats2']
        command = ['rsync', *flags, '--no-links', '--from0', f'--files-from={manifest}',
                   str(source) + '/', str(destination) + '/']
        with log.open('wb') as handle:
            subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=True)
        if verify and log.stat().st_size:
            raise RuntimeError(f'Checksum verification found differences: {log}')
        return count

    try:
        for verify in (False, True):
            state.update(phase='verifying' if verify else 'copying', completed_shards=0, phase_completed_files=0)
            report()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                pending = {pool.submit(run_shard, item, verify) for item in manifests}
                while pending:
                    done, pending = wait(pending, timeout=20)
                    for future in done:
                        state['phase_completed_files'] += future.result()
                        state['completed_shards'] += 1
                    report()
        # rsync does not create hardlinks here; also audit all destination inodes.
        state.update(phase='checking_independence')
        report()
        for name in names:
            src, dst = source / name, destination / name
            if dst.is_symlink() or os.path.samestat(src.stat(), dst.stat()):
                raise RuntimeError(f'Destination is not an independent file: {dst}')
        state.update(phase='complete', finished_at=time.time(), all_checksums_match=True, independent_files=True)
        report()
    except BaseException as error:
        state.update(phase='failed', error=repr(error))
        report()
        raise


if __name__ == '__main__':
    main()
