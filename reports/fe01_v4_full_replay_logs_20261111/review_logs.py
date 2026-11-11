"""Local, stdlib-only review of the downloaded Slurm log ZIP; no jobs/models."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import zipfile

EXPECTED_ZIP_SHA = 'e4733a6f0733633d9045b03750d03cfcb67545185aa5047e3ffb6071d321e66c'
POOLS = {'replay_stations': ('33474027', 328), 'replay_grid_normal': ('33474094', 164)}
EPOCHS = {'DiTing': 12, 'PhaseNet': 12, 'EQT': 10, 'TEAM': 11}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def stderr_review(text):
    categories, other = Counter(), []
    for number, line in enumerate(text.splitlines(), 1):
        if line.startswith('srun: ROUTE: split_hostlist:'):
            categories['slurm_route'] += 1
        elif 'FutureWarning: Importing from timm.models.layers is deprecated' in line:
            categories['timm_import_future_warning'] += 1
        elif 'FutureWarning: The behavior of DataFrame concatenation with empty or all-NA entries is deprecated.' in line:
            categories['pandas_concat_future_warning'] += 1
        elif line.strip().startswith('warnings.warn(') or line.strip() == 'grid_output=pd.concat([grid_output,invalid],ignore_index=True)':
            categories['warning_source_continuation'] += 1
        elif not line.strip():
            categories['blank'] += 1
        else:
            other.append(dict(line=number, text=line))
    require(not other, 'Unclassified stderr; inspect before declaring log pass: ' + str(other[:5]))
    return dict(categories=categories, unclassified_lines=other)


def review(archive, plan_path, output):
    archive, plan_path, output = Path(archive), Path(plan_path), Path(output)
    require(not output.exists(), 'Preserve existing review; use a new output directory')
    require(sha(archive.read_bytes()) == EXPECTED_ZIP_SHA, 'Wrong log archive')
    plan = json.loads(plan_path.read_text())
    pilot_review = Path(__file__).resolve().parents[1] / 'fe01_v4_pilot_results_20261109/verification.json'
    require(sha(plan_path.read_bytes()) == json.loads(pilot_review.read_text())['source_plan_sha256'],
            'Plan bytes differ from the previously reviewed pilot bundle')
    require(plan['source_identity']['result_commit'] == '50672e131d876a604432b283162d5b0189d40cf0', 'Wrong frozen main')
    require(plan['run_id'] == 'on_v4_20261025_inventory_fix', 'Wrong run')
    workers, latency, sources = [], [], {}
    with zipfile.ZipFile(archive) as z:
        require(z.testzip() is None, 'ZIP CRC failure')
        names = z.namelist()
        require(len(names) == len(set(names)) and len(names) == 24, 'Missing/duplicate log members')
        for name in names:
            data = z.read(name)
            sources[name] = dict(bytes=len(data), sha256=sha(data))
        for stage, (job, total) in POOLS.items():
            require(len(plan['tasks'][stage]) == total, 'Logical task population differs from frozen plan')
            coverage = []
            for slot in range(4):
                stem = stage + '_pool_' + job + '_' + str(slot)
                lines = z.read(stem + '.out').decode('utf-8').splitlines()
                begin, ended, current, inventory, summaries, already = [], [], None, {}, [], []
                for n, line in enumerate(lines, 1):
                    m = re.fullmatch(r'BEGIN_LOGICAL_TASK (\S+) (\d+) scheduler_slot=(\d+)', line)
                    if m:
                        require(m[1] == stage and int(m[3]) == slot and current is None, 'Wrong/overlapping logical task')
                        current = int(m[2]); begin.append(current)
                        continue
                    m = re.fullmatch(r'INVENTORY_CHECK (\S+) selected epoch (\d+)', line)
                    if m:
                        require(current is not None, 'Checkpoint identity outside task')
                        expected = plan['tasks'][stage][current]['family']
                        require(m[1] == expected and int(m[2]) == EPOCHS[expected], 'Checkpoint family/epoch differs')
                        inventory[current] = dict(family=m[1], epoch=int(m[2]), passed=False)
                        continue
                    m = re.fullmatch(r'INVENTORY_PASS (\S+) selected', line)
                    if m:
                        require(current in inventory and inventory[current]['family'] == m[1], 'Unmatched checkpoint pass')
                        inventory[current]['passed'] = True
                        continue
                    if line == 'ALREADY_COMPLETE':
                        already.append(current)
                    m = re.fullmatch(r'END_LOGICAL_TASK (\S+) (\d+) exit=(-?\d+)', line)
                    if m:
                        require(m[1] == stage and int(m[2]) == current and int(m[3]) == 0, 'Failed/unmatched logical task')
                        ended.append(dict(index=current, exit_code=int(m[3]), stdout_line=n)); current = None
                        continue
                    if line.startswith('POOL_SUMMARY='):
                        path = line.split('=', 1)[1]
                        require('/operational_submit/' in path, 'Unexpected summary location')
                        summaries.append('operational_submit/' + path.split('/operational_submit/', 1)[1])
                expected = list(range(slot, total, 4))
                require(begin == expected and [v['index'] for v in ended] == expected and current is None,
                        'Missing/duplicate/reordered task coverage')
                require(not already and len(inventory) == len(expected) and all(v['passed'] for v in inventory.values()),
                        'Incomplete checkpoint identity evidence')
                require(len(summaries) == 1 and lines[-1].startswith('POOL_SUMMARY='), 'Missing final pool summary marker')
                err = stderr_review(z.read(stem + '.err').decode('utf-8'))
                workers.append(dict(stage=stage, array_job_id=job, scheduler_slot=slot,
                                    logical_tasks=len(expected), original_indices=expected,
                                    exit_zero_count=len(ended), checkpoint_pass_count=len(inventory),
                                    family_task_counts=dict(Counter(v['family'] for v in inventory.values())),
                                    summary_relative_path=summaries[0], stderr=err,
                                    source_stdout=stem + '.out', source_stderr=stem + '.err'))
                coverage.extend(v['index'] for v in ended)
            require(sorted(coverage) == list(range(total)), 'Full stage task population mismatch')
        for slot, (family, epoch) in enumerate(EPOCHS.items()):
            stem = 'latency_33441392_' + str(slot)
            text = z.read(stem + '.out').decode('utf-8')
            require('START_STAGE latency TASK ' + str(slot) in text, 'Latency start identity absent')
            require('INVENTORY_CHECK ' + family + ' selected epoch ' + str(epoch) in text
                    and 'INVENTORY_PASS ' + family + ' selected' in text, 'Latency checkpoint identity absent')
            latency.append(dict(task=slot, family=family, epoch=epoch, job_id='33441392',
                                device_visible='ON_V4_DEVICE 1.13.1 5.4.23191-0e6bf66b True' in text,
                                evidence='START_AND_INVENTORY_PASS_ONLY; terminal state/seal not supplied',
                                stderr=stderr_review(z.read(stem + '.err').decode('utf-8'))))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        task_id='20261111-fe01-v4-full-replay-log-review', repository='rular099/team_pytorch',
        review_branch='handoff/fe01-on-20261108', review_base_commit='ecb7dfba2b68b40cab36cb59f3af2820e3dd9438',
        expected_operational_package='1cebcdc61285fb99767509cb2e513bee07aefb05',
        exact_operational_identity_readback='Pool operational JSON/manifests not included in log ZIP.',
        archive=dict(name=archive.name, bytes=archive.stat().st_size, sha256=EXPECTED_ZIP_SHA,
                     crc='PASS', member_count=len(sources), external_sha_sidecar='not supplied; SHA recorded locally'),
        frozen_plan_sha256=sha(plan_path.read_bytes()), frozen_run=plan['run_id'],
        scope='Expected frozen plan: validation, seed42, ON, fixed_query_replay_v4 and normal grid; selected epochs12/12/10/11. Prediction row provenance still requires artifacts/seals.',
        source_members=sources, workers=workers, latency=latency,
        stage_summary={s:dict(logical_tasks=n, exit_zero=n, workers=4,
                            evidence='ALL_LOGICAL_TASKS_EXIT_ZERO_IN_LOGS', missing_or_failed_indices=[])
                       for s, (j, n) in POOLS.items()},
        verified_facts=['All 328 station and 164 normal-grid logical tasks occur once in their expected scheduler slots and return exit0.',
                        'All eight pool logs print their final worker_summary.json paths.',
                        'Every logged logical task confirms the checkpoint family/selected epoch expected by the frozen plan.',
                        'All stderr lines classify as Slurm route messages, timm import or pandas concat FutureWarnings and their source continuations. No unclassified error lines.',
                        'The four latency logs confirm device availability and selected checkpoint identity, without an unambiguous completion marker.'],
        inferences=['The original execute returns to the child after the attempt context seals/publishes successfully, so exit0 supports successful execution at that time. Actual saved payloads/seals were not included for independent local verification.',
                    'The latency implementation has no final stdout completion marker; quiet logs alone do not establish failure, a stall, or completion.'],
        completion_boundary='LOG_EXECUTION_SUCCESS, not independent COMPLETE_SEAL_VERIFIED; no actual prediction rows or metrics received',
        next_step='User may explicitly request compute submit.sh --stage analyze. It verifies all original replay/covariates/crossing-truth seals on compute before scheduling analysis. Latency is required by pack, not analyze.',
        unchanged='No execution/model/config/threshold changes. No warning suppression, retraining, replays or latency resubmission.',
        verification='Stdlib log parser executed on the supplied ZIP and frozen plan; no production model or cluster job executed.')
    (output / 'verification.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(dict(status='PASS', stage_summary=report['stage_summary'], latency='completion unknown',
                         grid_concat_future_warnings=sum(w['stderr']['categories'].get('pandas_concat_future_warning', 0) for w in workers)), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    review(args.archive, args.plan, args.output_dir)
