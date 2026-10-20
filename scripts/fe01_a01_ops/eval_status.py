#!/usr/bin/env python3
"""Read-only status of the five sealed validation results; never submit jobs."""
import argparse
import importlib.util
from pathlib import Path
import re
import subprocess

LABELS = ('TEAM OFF', 'PhaseNet OFF', 'EQT OFF', 'DiTing ON', 'DiTing OFF')
ACTIVE = {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING', 'SUSPENDED',
          'RESIZING', 'REQUEUED', 'REQUEUE_HOLD', 'SIGNALING', 'STAGE_OUT'}
FAILED = {'FAILED', 'CANCELLED', 'TIMEOUT', 'NODE_FAIL', 'OUT_OF_MEMORY',
          'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED', 'SPECIAL_EXIT'}
OFF_COMMAND = 'bash a01_diting_eval_fix_20261020/submit_diting_eval_fix.sh'
PACK_COMMAND = 'bash a01_diting_eval_fix_20261020/submit_diting_eval_fix.sh pack'


def helper():
    path = Path(__file__).with_name('eval_pack.py')
    spec = importlib.util.spec_from_file_location('a01_status_result_gates', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def submissions(session):
    """Use append order: the TEAM retry supersedes its first failed array task."""
    path = Path(session) / 'jobs.tsv'
    latest = {}
    if not path.exists():
        return latest
    allowed = {'eval_first': {0, 1, 2, 3}, 'eval_team': {0},
               'eval_diting_off': {4}, 'pack': set()}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split('\t')
        if len(fields) != 3 or fields[0] not in allowed:
            raise ValueError('jobs.tsv 第 %d 行无法识别，停止判断提交状态' % number)
        stage, indices, raw_job = fields
        job = raw_job.split(';', 1)[0]
        if not re.fullmatch(r'[0-9]+', job):
            raise ValueError('jobs.tsv 第 %d 行作业号无效' % number)
        if stage == 'pack':
            if indices:
                raise ValueError('pack 提交记录意外包含数组编号')
            latest['pack'] = dict(stage=stage, job=job, task=job)
            continue
        if not re.fullmatch(r'[0-4](?:,[0-4])*', indices):
            raise ValueError('jobs.tsv 第 %d 行数组编号无效' % number)
        values = [int(value) for value in indices.split(',')]
        if not set(values).issubset(allowed[stage]):
            raise ValueError('jobs.tsv 提交阶段与数组编号不符')
        for index in values:
            latest[index] = dict(stage=stage, job=job, task=job + '_' + str(index))
    return latest


def scheduler(latest):
    """Query once per service. Missing accounting never means a completed eval."""
    jobs = ','.join(sorted({row['job'] for row in latest.values()}))
    if not jobs:
        return {}, []
    states, warnings = {}, []
    queries = (
        # Older cluster sacct rejects --array. JobID already identifies each
        # recorded task; never infer a task's state from its array parent.
        ['sacct', '-n', '-P', '-X', '-j', jobs,
         '--format=JobID%64,State%40,ExitCode'],
        ['squeue', '-h', '-r', '-j', jobs, '-o', '%i|%T'],
    )
    for command in queries:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            if result.returncode:
                raise RuntimeError(result.stderr.strip() or 'exit=' + str(result.returncode))
            for line in result.stdout.splitlines():
                fields = [value.strip() for value in line.split('|')]
                if len(fields) < 2 or not re.fullmatch(r'[0-9]+(?:_[0-9]+)?', fields[0]):
                    continue
                state = fields[1].split()[0].rstrip('+') if fields[1] else 'UNKNOWN'
                states[fields[0]] = dict(state=state, exit_code=fields[2] if len(fields) > 2 else '')
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
            warnings.append(command[0] + ' 查询不可用：' + str(exc))
    return states, warnings


def job_state(record, states):
    return states.get(record['task'], {}) if record else {}


def pending_status(record, states):
    state = job_state(record, states).get('state', 'UNKNOWN')
    if state in ACTIVE:
        return 'QUEUED' if state in {'PENDING', 'REQUEUED', 'REQUEUE_HOLD'} else 'RUNNING'
    if state in FAILED:
        return 'FAILED'
    if state == 'COMPLETED':
        return 'UNSEALED'
    return 'UNKNOWN' if record else 'NO_RECORD'


def archive_status(session, ep, record, states):
    archive = session / 'review.tar.gz'
    checksum = session / 'review.tar.gz.sha256'
    pending = pending_status(record, states)
    # A running pack may be writing the archive or its checksum right now.
    if pending in {'RUNNING', 'QUEUED'}:
        return pending, ''
    if archive.exists() and checksum.is_file():
        try:
            fields = checksum.read_text().strip().split()
            if (len(fields) != 2 or not re.fullmatch(r'[0-9a-f]{64}', fields[0]) or
                    fields[1].lstrip('*') != archive.name or archive.stat().st_size == 0 or
                    ep.core()[1].sha256(archive) != fields[0]):
                raise ValueError('review.tar.gz SHA256 校验未通过')
            return 'COMPLETE', ''
        except (OSError, ValueError) as exc:
            return 'INVALID', str(exc)
    if archive.exists() or checksum.exists():
        return 'INVALID', '归档或 SHA256 文件不完整；请检查 pack 日志，不要覆盖现有归档'
    return ('NOT_STARTED', '') if not record else (pending, '')


def inspect(session, ep=None, query=scheduler, emit=print):
    session = Path(session).resolve()
    ep = ep or helper()
    plan = ep.load_plan(session)
    if [row['index'] for row in plan['runs']] != list(range(5)):
        raise ValueError('评价计划必须包含按顺序排列的 0–4 五项')
    records = submissions(session)
    states, warnings = query(records)
    emit('评价目录：' + str(session))
    emit('正在逐项校验完整结果及 SHA256；大文件可能需要几分钟。')
    rows = []
    text = dict(COMPLETE='完成，完整性检查通过', INVALID='结果校验失败',
                QUEUED='已提交，正在排队', RUNNING='正在运行或收尾', FAILED='作业失败',
                UNSEALED='作业已结束，但缺少完整成功结果', UNKNOWN='已提交，当前作业状态未知',
                NO_RECORD='未见提交记录，尚无成功结果')
    for index, label in enumerate(LABELS):
        record = records.get(index)
        directory = session / 'results' / ('index' + str(index))
        error = ''
        if directory.exists():
            try:
                ep.result_gate(session, index)
                status = 'COMPLETE'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                status, error = 'INVALID', str(exc)
        else:
            status = pending_status(record, states)
        row = dict(index=index, label=label, status=status, error=error,
                   submission=record, scheduler=job_state(record, states))
        rows.append(row)
        detail = '' if not record else '；作业 ' + record['task'] + ' ' + row['scheduler'].get('state', 'UNKNOWN')
        emit('%d  %-12s %s%s' % (index, label, text[status], detail))
        if error:
            emit('   原因：' + error)
        if record and status not in {'COMPLETE', 'QUEUED'}:
            emit('   日志：' + str(session / 'slurm' / (record['stage'] + '_' + record['job'] + '_' + str(index) + '.err')))
    complete = [row['index'] for row in rows if row['status'] == 'COMPLETE']
    ready = len(complete) == 5
    emit('已完成并校验通过：%d/5；PACK_READY=%s' % (len(complete), 'YES' if ready else 'NO'))
    emit('未完成编号：' + (','.join(str(row['index']) for row in rows if row['status'] != 'COMPLETE') or '无'))
    for warning in warnings:
        emit('提示：' + warning)
    pack, error = archive_status(session, ep, records.get('pack'), states)
    emit('PACK_STATUS=' + pack)
    if error:
        emit('pack 检查：' + error)
    if pack == 'COMPLETE':
        emit('打包文件 SHA256 校验通过。' + ('' if ready else '当前 eval 检查未全部通过，请先核查。'))
        emit('下载：' + str(session / 'review.tar.gz'))
        emit('下载：' + str(session / 'review.tar.gz.sha256'))
    elif pack in {'RUNNING', 'QUEUED'}:
        emit('pack 已提交，请等待结束后重新运行本检查；不重复提交。')
    elif pack != 'NOT_STARTED':
        emit('pack 未确认成功，请检查 ' + str(session / 'slurm') + ' 内 pack 日志。')
    elif ready:
        emit('现在可以在 FE01 根目录手动提交 pack（CPU sbatch）：')
        emit(PACK_COMMAND)
    elif rows[3]['status'] == 'COMPLETE' and rows[4]['status'] == 'NO_RECORD':
        emit('DiTing ON 已完成，现在可在 FE01 根目录手动提交 DiTing OFF 的 4 号 eval：')
        emit(OFF_COMMAND)
        emit('0/1/2 如仍在运行可同时等待；4 完成后重新检查，五项齐全才 pack。')
    else:
        emit('暂时不能 pack。等待运行项结束；失败、未知或校验失败的项目先检查上述日志。')
        if rows[3]['status'] != 'COMPLETE' and rows[4]['status'] == 'NO_RECORD':
            emit('DiTing ON（3）完整通过后，再提交 DiTing OFF（4）：' + OFF_COMMAND)
    emit('本脚本只读检查，不提交、取消或修改任何作业与结果。')
    return dict(rows=rows, pack_ready=ready, pack_status=pack, warnings=warnings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True)
    args = parser.parse_args()
    try:
        inspect(args.session, emit=lambda value: print(value, flush=True))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, '检查停止；尚不能确认 pack 就绪：' + str(exc) + '\n')


if __name__ == '__main__':
    main()
