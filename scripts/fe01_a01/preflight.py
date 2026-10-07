#!/usr/bin/env python3
"""LOGIN NODE: small JSON/SHA checks only; no torch, weights or scheduler."""
import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
# This frozen module uses only the standard library for our JSON configs.
from fe01.config import fingerprint, load, sha256


def current_code_sha():
    files = {}
    for directory in ('fe01_a01', 'scripts/fe01_a01', 'configs/fe01_a01'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.json','.sh','.sbatch'):
                files[str(path.relative_to(ROOT))] = sha256(path)
    for path in sorted((ROOT/'fe01_review').glob('*.py')):
        files[str(path.relative_to(ROOT))] = sha256(path)
    return fingerprint(files)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_file(), '前置文件缺失，先完成上一阶段：'+str(path))
    return json.loads(path.read_text())


def check(stage, indices, output):
    output = Path(output)
    if stage == 'environment':
        target = output/'environment'
        require(not target.exists() or not any(target.iterdir()), '环境输出已有内容，请保留并使用新批次')
        return
    code_sha = current_code_sha()
    env = read(output/'environment/environment.json')
    require(env['status']=='PASS' and env['source']['a01_code_sha256']==code_sha,
            '当前源码的environment门控未通过；先提交environment并等待成功')
    if stage == 'pack':
        require(not (output/'review.tar.gz').exists(), '审阅包已存在，请保留并使用新文件/批次')
        return
    matrix = read(ROOT/'configs/fe01_a01/matrix.json')['new_training']
    def audit_for(index):
        cfg = load(ROOT/matrix[index]['config'])
        require(Path(cfg['output_root']).resolve()==output.resolve(), '配置输出根与提交输出根不一致')
        audit_dir = output/'audits'/cfg['run_id']
        path = audit_dir/'protocol.lock.json'
        lock = read(path)
        require(lock['status']=='AUDIT_PASS' and lock['a01_code_sha256']==code_sha and
                lock['config_sha256']==fingerprint(cfg), 'audit门控失败/过期：'+str(path))
        return cfg, path, lock
    for index in indices:
        if stage == 'audit':
            # Actual checkpoints/data are authenticated on the compute node.
            cfg = load(ROOT/matrix[index]['config'])
            target = output/'audits'/cfg['run_id']
            require(not target.exists() or not any(target.iterdir()), 'audit输出已有内容，请保留并使用新批次')
            continue
        cfg, audit_path, lock = audit_for(index)
        if cfg['model_family']=='diting_pretrained_frozen':
            _, _, other = audit_for(4 if index==3 else 3)
            require(other['sampling']==lock['sampling'], 'DiTing ON/OFF采样门控不匹配')
            for key in ('state_sha256','common_sha256','encoder_sha256'):
                require(other['initial_state'][key]==lock['initial_state'][key], 'DiTing配对初始化不匹配：'+key)
        if stage == 'diagnostics':
            target = output/'diagnostics'/cfg['run_id']
            require(not target.exists() or not any(target.iterdir()), '诊断输出已有内容，请保留并使用新批次')
            continue
        diagnostics = read(output/'diagnostics'/cfg['run_id']/'diagnostics.json')
        require(diagnostics['status']=='PASS' and diagnostics['audit_lock_sha256']==sha256(audit_path) and
                diagnostics['a01_code_sha256']==code_sha, 'diagnostics门控失败/过期：'+cfg['run_id'])
        if stage == 'pilot':
            target = output/'pilot'/cfg['run_id']
            require(not target.exists() or not any(target.iterdir()), 'pilot输出已有内容，请保留并使用新批次')
            continue
        pilot = read(output/'pilot'/cfg['run_id']/'pilot.json')
        require(pilot['status']=='PASS' and pilot['audit_lock_sha256']==sha256(audit_path) and
                pilot['a01_code_sha256']==code_sha, 'pilot门控失败/过期：'+cfg['run_id'])
        run = output/cfg['run_id']
        if stage == 'train':
            require(not run.exists() or not any(run.iterdir()), '训练输出已有内容，请保留并使用新批次：'+str(run))
        elif stage == 'eval':
            require((run/'best.pth').is_file() and (run/'last.pth').is_file(), '训练checkpoint尚未齐备：'+str(run))
            curve = run/'training_curves.csv'
            require(curve.is_file(), '尚无完整训练曲线：'+str(curve))
            with curve.open() as stream:
                rows = list(csv.DictReader(stream))
            require(rows and int(rows[-1]['epoch'])==cfg['training']['epochs'], '12轮训练尚未完成：'+str(run))
            require(int(rows[-1]['updates'])==lock['sampling']['budget']['total_updates'], '训练更新预算未完成：'+str(run))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['environment','audit','diagnostics','pilot','train','eval','pack'])
    parser.add_argument('--indices', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        bounds = [int(value) for value in args.indices.split('-')]
        indices = list(range(bounds[0], bounds[-1]+1))
        require(1<=len(bounds)<=2 and indices and all(0<=i<=4 for i in indices), '无效索引')
        check(args.stage, indices, args.output)
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit('A01_PREFLIGHT_BLOCKED: '+str(error)+'\n没有提交作业，请检查前置阶段。')
    print('A01_PREFLIGHT_PASS stage='+args.stage+' indices='+args.indices)


if __name__ == '__main__':
    main()
