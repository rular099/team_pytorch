"""Standard-library-only project memory. Never imports experiment code or submits jobs."""
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile
from urllib.parse import quote, unquote, urlsplit

MEMORY = Path('docs/ai/memory')
STATES = {'verified_complete', 'in_progress', 'planned', 'blocked', 'superseded', 'unknown'}
BEGIN = '<!-- ai-memory:begin -->'
END = '<!-- ai-memory:end -->'
SHA = re.compile(r'^[0-9a-f]{40}$')
TRACK = re.compile(r'^[A-Z][A-Z0-9_]*$')


def git(root, *args):
    run = subprocess.run(['git', '-C', str(root), *args], stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
    if run.returncode:
        raise ValueError(run.stderr.decode(errors='replace').strip())
    return run.stdout.decode('utf-8')


def root_path(value=None):
    return Path(value or git(Path.cwd(), 'rev-parse', '--show-toplevel').strip()).resolve()


def safe_path(root, value):
    path = PurePosixPath(value)
    if not value or path.is_absolute() or '..' in path.parts or '\\' in value or ':' in value:
        raise ValueError('Unsafe repository path: ' + value)
    resolved = (root / value).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError('Path escapes repository: ' + value)
    return resolved


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + '\n'


def digest(value):
    return hashlib.sha256(json_text(value).encode()).hexdigest()


def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def timestamp(value):
    time = dt.datetime.fromisoformat(value.replace('Z', '+00:00'))
    if time.tzinfo is None:
        raise ValueError('Timestamp must include timezone: ' + value)
    return time


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError('A track record must be an object')
    required = ('track_id', 'branch', 'commit_sha', 'last_verified_at', 'evidence_as_of',
                'current_state', 'completion_scope', 'summary', 'verified_results',
                'pending_tasks', 'limitations', 'evidence_paths', 'decisions', 'experiments')
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError('Missing record fields: ' + ', '.join(missing))
    if not TRACK.fullmatch(record['track_id']) or not SHA.fullmatch(record['commit_sha']):
        raise ValueError('Invalid track ID or full commit SHA')
    if not record['branch'] or not record['completion_scope'] or not record['summary']:
        raise ValueError('Branch, completion scope and summary must be nonempty')
    if record['current_state'] not in STATES:
        raise ValueError('Invalid current_state: ' + record['current_state'])
    timestamp(record['last_verified_at'])
    dt.date.fromisoformat(record['evidence_as_of'])
    for key in ('verified_results', 'pending_tasks', 'limitations', 'evidence_paths', 'decisions', 'experiments'):
        if not isinstance(record[key], list):
            raise ValueError(key + ' must be a list')
    ids = set()
    for item in record['evidence_paths']:
        if not {'evidence_id', 'commit_sha', 'path', 'kind'} <= item.keys():
            raise ValueError('Incomplete evidence locator')
        if item['evidence_id'] in ids or not SHA.fullmatch(item['commit_sha']):
            raise ValueError('Duplicate evidence ID or invalid evidence commit')
        ids.add(item['evidence_id'])
        safe_path(Path('/tmp/memory-schema-root'), item['path'])
    for result in record['verified_results']:
        if not result.get('text') or not result.get('evidence_ids'):
            raise ValueError('Every verified result needs text and evidence_ids')
        if not set(result['evidence_ids']) <= ids:
            raise ValueError('Unknown result evidence ID')
    for group, required_keys in (
        ('decisions', {'decision_id', 'date', 'problem', 'options', 'decision', 'evidence_ids',
                       'applies_to', 'superseded_by', 'history_status'}),
        ('experiments', {'experiment_id', 'scientific_question', 'dataset_protocol', 'result_nature',
                         'review_status', 'relationship', 'evidence_ids', 'state'})):
        key = 'decision_id' if group == 'decisions' else 'experiment_id'
        values = [item.get(key) for item in record[group]]
        if len(set(values)) != len(values):
            raise ValueError('Duplicate ' + key)
        for item in record[group]:
            if not required_keys <= item.keys() or not item['evidence_ids']:
                raise ValueError('Incomplete ' + group + ' entry')
            if not set(item['evidence_ids']) <= ids:
                raise ValueError('Unknown evidence in ' + group)
            if group == 'experiments' and item['state'] not in STATES:
                raise ValueError('Invalid experiment state')
    if 'handoff' in record:
        validate_handoff(record)
    return record


def validate_handoff(record):
    """A next task is a proposal, never a stored execution authorization."""
    h = record.get('handoff')
    if not isinstance(h, dict):
        raise ValueError('Missing persistent handoff: ' + record['track_id'])
    required = {'schema_version', 'repository', 'goal', 'default_action', 'next_task', 'last_session'}
    if not required <= h.keys() or h['schema_version'] != 1:
        raise ValueError('Incomplete/versioned handoff fields')
    if h['repository'] != 'rular099/team_pytorch' or h['default_action'] != 'read_verify_report':
        raise ValueError('Handoff repository/default action must be read-only')
    if not isinstance(h['goal'], str) or not h['goal'].strip():
        raise ValueError('Missing experiment goal')
    next_task = h['next_task']
    needed = {'task_id', 'objective', 'kind', 'execution_authorization', 'read_steps', 'acceptance_checks'}
    if not isinstance(next_task, dict) or not needed <= next_task.keys():
        raise ValueError('Incomplete next_task fields')
    if next_task['execution_authorization'] != 'not_granted_by_this_file':
        raise ValueError('NEXT_TASK cannot grant execution authorization')
    if next_task['kind'] not in {'read_only_review', 'evidence_collection', 'implementation_proposal'}:
        raise ValueError('Invalid next task kind')
    if not next_task['task_id'].startswith(record['track_id'] + '-') or not next_task['objective']:
        raise ValueError('Next task belongs to another track or has no objective')
    for field in ('read_steps', 'acceptance_checks'):
        if not isinstance(next_task[field], list) or not next_task[field] or not all(isinstance(x, str) and x.strip() for x in next_task[field]):
            raise ValueError('Next task requires nonempty ' + field)
    session = h['last_session']
    needed = {'task_id', 'result_commit', 'result_branch', 'scope', 'evidence_ids', 'unfinished', 'resume_steps'}
    if not isinstance(session, dict) or not needed <= session.keys():
        raise ValueError('Incomplete last_session fields')
    if not session['task_id'] or not SHA.fullmatch(session['result_commit']):
        raise ValueError('Last session requires a full result commit')
    if session['scope'] not in {'scientific_result', 'evidence_review', 'handoff_infrastructure'}:
        raise ValueError('Invalid session scope')
    if not session['result_branch'] or session['result_branch'].startswith('-'):
        raise ValueError('Invalid last result branch')
    ids = {x['evidence_id'] for x in record['evidence_paths']}
    if not session['evidence_ids'] or not set(session['evidence_ids']) <= ids:
        raise ValueError('Last session evidence IDs are missing or unknown')
    for field in ('unfinished', 'resume_steps'):
        if not isinstance(session[field], list) or not session[field] or not all(isinstance(x, str) and x.strip() for x in session[field]):
            raise ValueError('Last session requires nonempty ' + field)
    return h


def preserve_history(previous, candidate):
    if timestamp(candidate['last_verified_at']) < timestamp(previous['last_verified_at']):
        raise ValueError('Older verification cannot replace a newer track snapshot')
    for collection, key in (('decisions', 'decision_id'), ('experiments', 'experiment_id')):
        before = {item[key] for item in previous[collection]}
        after = {item[key] for item in candidate[collection]}
        if not before <= after:
            raise ValueError('Preserve historical IDs; mark superseded instead of deleting: ' + collection)


def handoff_registry(root):
    registry = load(root / MEMORY / 'registry.json')
    config = registry.get('handoff', {})
    if not config:
        return config  # Existing non-FE snapshots keep their original schema.
    if (config.get('schema_version') != 1 or not config.get('coordinator_branches') or
            config.get('entry_branch') not in config['coordinator_branches']):
        raise ValueError('Invalid handoff registry')
    if registry.get('repository') != 'rular099/team_pytorch':
        raise ValueError('Registry repository mismatch')
    if set(config.get('tracks', {})) != {'FE01', 'FE02'}:
        raise ValueError('Persistent handoff requires distinct FE01 and FE02 routes')
    branches = list(config['coordinator_branches'])
    for track, route in config['tracks'].items():
        if route.get('directory') != str(MEMORY / 'tracks' / track):
            raise ValueError('Track directory violates ownership: ' + track)
        safe_path(root, route['directory'])
        if not isinstance(route.get('branch'), str):
            raise ValueError('Missing route branch: ' + track)
        branches.append(route['branch'])
    if len(branches) != len(set(branches)):
        raise ValueError('Coordinator and experiment branches must be distinct')
    for branch in branches:
        git(root, 'check-ref-format', 'refs/heads/' + branch)
    return config


def handoff_provenance(root, record, config):
    h = validate_handoff(record)
    route = config['tracks'][record['track_id']]
    if record['branch'] != route['branch']:
        raise ValueError('Record branch does not own track: ' + record['track_id'])
    session = h['last_session']
    git(root, 'check-ref-format', 'refs/heads/' + session['result_branch'])
    allowed = {record['branch']}
    if session['scope'] == 'handoff_infrastructure':
        allowed.update(config['coordinator_branches'])
    if session['result_branch'] not in allowed:
        raise ValueError('Last session result branch belongs to another track')
    for sha in (record['commit_sha'], session['result_commit']):
        git(root, 'cat-file', '-e', sha + '^{commit}')
    for branch, sha in [(record['branch'], record['commit_sha']),
                        (session['result_branch'], session['result_commit'])]:
        refs = ['refs/heads/' + branch, 'refs/remotes/origin/' + branch]
        available = []
        for ref in refs:
            try:
                available.append(git(root, 'rev-parse', '--verify', ref + '^{commit}').strip())
            except ValueError:
                continue
        if available and not any(is_ancestor(root, sha, head) for head in available):
            raise ValueError('Recorded commit is not on its declared branch: ' + branch)
    return h


def is_ancestor(root, older, newer):
    try:
        git(root, 'merge-base', '--is-ancestor', older, newer)
        return True
    except ValueError:
        return False


def evidence_link(evidence):
    path = quote(evidence['path'], safe='/')
    return '[{}](https://github.com/rular099/team_pytorch/blob/{}/{})'.format(
        evidence['path'], evidence['commit_sha'], path)


def bullets(items):
    return '\n'.join('- ' + str(item) for item in items) or '- 无记录。'


def references(record, ids):
    by_id = {e['evidence_id']: e for e in record['evidence_paths']}
    return '；'.join(evidence_link(by_id[i]) for i in ids)


def decision_body(record):
    sections = []
    for d in record['decisions']:
        sections.append('### {decision_id} · {date}\n\n'
                        '- 实验线：{track}\n- 背景：{problem}\n- 讨论选项：{options}\n'
                        '- 决定：{decision}\n- 适用范围：{applies_to}\n'
                        '- 后续推翻：{superseded}\n- 历史依据：{history_status}\n'
                        '- 来源：{sources}\n'.format(
                            **d, track=record['track_id'], superseded=d['superseded_by'] or '未见记录',
                            sources=references(record, d['evidence_ids'])))
    return '\n'.join(sections) or '未登记；不代表没有历史决策。\n'


def experiment_body(record):
    sections = []
    for e in record['experiments']:
        sections.append('### {experiment_id}\n\n'
                        '- 科学问题：{scientific_question}\n- branch：`{branch}`\n'
                        '- commit：`{commit}`\n- 状态：`{state}`\n- 数据/协议：{dataset_protocol}\n'
                        '- 结果性质：{result_nature}\n- 审阅：{review_status}\n'
                        '- 关系：{relationship}\n- 原始入口：{sources}\n'.format(
                            **e, branch=record['branch'], commit=record['commit_sha'],
                            sources=references(record, e['evidence_ids'])))
    return '\n'.join(sections) or '未登记；不代表没有历史实验。\n'


def track_views(record):
    r = record
    meta = ('track_id: `{track_id}`\n\nbranch: `{branch}`\n\ncommit_sha: `{commit_sha}`\n\n'
            'last_verified_at: `{last_verified_at}`\n\ncurrent_state: `{current_state}`\n\n'
            '原证据日期：{evidence_as_of}；本次只核验可见来源，不代表重新跑过实验。\n\n'
            '完成范围：{completion_scope}\n\n{summary}\n').format(**r)
    h = r.get('handoff')
    if h:
        meta += ('\n实验目标：' + h['goal'] + '\n\n默认动作：只读、核验和报告；待办不授予执行权限。\n\n'
                 '最近工作 result_commit：`' + h['last_session']['result_commit'] + '`；branch：`' +
                 h['last_session']['result_branch'] + '`；scope：`' + h['last_session']['scope'] + '`。\n')
    results = bullets([x['text'] + ' 来源：' + references(r, x['evidence_ids']) for x in r['verified_results']])
    status = meta + '\n### verified_results\n\n' + results
    status += '\n\n### pending_tasks\n\n' + bullets(r['pending_tasks'])
    status += '\n\n### limitations\n\n' + bullets(r['limitations'])
    status += '\n\n### evidence_paths\n\n' + bullets([evidence_link(e) for e in r['evidence_paths']]) + '\n'
    handoff = meta + '\n已核验：\n\n' + results + '\n\n下一步：\n\n' + bullets(r['pending_tasks'])
    handoff += '\n\n证据范围：\n\n' + bullets(r['limitations']) + '\n'
    views = {'STATUS.md': status, 'DECISIONS.md': decision_body(r),
             'EXPERIMENT_INDEX.md': experiment_body(r), 'CHAT_HANDOFF.md': handoff}
    if h:
        task, session = h['next_task'], h['last_session']
        resume = '\n\n### 未完成事项\n\n' + bullets(session['unfinished'])
        resume += '\n\n### 恢复步骤（先读后报告）\n\n' + bullets(session['resume_steps']) + '\n'
        views['STATUS.md'] += resume
        views['CHAT_HANDOFF.md'] += resume
        views['NEXT_TASK.md'] = (meta + '\n### 下一步提案\n\n任务：`' + task['task_id'] +
            '`；类型：`' + task['kind'] + '`。\n\n' + task['objective'] +
            '\n\n**execution_authorization: not_granted_by_this_file**\n\n'
            '默认只核验并报告当前状态；是否实施取决于当前用户任务。不得根据此文件自动训练、'
            '提交HPC作业、续跑或修改重要实验协议。\n\n### 可先读取的材料\n\n' +
            bullets(task['read_steps']) + '\n\n### 若获授权后的验收条件\n\n' +
            bullets(task['acceptance_checks']) + resume)
    return views


def generated(old, title, body):
    block = BEGIN + '\n' + body.rstrip() + '\n' + END
    if not old:
        return '# ' + title + '\n\n' + block + '\n\n<!-- 人工补充写在生成区外；record.json 是结构化事实源。 -->\n'
    if old.count(BEGIN) != 1 or old.count(END) != 1 or old.index(BEGIN) > old.index(END):
        raise ValueError('Missing, duplicated or reversed generation markers: ' + title)
    return old.split(BEGIN)[0] + block + old.split(END)[1]


def global_views(snapshot):
    meta = snapshot['metadata']
    records = snapshot['tracks']
    header = ('更新时间：{generated_at}\n\n生成所依据的协调分支 commit：`{generated_from_commit}`\n\n'
              '协调分支：`{coordinator_branch}`\n\n'
              '此 SHA 是生成输入，不是整个科研项目的单一 HEAD；各线见下表。\n').format(**meta)
    table = '| track_id | branch | commit_sha | last_verified_at | current_state |\n|---|---|---|---|---|\n'
    table += '\n'.join('| {track_id} | `{branch}` | `{commit_sha}` | {last_verified_at} | `{current_state}` |'.format(**r) for r in records)
    status = header + '\n' + table + '\n'
    handoff = ('科研目标：利用截至当前时刻可用的多台站三分量波形及输入/查询台站位置，'
               '预测查询台站最终 PGA 的概率分布；同时关注非输入目标、空间动态范围、校准与实时可用性。'
               '报告坐标 log10(m/s²)。保留 RT55 checkpoint 加载与推理兼容，不用 held-out test 调参。\n\n'
               '当前任务是项目记忆与并行隔离初始化。全局由协调者维护，各科研实例只写自己的 tracks/<ID>/。'
               '公共改动经审阅后再引入工作分支，本轮没有开始新训练或提交超算。\n\n')
    handoff += '状态核验时间：' + meta['generated_at'] + '。精确 branch/commit 见 [CURRENT_STATUS.md](CURRENT_STATUS.md)。\n\n'
    for r in records:
        track = r['track_id']
        status += '\n### ' + track + '\n\n' + r['completion_scope'] + '\n\n'
        status += '**verified_results**\n\n' + bullets([x['text'] + ' 来源：' + references(r, x['evidence_ids']) for x in r['verified_results']])
        status += '\n\n**pending_tasks**\n\n' + bullets(r['pending_tasks'])
        status += '\n\n**evidence_paths**\n\n' + bullets([evidence_link(e) for e in r['evidence_paths']]) + '\n'
        status += '\n同步：' + snapshot['sync'][track]['state'] + '；来源 `' + snapshot['sync'][track]['source'] + '`。\n'
        handoff += '- **' + track + '** (`' + r['current_state'] + '`)：' + r['summary']
        if r['verified_results']:
            handoff += ' ' + r['verified_results'][0]['text']
        handoff += ' [本线来源](tracks/' + track + '/STATUS.md)。\n'
    handoff += ('\n尚未解决：任意非输入查询的空间辨识、单输入空间场压缩、normal/random收益保持与概率校准。'
                '单seed和反复使用validation限制结论；原始权重/数据/调度退出码缺失必须分别保留，不能用缺同步推断未完成。\n\n'
                '逐线下一步以各线当前 NEXT_TASK（若已部署）及 pending_tasks 为准；'
                '旧日期的优先级不覆盖新证据。所有待办均不自动触发计算。\n\n'
                '追溯方法：用 [EXPERIMENT_INDEX.md](EXPERIMENT_INDEX.md) 找实验ID，用 '
                '[DECISION_LOG.md](DECISION_LOG.md) 找决策ID，再打开条目中的固定commit链接。'
                '[PROJECT_MEMORY.md](PROJECT_MEMORY.md) 保存长期规则；每线 STATUS/DECISIONS/EXPERIMENT_INDEX/CHAT_HANDOFF '
                '提供详细入口。Git之外的历史聊天未全部读取，缺原讨论的原因标 needs_history_review；'
                '使用 [HISTORY_IMPORT_TEMPLATE.md](HISTORY_IMPORT_TEMPLATE.md) 补录。旧报告保留其日期语境，不能当当前全局状态。\n')
    return {'CURRENT_STATUS.md': status, 'CHAT_HANDOFF.md': handoff,
            'DECISION_LOG.md': header + '\n' + '\n'.join('## ' + r['track_id'] + '\n\n' + decision_body(r) for r in records),
            'EXPERIMENT_INDEX.md': header + '\n' + '\n'.join('## ' + r['track_id'] + '\n\n' + experiment_body(r) for r in records)}


def atomic_write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as handle:
        name = handle.name
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextlib.contextmanager
def lock(root, name):
    path = Path(git(root, 'rev-parse', '--git-common-dir').strip())
    if not path.is_absolute():
        path = root / path
    path = path / ('ai-memory-' + name + '.lock')
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, json_text({'pid': os.getpid(), 'created_at': utcnow()}).encode())
        os.close(descriptor)
        yield
    finally:
        path.unlink()


def evidence_available(root, evidence, remote):
    sha, path = evidence['commit_sha'], evidence['path']
    try:
        git(root, 'cat-file', '-e', sha + ':' + path)
        return 'git_object'
    except ValueError:
        for item in remote.get('remote_files', []):
            if (item['commit_sha'], item['path']) == (sha, path):
                raw = safe_path(root, item['snapshot_path']).read_bytes()
                blob = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
                if blob != item['blob_sha']:
                    raise ValueError('Remote snapshot blob hash mismatch: ' + path)
                return 'connector_snapshot'
        raise ValueError('Cannot verify evidence at commit: ' + sha + ':' + path)


def resource_errors(manifest):
    """Read-only namespace audit; paths are expanded and symlinks resolved, never created."""
    writes, reads, errors = [], [], []
    for record in manifest:
        for kind, dest in (('write_paths', writes), ('readonly_paths', reads)):
            for value in record.get(kind, []):
                expanded = os.path.expandvars(value)
                if '$' in expanded or not Path(expanded).is_absolute():
                    errors.append('Unresolved or relative resource path: ' + value)
                    continue
                dest.append((record['track_id'], Path(expanded).resolve()))
    for i, (owner, path) in enumerate(writes):
        for other, candidate in writes[i+1:] + reads:
            if owner != other and (path == candidate or path in candidate.parents or candidate in path.parents):
                errors.append('Cross-track resource overlap: {}:{} and {}:{}'.format(owner, path, other, candidate))
    return errors


def merge_risk(root, left, right):
    left_sha = git(root, 'rev-parse', '--verify', left + '^{commit}').strip()
    right_sha = git(root, 'rev-parse', '--verify', right + '^{commit}').strip()
    base = git(root, 'merge-base', left_sha, right_sha).strip()
    changed = [set(git(root, 'diff', '--name-only', base, sha).splitlines()) for sha in (left_sha, right_sha)]
    common = sorted(changed[0] & changed[1])
    architecture = sorted(p for p in changed[0] | changed[1]
                          if p == 'gemini_models.py' or p.endswith('/model.py'))
    return {'left': left_sha, 'right': right_sha, 'merge_base': base, 'shared_changed_files': common,
            'architecture_changed_files': architecture,
            'requires_review': bool(common or architecture), 'requires_architecture_review': bool(architecture),
            'action': 'review compatibility and run relevant regression before any merge; no merge performed'}
