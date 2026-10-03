"""Torch-free configuration inheritance for login nodes and shell launchers.

Keep the JSON merge/environment contract identical to train_light.load_config_file.
YAML is imported only when a YAML file is actually encountered.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import re
import tempfile


_ENV_PATTERN = re.compile(r'\$(?:[A-Za-z_][A-Za-z0-9_]*|\{[A-Za-z_][A-Za-z0-9_]*\})')


def _deep_merge(base, override):
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _expand(value):
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, str):
        expanded = os.path.expanduser(os.path.expandvars(value))
        unresolved = _ENV_PATTERN.search(expanded)
        if unresolved:
            raise ValueError(
                f'Unresolved environment variable {unresolved.group(0)!r} '
                f'in config value {value!r}.'
            )
        return expanded
    return value


def load_config_file(path, _stack=None, _expand_environment=True):
    path = os.path.abspath(os.path.expanduser(os.path.expandvars(str(path))))
    stack = [] if _stack is None else list(_stack)
    if path in stack:
        raise ValueError(f'Config inheritance cycle: {stack + [path]}')
    stack.append(path)
    with open(path) as stream:
        if path.endswith(('.yml', '.yaml')):
            import yaml
            config = yaml.safe_load(stream)
        else:
            config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'Config root must be an object: {path}')
    parent = config.pop('extends', None)
    if parent:
        parent = os.path.expanduser(os.path.expandvars(str(parent)))
        if not os.path.isabs(parent):
            parent = os.path.join(os.path.dirname(path), parent)
        config = _deep_merge(
            load_config_file(parent, _stack=stack, _expand_environment=False), config
        )
    return _expand(config) if _expand_environment else config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config')
    parser.add_argument('field', choices=('weight-path', 'single-station-enabled', 'resolved-json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--metadata-cache-dir')
    args = parser.parse_args()
    config = load_config_file(args.config)
    if args.field == 'weight-path':
        value = config['training_params']['weight_path']
        if not isinstance(value, str) or not value.strip():
            raise ValueError('training_params.weight_path must be a non-empty string.')
        print(value)
    elif args.field == 'single-station-enabled':
        enabled = config['training_params'].get('single_station_pretrain', {}).get('enabled', False)
        print('1' if enabled else '0')
    else:
        if args.metadata_cache_dir:
            config['training_params']['metadata_cache_dir'] = _expand(args.metadata_cache_dir)
        text = json.dumps(config, indent=2) + '\n'
        if args.output is None:
            print(text, end='')
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix='.' + args.output.name + '.', dir=str(args.output.parent))
            try:
                with os.fdopen(fd, 'w') as stream:
                    stream.write(text)
                os.replace(tmp, str(args.output))
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)


if __name__ == '__main__':
    main()
