"""Verify the fixed V01 weight identity without ever writing a checkpoint."""
import sys
from pathlib import Path

from tools.v01_validation_contract import CHECKPOINTS, sha256


def main():
    arm, name = sys.argv[1:]
    actual = sha256(Path(name))
    if actual != CHECKPOINTS[arm][1]:
        raise SystemExit('Fixed V01 checkpoint SHA256 mismatch: ' + arm)
    print('[OK] fixed epoch8 checkpoint byte identity verified:', arm, actual)


if __name__ == '__main__':
    main()
