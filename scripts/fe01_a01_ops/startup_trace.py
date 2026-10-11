"""Per-rank startup evidence, independent of model state and RNG streams."""
import datetime
import faulthandler
import json
import os
from pathlib import Path
import resource
import socket
import sys
import time


class StartupTrace:
    def __init__(self, operation, index, rank):
        self.directory = Path(operation)/'startup'
        self.directory.mkdir(exist_ok=True)
        prefix = f'index{index}_rank{rank}'
        self.progress = (self.directory/(prefix+'.jsonl')).open('x')
        self.stack = (self.directory/(prefix+'.stack.log')).open('x')
        self.index, self.rank = index, rank
        self.started = time.monotonic()

    def __enter__(self):
        faulthandler.dump_traceback_later(120, repeat=True, file=self.stack)
        self.stage('process.begin')
        return self

    def stage(self, name):
        record = dict(stage=name, index=self.index, rank=self.rank, pid=os.getpid(),
                      host=socket.gethostname(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      elapsed_seconds=time.monotonic()-self.started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        line = json.dumps(record)
        self.progress.write(line+'\n');self.progress.flush()
        print('A01_STARTUP_STAGE '+line, file=sys.stderr, flush=True)
        if name == 'first_update.complete':
            faulthandler.cancel_dump_traceback_later()

    def __exit__(self, kind, error, traceback):
        if kind is not None:
            self.stage('process.error:'+kind.__name__+':'+str(error))
            faulthandler.dump_traceback(file=self.stack, all_threads=True)
        else:
            self.stage('process.complete')
        faulthandler.cancel_dump_traceback_later()
        self.progress.close();self.stack.close()
