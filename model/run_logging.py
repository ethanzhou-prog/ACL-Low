"""Mirror Python stdout/stderr to a timestamped log without training dependencies."""
from contextlib import contextmanager
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import platform
import re
import sys
import threading
import traceback
from zoneinfo import ZoneInfo


def _now():
    return datetime.now(ZoneInfo('Asia/Shanghai'))


class _LogWriter:
    def __init__(self, handle):
        self.handle = handle
        self.lock = threading.RLock()
        self.line_start = True

    def write(self, text, stream):
        # Turn terminal progress rewrites into readable lines in the saved log.
        text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text).replace('\r', '\n')
        with self.lock:
            for part in text.splitlines(keepends=True):
                if self.line_start:
                    self.handle.write(f'[{_now().isoformat(timespec="seconds")}] [{stream}] ')
                self.handle.write(part)
                self.line_start = part.endswith('\n')
            self.handle.flush()


class _Tee:
    def __init__(self, terminal, writer, name):
        self.terminal, self.writer, self.name = terminal, writer, name

    def write(self, text):
        self.terminal.write(text)
        self.writer.write(text, self.name)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.writer.handle.flush()

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def run_logging(args):
    folder = Path(args.log_dir) if args.log_dir else Path(args.output_dir) / 'logs'
    folder.mkdir(parents=True, exist_ok=True)
    stages = '-'.join(s for s in ('dae', 'sft', 'dpo') if not getattr(args, 'skip_' + s))
    path = folder / f'{stages}_{_now():%Y%m%d_%H%M%S_%f}_{os.getpid()}.log'
    stdout, stderr = sys.stdout, sys.stderr
    root = logging.getLogger()
    previous_level = root.level
    with path.open('x', encoding='utf-8', buffering=1) as handle:
        writer = _LogWriter(handle)
        sys.stdout = _Tee(stdout, writer, 'stdout')
        sys.stderr = _Tee(stderr, writer, 'stderr')
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter('%(levelname)s %(name)s: %(message)s'))
        root.addHandler(handler)
        root.setLevel(getattr(logging, args.log_level))
        try:
            print(f'[Run] Log file: {path.resolve()}')
            print(f'[Run] Python: {sys.version}; platform: {platform.platform()}')
            print(f'[Run] Working directory: {Path.cwd()}')
            print(f'[Run] Arguments:\n{json.dumps(vars(args), indent=2, ensure_ascii=False)}')
            yield path
        except BaseException:
            # The interpreter prints to the terminal after streams are restored.
            writer.write('\n[Run] FAILED or interrupted\n' + traceback.format_exc(), 'stderr')
            raise
        else:
            print('[Run] Completed successfully')
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            root.removeHandler(handler)
            handler.close()
            root.setLevel(previous_level)
            sys.stdout, sys.stderr = stdout, stderr


def make_training_log_callback(stage):
    # Imported only after train.py has configured CUDA visibility.
    from transformers import TrainerCallback

    class TrainingLogCallback(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            print(f'[{stage}] Training begins: optimizer_steps={state.max_steps}, '
                  f'epochs={args.num_train_epochs}, output_dir={args.output_dir}')

        def on_epoch_begin(self, args, state, control, **kwargs):
            print(f'[{stage}] Epoch begins: epoch={state.epoch}, step={state.global_step}')

        def on_epoch_end(self, args, state, control, **kwargs):
            print(f'[{stage}] Epoch ends: epoch={state.epoch}, step={state.global_step}')

        def on_log(self, args, state, control, logs=None, **kwargs):
            metrics = dict(logs or {})
            metrics.update(stage=stage, step=state.global_step, epoch=state.epoch)
            print('[Metrics] ' + json.dumps(metrics, ensure_ascii=False, default=str))

        def on_save(self, args, state, control, **kwargs):
            print(f'[{stage}] Checkpoint saved: {Path(args.output_dir) / ("checkpoint-" + str(state.global_step))}')

        def on_train_end(self, args, state, control, **kwargs):
            print(f'[{stage}] Training ends: epoch={state.epoch}, step={state.global_step}')

    return TrainingLogCallback()
