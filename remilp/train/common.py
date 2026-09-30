"""Run directories (config.json, metrics.jsonl, results.json, status.json, log.txt),
checkpoints and seeding."""

import _thread
import contextlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch_geometric import seed_everything as _seed_everything


def seed_everything(seed: int) -> None:
    _seed_everything(seed)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def read_json(path: Path):
    with Path(path).open() as f:
        return json.load(f)


def write_json(path: Path, obj) -> None:
    with Path(path).open("w") as f:
        json.dump(obj, f, indent=2)


class RunLogger:
    def __init__(self, run_dir: Path, config: dict):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config = {
            **config,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        }
        write_json(self.run_dir / "config.json", self.config)
        self.set_status("running")
        self._metrics = (self.run_dir / "metrics.jsonl").open("a")

    def set_status(self, status: str) -> None:
        write_json(
            self.run_dir / "status.json",
            {"status": status, "pid": os.getpid(), "updated": time.time()},
        )

    def log(self, step: int, phase: str, **scalars) -> None:
        row = {
            "step": step,
            "phase": phase,
            "t": time.time(),
            **{k: _scalar(v) for k, v in scalars.items()},
        }
        self._metrics.write(json.dumps(row) + "\n")
        self._metrics.flush()

    def finish(self, results: dict) -> None:
        write_json(
            self.run_dir / "results.json", {k: _scalar(v) for k, v in results.items()}
        )
        self.set_status("done")
        self._metrics.close()

    def fail(self) -> None:
        self.set_status("failed")
        self._metrics.close()


def _scalar(v):
    return v.item() if torch.is_tensor(v) else v


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextlib.contextmanager
def tee_output(run_dir: Path):
    Path(run_dir).mkdir(parents=True, exist_ok=True)
    with (Path(run_dir) / "log.txt").open("a") as f:
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, f), _Tee(err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = out, err


def save_checkpoint(path: Path, state: dict) -> None:
    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def device_and_bf16(want_bf16: bool) -> tuple[str, bool]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, want_bf16 and device == "cuda" and torch.cuda.is_bf16_supported()


def autocast(enabled: bool):
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=enabled)


def compile_encoder(encoder):
    import logging

    logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
    logging.getLogger("torch._inductor").setLevel(logging.WARNING)
    import torch._inductor.config as inductor

    inductor.triton.cudagraph_skip_dynamic_graphs = True
    inductor.shape_padding = False
    torch._dynamo.config.recompile_limit = 32
    return torch.compile(encoder, mode="reduce-overhead")


def unwrap(module):
    return getattr(module, "_orig_mod", module)


STALL_TIMEOUT_S = 900.0


class ProgressWatchdog:
    """Interrupts the process when no step completes within `timeout` seconds."""

    def __init__(self, timeout: float, label: str = ""):
        self.timeout = timeout
        self.label = label
        self._last = time.monotonic()
        self._stop = threading.Event()

    def tick(self) -> None:
        self._last = time.monotonic()

    def _watch(self) -> None:
        while not self._stop.wait(min(30.0, self.timeout / 4)):
            idle = time.monotonic() - self._last
            if idle > self.timeout:
                print(
                    f"{self.label}: no progress for {idle:.0f}s, aborting", flush=True
                )
                _thread.interrupt_main()
                if not self._stop.wait(60.0):
                    os._exit(75)
                return

    def __enter__(self) -> "ProgressWatchdog":
        threading.Thread(target=self._watch, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
