"""Claim-once execution so a local run and a Slurm array never write the same file.

A task first creates ``<target>.lock`` with O_EXCL; if the lock or the finished
target already exists it is skipped.  Output is written to ``<target>.partial``
and renamed only on success, so a partial file never looks complete.
"""
from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def claim(target: Path):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(target.name + ".lock")
    if target.exists():
        yield None
        return
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        yield None
        return
    os.write(handle, f"{socket.gethostname()} pid={os.getpid()} slurm={os.environ.get('SLURM_JOB_ID', '-')}\n".encode())
    os.close(handle)
    partial = target.with_name(target.name + ".partial")
    try:
        yield partial
        os.replace(partial, target)
    finally:
        if lock.exists():
            lock.unlink()
