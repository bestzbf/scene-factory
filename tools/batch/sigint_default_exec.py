"""exec a command with SIGINT restored to its default disposition.

A command started with `&` from a non-interactive shell inherits SIGINT=SIG_IGN, and
Python then never installs its KeyboardInterrupt handler, so `kill -INT` is ignored.
Restoring SIG_DFL before exec lets scene-factory's own graceful stop work:
`kill -INT <pid>` -> no new scenes start, in-flight model calls and renders finish,
state is checkpointed, and the same command (with --resume-batch if needed) resumes.

    python3 -I sigint_default_exec.py COMMAND [ARGS...]
"""
import os
import signal
import sys

if len(sys.argv) < 2:
    sys.exit('usage: sigint_default_exec.py COMMAND [ARGS...]')
signal.signal(signal.SIGINT, signal.SIG_DFL)
os.execvp(sys.argv[1], sys.argv[1:])
