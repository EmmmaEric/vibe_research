"""Sandboxed command execution.

The one place in the harness that runs a subprocess. Properties:

* **No shell.** Commands are argument lists; there is no string interpolation
  into a shell, so nothing retrieved from the web or a model can inject one.
* **Root confinement.** The working directory must resolve inside a declared
  root.
* **Denylist.** Obviously destructive or credential-touching executables are
  refused outright, before anything runs.
* **Minimal environment.** Only an allowlist of variables is passed through, so
  API keys and tokens in the parent environment do not leak into experiment code.
* **Hard timeout and output caps.**

This is defence in depth, not a security boundary: it runs code the user pointed
it at, in-process privileges. For untrusted code, run the harness inside a
container or VM.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_BYTES = 200_000

#: Executables never run, regardless of arguments.
DENIED_EXECUTABLES = frozenset(
    {
        "rm", "rmdir", "del", "format", "mkfs", "dd", "shutdown", "reboot",
        "sudo", "su", "chmod", "chown", "curl", "wget", "ssh", "scp", "nc",
        "netcat", "git", "pip", "npm", "docker", "kubectl", "systemctl",
        "powershell", "cmd", "bash", "sh", "zsh", "eval",
    }
)

#: Argument substrings that suggest an attempt to reach credentials.
DENIED_ARG_PATTERNS = (
    ".ssh", ".aws", ".config/anthropic", "id_rsa", ".netrc", ".env",
    "credentials", "ANTHROPIC_API_KEY", "/etc/passwd",
)

ENV_ALLOWLIST = (
    "PATH", "SYSTEMROOT", "WINDIR", "TEMP", "TMP", "TMPDIR", "HOME",
    "USERPROFILE", "LANG", "LC_ALL", "PYTHONPATH", "PYTHONHASHSEED",
)


class ExecutionRefused(RuntimeError):
    """Raised before execution when a command fails a safety check."""


@dataclass
class CommandResult:
    command: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration_seconds: float
    timed_out: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None


def _check(command: list[str]) -> None:
    if not command:
        raise ExecutionRefused("empty command")
    exe = Path(command[0]).name.lower()
    for suffix in (".exe", ".bat", ".cmd", ".ps1"):
        if exe.endswith(suffix):
            exe = exe[: -len(suffix)]
    # The current interpreter is allowed even though "python" would otherwise be
    # unremarkable; a shell is not, because it reintroduces string injection.
    if exe in DENIED_EXECUTABLES and command[0] != sys.executable:
        raise ExecutionRefused(
            f"executable '{exe}' is on the denylist. Experiments must not perform "
            "destructive, network or package-management actions."
        )
    blob = " ".join(command).lower()
    for pat in DENIED_ARG_PATTERNS:
        if pat.lower() in blob:
            raise ExecutionRefused(
                f"command references '{pat}', which looks like credential or system "
                "material; refused."
            )


def build_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    """Minimal environment: allowlisted variables plus explicit extras."""
    env = {k: v for k, v in os.environ.items() if k in ENV_ALLOWLIST}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra:
        env.update(extra)
    return env


def run_command(
    command: list[str],
    *,
    cwd: str | Path,
    root: str | Path | None = None,
    timeout_seconds: float = 300.0,
    extra_env: dict[str, str] | None = None,
) -> CommandResult:
    """Run ``command`` with no shell, a minimal environment and a hard timeout."""
    _check(command)
    cwd_path = Path(cwd).resolve()
    if root is not None:
        root_path = Path(root).resolve()
        try:
            cwd_path.relative_to(root_path)
        except ValueError as exc:
            raise ExecutionRefused(
                f"working directory {cwd_path} is outside the permitted root {root_path}"
            ) from exc
    if not cwd_path.is_dir():
        raise ExecutionRefused(f"working directory {cwd_path} does not exist")

    started = time.monotonic()
    try:
        proc = subprocess.run(
            command,
            cwd=str(cwd_path),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=build_env(extra_env),
            shell=False,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return CommandResult(
            command=list(command),
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=round(time.monotonic() - started, 4),
            timed_out=True,
            error=f"exceeded the {timeout_seconds}s time limit and was terminated",
        )
    except OSError as exc:
        return CommandResult(
            command=list(command),
            exit_code=None,
            stdout="",
            stderr="",
            duration_seconds=round(time.monotonic() - started, 4),
            error=f"could not start process: {exc}",
        )

    return CommandResult(
        command=list(command),
        exit_code=proc.returncode,
        stdout=proc.stdout[:MAX_OUTPUT_BYTES],
        stderr=proc.stderr[:MAX_OUTPUT_BYTES],
        duration_seconds=round(time.monotonic() - started, 4),
    )
