"""Layered TOML configuration (spec section 14).

Precedence, lowest to highest::

    config/global_config/*.toml  ->  config/tasks/<task>.toml  ->  CLI flags

Nested tables are deep-merged; arrays are *replaced* wholesale rather than
concatenated, because appending silently to an inherited list is the kind of
behaviour nobody predicts correctly. That choice is documented here and in the
shipped config files.

Two properties worth stating:

* **Unknown keys are an error.** A silently ignored config entry is worse than
  a failure: it looks applied and is not.
* **Secrets never enter config.** Keys come from the environment (or
  ``local.env``) only, and :func:`redact` strips anything secret-shaped before
  the effective-config snapshot is written. Reports and logs get the redacted
  form.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

#: Sections a global config file may define.
KNOWN_SECTIONS = frozenset(
    {"agent", "judge", "target", "budget", "strategy", "evaluation", "backends",
     "security", "search", "experiment"}
)

#: Flat option keys the CLI understands, per section.
KNOWN_KEYS: dict[str, frozenset[str]] = {
    "agent": frozenset({"backend", "provider", "model", "effort", "timeout_s", "mode"}),
    "judge": frozenset({"provider", "mode", "model"}),
    "target": frozenset({"maturity_level"}),
    "budget": frozenset(
        {
            "max_iterations", "max_wall_time_minutes", "max_cost_usd",
            "max_parallel_agents", "max_experiments", "max_repairs",
            "max_no_gain", "max_tokens",
        }
    ),
    "strategy": frozenset({"mode", "tradeoff_threshold"}),
    "evaluation": frozenset(
        {"mode", "weighting", "compare_top_k", "pairwise_reverse_order",
         "require_independent_validation"}
    ),
    "search": frozenset({"mode", "fixture_dir", "local_root", "human_evidence"}),
    "experiment": frozenset({"adapter", "project_dir", "test_target", "timeout_s"}),
    "backends": frozenset({"base_url", "api_key_env"}),
    "security": frozenset({"allow_external_write", "require_approval", "offline"}),
}

#: Environment-variable and key names whose values must never be printed.
_SECRET_HINTS = ("key", "token", "secret", "password", "credential", "auth")


class ConfigError(ValueError):
    pass


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open("rb") as fh:
            return tomllib.load(fh)
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"could not parse {path}: {exc}") from exc


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base``.

    Tables merge recursively; scalars and arrays are replaced. Replacement for
    arrays is deliberate — see the module docstring.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def validate(config: dict[str, Any], *, source: str) -> None:
    """Reject unknown sections and keys rather than ignoring them."""
    unknown_sections = set(config) - KNOWN_SECTIONS
    if unknown_sections:
        raise ConfigError(
            f"{source}: unknown section(s) {sorted(unknown_sections)}; known "
            f"sections are {sorted(KNOWN_SECTIONS)}"
        )
    for section, values in config.items():
        if not isinstance(values, dict):
            raise ConfigError(f"{source}: section '{section}' must be a table")
        allowed = KNOWN_KEYS.get(section, frozenset())
        # Sub-tables (e.g. [agent.teacher]) are allowed through but must be
        # consumed by something; they are not silently mapped to CLI options.
        flat = {k for k, v in values.items() if not isinstance(v, dict)}
        unknown = flat - allowed
        if unknown:
            raise ConfigError(
                f"{source}: unknown key(s) {sorted(unknown)} in [{section}]; known "
                f"keys are {sorted(allowed)}"
            )


def load_layered(
    *,
    global_dir: str | Path | None = None,
    task_file: str | Path | None = None,
) -> dict[str, Any]:
    """Merge every global config file, then the task overlay."""
    merged: dict[str, Any] = {}
    if global_dir:
        gdir = Path(global_dir)
        if not gdir.is_dir():
            raise ConfigError(f"global config directory not found: {gdir}")
        for path in sorted(gdir.glob("*.toml")):
            data = _load(path)
            validate(data, source=str(path))
            merged = deep_merge(merged, data)
    if task_file:
        path = Path(task_file)
        data = _load(path)
        validate(data, source=str(path))
        merged = deep_merge(merged, data)
    return merged


#: config path -> CLI argparse destination.
_CLI_MAP: dict[tuple[str, str], str] = {
    ("agent", "provider"): "provider",
    ("agent", "model"): "model",
    ("agent", "effort"): "effort",
    ("judge", "mode"): "judge",
    ("target", "maturity_level"): "depth",
    ("budget", "max_iterations"): "max_iterations",
    ("budget", "max_cost_usd"): "max_cost_usd",
    ("budget", "max_tokens"): "max_tokens",
    ("budget", "max_repairs"): "max_repairs",
    ("budget", "max_no_gain"): "max_no_gain",
    ("strategy", "mode"): "mode",
    ("search", "mode"): "search",
    ("search", "fixture_dir"): "fixture_dir",
    ("search", "local_root"): "local_root",
    ("search", "human_evidence"): "human_evidence",
    ("experiment", "adapter"): "adapter",
    ("experiment", "project_dir"): "project_dir",
    ("experiment", "test_target"): "test_target",
    ("security", "offline"): "offline",
}


def to_cli_defaults(config: dict[str, Any]) -> dict[str, Any]:
    """Project a merged config onto CLI option defaults."""
    out: dict[str, Any] = {}
    for (section, key), dest in _CLI_MAP.items():
        if section in config and isinstance(config[section], dict):
            if key in config[section]:
                out[dest] = config[section][key]
    minutes = config.get("budget", {}).get("max_wall_time_minutes")
    if minutes is not None:
        out["max_seconds"] = float(minutes) * 60.0
    return out


def apply_defaults(args, defaults: dict[str, Any], explicit: set[str]) -> None:
    """Apply config values only where the user did not pass a flag."""
    for key, value in defaults.items():
        if key in explicit:
            continue
        if hasattr(args, key):
            setattr(args, key, value)


# --------------------------------------------------------------------------
# Redaction
# --------------------------------------------------------------------------


def _is_secret(name: str) -> bool:
    low = name.lower()
    return any(hint in low for hint in _SECRET_HINTS)


def redact(value: Any, *, key: str = "") -> Any:
    """Recursively replace secret-shaped values with a marker.

    Keys are matched by *name*, so a new secret field is redacted by default
    rather than needing to be added to a list. ``api_key_env`` holds the *name*
    of an environment variable, not a secret, so names ending in ``_env`` pass
    through — that is the one carve-out, and it is narrow on purpose.
    """
    if isinstance(value, dict):
        return {k: redact(v, key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, key=key) for v in value]
    if key and _is_secret(key) and not key.lower().endswith("_env"):
        return "***REDACTED***"
    return value


def effective_config(
    config: dict[str, Any], *, cli_overrides: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Redacted snapshot of what the run actually used (spec section 14)."""
    env_names = sorted(
        name
        for name in os.environ
        if _is_secret(name) and ("ANTHROPIC" in name or "OPENAI" in name)
    )
    return {
        "config": redact(config),
        "cli_overrides": redact(cli_overrides or {}),
        "credentials_present": {name: True for name in env_names},
        "note": (
            "Secret values are never recorded. Credential environment variables "
            "are listed by name only, to show whether a key was available without "
            "disclosing it."
        ),
    }
