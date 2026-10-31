"""Confined local file access.

Reads are restricted to a declared root, with symlinks resolved before the
containment check, so a path assembled from retrieved text or model output
cannot escape the root. Nothing here writes or deletes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXTENSIONS = (
    ".md", ".txt", ".rst", ".py", ".json", ".yaml", ".yml", ".toml", ".csv", ".cfg",
)
MAX_READ_BYTES = 1_000_000


class PathRefused(RuntimeError):
    """Raised when a path resolves outside the permitted root."""


@dataclass
class FileRecord:
    path: str
    size_bytes: int
    text: str
    truncated: bool


class FileStore:
    """Read-only, root-confined view of a directory."""

    def __init__(
        self,
        root: str | Path,
        *,
        extensions: tuple[str, ...] = DEFAULT_EXTENSIONS,
        max_read_bytes: int = MAX_READ_BYTES,
    ) -> None:
        self.root = Path(root).resolve()
        self.extensions = extensions
        self.max_read_bytes = max_read_bytes

    def _resolve(self, relative: str | Path) -> Path:
        candidate = (self.root / relative).resolve()
        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise PathRefused(
                f"{candidate} is outside the permitted root {self.root}"
            ) from exc
        return candidate

    def list_files(self, *, limit: int = 500) -> list[str]:
        if not self.root.is_dir():
            return []
        out: list[str] = []
        for p in sorted(self.root.rglob("*")):
            if len(out) >= limit:
                break
            if p.is_file() and p.suffix.lower() in self.extensions:
                try:
                    out.append(str(p.resolve().relative_to(self.root)))
                except ValueError:
                    continue  # symlink pointing outside the root
        return out

    def read(self, relative: str | Path) -> FileRecord:
        path = self._resolve(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        data = path.read_bytes()[: self.max_read_bytes]
        return FileRecord(
            path=str(path),
            size_bytes=size,
            text=data.decode("utf-8", errors="replace"),
            truncated=size > self.max_read_bytes,
        )

    def exists(self, relative: str | Path) -> bool:
        try:
            return self._resolve(relative).exists()
        except PathRefused:
            return False
