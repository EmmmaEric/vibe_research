"""Research tools: pluggable evidence retrieval.

Every tool returns :class:`SearchHit` objects carrying explicit provenance, and
the Researcher turns those into :class:`~research_harness.core.schemas.Evidence`
without upgrading their status. The distinction that matters throughout:

* ``Provenance.SOURCE``    -- a real document said this
* ``Provenance.SIMULATED`` -- fixture/demo content, NOT research evidence
* ``Provenance.HUMAN``     -- a person entered it

Retrieved text is untrusted data. It is never executed, and never treated as an
instruction to the harness -- a web page saying "ignore your constraints" is just
a string in a field.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..core.schemas import Provenance, Stance


@dataclass
class SearchHit:
    claim: str
    source_type: str
    source_uri: str | None
    provenance: Provenance
    stance: Stance = Stance.UNCERTAIN
    quality: int = 1
    applicability_conditions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    verification_status: str = "unverified"
    published_at: str | None = None
    related_metric: str | None = None


class ToolUnavailable(RuntimeError):
    """Raised when a tool cannot run (missing credentials, package, or network)."""


@runtime_checkable
class ResearchTool(Protocol):
    tool_id: str
    produces_real_evidence: bool

    def search(self, query: str, context: dict) -> list[SearchHit]: ...


# --------------------------------------------------------------------------


class FixtureSearchTool:
    """Serves hits from local JSONL fixtures. Offline and reproducible.

    Fixture records may declare ``"provenance": "source"`` only if they cite a
    genuinely verifiable document. The fixtures shipped with this repository are
    all ``"simulated"``, because inventing plausible-looking citations is exactly
    the failure mode the harness is built to avoid.
    """

    tool_id = "fixture-search"
    produces_real_evidence = False

    def __init__(self, fixture_dir: str | Path) -> None:
        self.dir = Path(fixture_dir)

    def search(self, query: str, context: dict) -> list[SearchHit]:
        if not self.dir.is_dir():
            return []
        terms = [t for t in query.lower().replace(",", " ").split() if len(t) > 3]
        hits: list[SearchHit] = []
        for path in sorted(self.dir.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("//"):
                    continue
                rec = json.loads(line)
                blob = " ".join(
                    str(rec.get(k, "")) for k in ("claim", "topic", "keywords")
                ).lower()
                if terms and not any(t in blob for t in terms):
                    continue
                prov = Provenance(rec.get("provenance", "simulated"))
                claim = rec["claim"]
                if prov is Provenance.SIMULATED and not claim.startswith("[SIMULATED]"):
                    claim = f"[SIMULATED] {claim}"
                hits.append(
                    SearchHit(
                        claim=claim,
                        source_type=rec.get("source_type", "simulated"),
                        source_uri=rec.get("source_uri"),
                        provenance=prov,
                        stance=Stance(rec.get("stance", "uncertain")),
                        quality=int(rec.get("quality", 1)),
                        applicability_conditions=list(rec.get("applicability_conditions", [])),
                        limitations=list(rec.get("limitations", []))
                        + (
                            ["Fixture content for demonstration; not a real source."]
                            if prov is Provenance.SIMULATED
                            else []
                        ),
                        verification_status=rec.get("verification_status", "unverified"),
                        published_at=rec.get("published_at"),
                        related_metric=rec.get("related_metric"),
                    )
                )
        return hits


class LocalFileSearchTool:
    """Greps a local directory. Produces REAL evidence about local artefacts.

    Restricted to a declared root; refuses to follow paths outside it, so a
    traversal in a generated query cannot read arbitrary user files.
    """

    tool_id = "local-file-search"
    produces_real_evidence = True

    def __init__(
        self,
        root: str | Path,
        *,
        extensions: tuple[str, ...] = (".md", ".txt", ".py", ".rst", ".json", ".yaml"),
        max_hits: int = 12,
        max_bytes: int = 2_000_000,
    ) -> None:
        self.root = Path(root).resolve()
        self.extensions = extensions
        self.max_hits = max_hits
        self.max_bytes = max_bytes

    def search(self, query: str, context: dict) -> list[SearchHit]:
        if not self.root.is_dir():
            return []
        terms = [t for t in query.lower().split() if len(t) > 3]
        if not terms:
            return []
        hits: list[SearchHit] = []
        for path in sorted(self.root.rglob("*")):
            if len(hits) >= self.max_hits:
                break
            if not path.is_file() or path.suffix.lower() not in self.extensions:
                continue
            try:
                resolved = path.resolve()
                resolved.relative_to(self.root)  # reject anything outside the root
                if resolved.stat().st_size > self.max_bytes:
                    continue
                text = resolved.read_text(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                continue
            low = text.lower()
            for term in terms:
                idx = low.find(term)
                if idx < 0:
                    continue
                snippet = " ".join(text[max(0, idx - 120) : idx + 240].split())
                hits.append(
                    SearchHit(
                        claim=f"Local file {path.name} mentions '{term}': ...{snippet}...",
                        source_type="local_file",
                        source_uri=str(path),
                        provenance=Provenance.SOURCE,
                        stance=Stance.UNCERTAIN,
                        quality=2,
                        applicability_conditions=["describes local artefacts only"],
                        limitations=[
                            "A local file documents intent or implementation; it is not "
                            "independent evidence that a claim is true."
                        ],
                        verification_status="source_checked",
                    )
                )
                break
        return hits


class HumanEvidenceTool:
    """Structured manual evidence intake.

    For domains where the real evidence is an interview, a bench measurement or a
    supplier quote, a human records it in a JSONL file and it enters the pipeline
    as ``Provenance.HUMAN`` -- first-class, attributable, and never conflated
    with something the harness measured itself.
    """

    tool_id = "human-evidence"
    produces_real_evidence = True

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path else None

    def search(self, query: str, context: dict) -> list[SearchHit]:
        if not self.path or not self.path.is_file():
            return []
        hits: list[SearchHit] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            rec = json.loads(line)
            hits.append(
                SearchHit(
                    claim=rec["claim"],
                    source_type=rec.get("source_type", "human"),
                    source_uri=rec.get("source_uri"),
                    provenance=Provenance.HUMAN,
                    stance=Stance(rec.get("stance", "uncertain")),
                    quality=int(rec.get("quality", 3)),
                    applicability_conditions=list(rec.get("applicability_conditions", [])),
                    limitations=list(rec.get("limitations", [])),
                    verification_status=rec.get("verification_status", "source_checked"),
                    related_metric=rec.get("related_metric"),
                )
            )
        return hits


class AnthropicWebSearchTool:
    """Real web search via the Claude server-side web_search tool.

    Requires the ``anthropic`` package and credentials. Not exercised by the
    default test suite (see ``requires_network``). Results are model-reported
    summaries of real pages: recorded as ``Provenance.SOURCE`` but left at
    ``verification_status="unverified"``, because a summary is not the same as
    having read the primary document.
    """

    tool_id = "anthropic-web-search"
    produces_real_evidence = True

    def __init__(
        self,
        *,
        model: str = "claude-opus-5-5",
        max_uses: int = 4,
        tool_type: str = "web_search_20260209",
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415
        except ImportError as exc:
            raise ToolUnavailable(
                "web search needs the 'anthropic' package: pip install anthropic"
            ) from exc
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise ToolUnavailable(
                "no Anthropic credentials found. Set ANTHROPIC_API_KEY or run "
                "`ant auth login`, or use --search fixture for an offline demo."
            )
        self._client = anthropic.Anthropic()
        self.model = model
        self.max_uses = max_uses
        self.tool_type = tool_type

    def search(self, query: str, context: dict) -> list[SearchHit]:
        from pydantic import Field  # noqa: PLC0415

        from ..core.schemas import Base  # noqa: PLC0415

        class _Finding(Base):
            claim: str
            source_uri: str = ""
            source_type: str = "web"
            stance: str = "uncertain"
            published_at: str = ""
            limitations: list[str] = Field(default_factory=list)

        class _Findings(Base):
            findings: list[_Finding] = Field(default_factory=list)

        system = (
            "You are a Researcher. Search the web and report only what the sources "
            "actually say. Every finding must carry the real URL you read it from. "
            "Never invent a source, a URL or a publication date. Include findings "
            "that contradict the premise -- those are the most valuable. If you "
            "found nothing relevant, return an empty list."
        )
        response = self._client.messages.create(
            model=self.model,
            max_tokens=16_000,
            system=system,
            tools=[
                {"type": self.tool_type, "name": "web_search", "max_uses": self.max_uses}
            ],
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Research question: {query}\n\n"
                        "After searching, output JSON: "
                        '{"findings":[{"claim","source_uri","source_type","stance",'
                        '"published_at","limitations"}]}. '
                        "stance is one of supports/refutes/uncertain, relative to the "
                        "research question."
                    ),
                }
            ],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _Findings.model_json_schema(),
                }
            },
        )
        text = next(
            (b.text for b in response.content if getattr(b, "type", None) == "text"), None
        )
        if not text:
            return []
        try:
            findings = _Findings.model_validate_json(text).findings
        except ValueError:
            return []

        hits = []
        for f in findings:
            if not f.source_uri.strip():
                # A claim with no URL is an assertion, not a retrieved source.
                continue
            hits.append(
                SearchHit(
                    claim=f.claim,
                    source_type=f.source_type or "web",
                    source_uri=f.source_uri,
                    provenance=Provenance.SOURCE,
                    stance=Stance(f.stance if f.stance in {s.value for s in Stance} else "uncertain"),
                    quality=3,
                    limitations=list(f.limitations)
                    + ["Model-reported summary; primary document not independently read."],
                    verification_status="unverified",
                    published_at=f.published_at or None,
                )
            )
        return hits


class UnavailableSearchTool:
    """Explicit placeholder recording that retrieval could not be performed.

    Returning nothing with a stated reason is the correct behaviour when no
    search backend is configured -- silence is preferable to filler.
    """

    tool_id = "unavailable-search"
    produces_real_evidence = False

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def search(self, query: str, context: dict) -> list[SearchHit]:
        return []


def build_search_tools(
    *,
    mode: str,
    fixture_dir: str | Path | None = None,
    local_root: str | Path | None = None,
    human_evidence: str | Path | None = None,
) -> tuple[list[ResearchTool], list[str]]:
    """Assemble the tool chain for a run.

    Returns ``(tools, notes)`` where ``notes`` records any capability that was
    requested but is unavailable, for honest inclusion in the final report.
    """
    tools: list[ResearchTool] = []
    notes: list[str] = []

    if mode in ("fixture", "all") and fixture_dir:
        tools.append(FixtureSearchTool(fixture_dir))
    if mode in ("local", "all") and local_root:
        tools.append(LocalFileSearchTool(local_root))
    if human_evidence:
        tools.append(HumanEvidenceTool(human_evidence))
    if mode in ("web", "all"):
        try:
            tools.append(AnthropicWebSearchTool())
        except ToolUnavailable as exc:
            notes.append(f"Web search unavailable: {exc}")

    if not tools:
        reason = f"no search backend available for mode '{mode}'"
        tools.append(UnavailableSearchTool(reason))
        notes.append(
            f"No evidence retrieval was possible ({reason}). All conclusions in this "
            "run rest on zero gathered evidence and must be treated as unverified."
        )
    return tools, notes
