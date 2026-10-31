# Demo fixtures — SIMULATED, not real sources

Every record here has `"provenance": "simulated"`. They exist so the harness can
run end to end with no network access, and so the pipeline's handling of
supporting vs. refuting vs. uncertain evidence can be exercised.

They are **not** real literature. None of the `source_uri` values point to a real
document, which is why none of them claims to. Fabricating realistic-looking
citations is precisely the failure mode this harness is built to detect, so the
demo data declares itself instead.

To supply real evidence:
- `--search web` (real web search; needs Anthropic credentials)
- `--search local --local-root <dir>` (real local documents)
- `--human-evidence <file.jsonl>` (findings a person collected)
