"""Versioned task instructions, independent of the mirrored Bionic prompt.

This module has no model, network, runner, or training dependencies. The legacy
profile is opt-in and supplied by the caller; it is never mixed with this policy.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

DEFAULT_PROFILE = "work-contract-v1"
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "work-contract-v1.md"
TOOLS_MARKER = "\n\nAvailable tool schemas (use these exact names and argument contracts):\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _policy(profile: str) -> str:
    if profile != DEFAULT_PROFILE:
        raise ValueError(f"Unknown prompt profile: {profile!r}")
    return PROMPT_PATH.read_text(encoding="utf-8").strip()


def build_system(tools, profile: str = DEFAULT_PROFILE, *,
                 legacy_system: Callable | None = None) -> str:
    """Build a replacement policy without changing caller-owned tool schemas.

    ``tools`` is the native API tool-schema list, also passed unchanged to the
    model API by the caller. ``legacy`` requires its original prompt builder.
    Unknown profiles and invalid schemas fail before a request can be made.
    """
    if profile == "legacy":
        if legacy_system is None:
            raise ValueError("The legacy profile requires legacy_system(tools)")
        result = legacy_system(tools)
        if not isinstance(result, str) or not result.strip():
            raise ValueError("legacy_system must return a nonempty prompt string")
        return result
    policy = _policy(profile)
    if not isinstance(tools, (list, tuple)) or any(not isinstance(t, dict) for t in tools):
        raise TypeError("tools must be a list or tuple of native tool-schema dictionaries")
    schemas = json.dumps(tools, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return policy + TOOLS_MARKER + schemas


def prompt_metadata(profile: str = DEFAULT_PROFILE, *, tools=None,
                    legacy_system: Callable | None = None) -> dict:
    """Identify the policy and, when tools are given, the exact rendered prompt.

    Hashes use normalized UTF-8 text, not platform-dependent file newline bytes.
    Legacy metadata requires tools and the builder because its source is external.
    Nothing is executed except the explicitly supplied legacy prompt builder.
    """
    if profile == "legacy":
        if tools is None:
            raise ValueError("Legacy metadata requires tools and legacy_system")
        rendered = build_system(tools, profile, legacy_system=legacy_system)
        return {"profile": "legacy", "version": "legacy", "source_sha256": None,
                "rendered_sha256": _sha256(rendered)}
    policy = _policy(profile)
    metadata = {"profile": profile, "version": "1", "source_sha256": _sha256(policy)}
    if tools is not None:
        metadata["rendered_sha256"] = _sha256(build_system(tools, profile))
    return metadata
