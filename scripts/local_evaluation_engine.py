"""Local evaluation engine — drop-in replacement for LLM-based evaluation.

Activated by setting env var LOCAL_SCORER=1. Only handles data quality
evaluation tasks. Non-evaluation tasks (puzzles, general prompts) are
not routed here.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

log = logging.getLogger("validator.local_evaluation")

# Add the minework_validator src to path so we can import local_scorer
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC_PATH = _PROJECT_ROOT.parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from minework_validator.local_scorer import score_submission, detect_schema_name


def is_local_scorer_enabled() -> bool:
    return os.environ.get("LOCAL_SCORER", "").strip() in ("1", "true", "yes")


def local_evaluate(
    cleaned_data: str | dict[str, Any],
    structured_data: dict[str, Any],
    schema_fields: list[str],
    repeat_cleaned_data: str = "",
    dataset_schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Run local rule-based evaluation.

    Returns a dict with keys matching EvaluationResult fields,
    or None if the local scorer cannot handle this input (fallback to LLM).
    """
    if not is_local_scorer_enabled():
        return None

    if not structured_data or not isinstance(structured_data, dict):
        return None

    # Detect schema — if we can't determine it, fall back to LLM
    try:
        schema_name = detect_schema_name(structured_data)
    except ValueError:
        log.debug("Local scorer: cannot detect schema, falling back to LLM")
        return None

    # Normalize cleaned_data
    if isinstance(cleaned_data, dict):
        cleaned_str = json.dumps(cleaned_data, ensure_ascii=False)
    else:
        cleaned_str = str(cleaned_data) if cleaned_data else None

    # Normalize repeat_cleaned_data
    repeat_str: str | None = None
    if repeat_cleaned_data and str(repeat_cleaned_data).strip():
        repeat_str = str(repeat_cleaned_data)

    result = score_submission(
        structured_data=structured_data,
        schema_name=schema_name,
        cleaned_data=cleaned_str,
        repeat_cleaned_data=repeat_str,
        schema_fields=schema_fields if schema_fields else None,
    )

    log.info(
        "Local scorer: schema=%s score=%d result=%s (C=%.0f A=%.0f T=%.0f S=%.0f)",
        schema_name, result.score, result.result,
        result.breakdown.completeness, result.breakdown.accuracy,
        result.breakdown.type_correctness, result.breakdown.sufficiency,
    )

    return {
        "result": result.result,
        "verdict": result.verdict,
        "consistent": result.consistent,
        "score": result.score,
    }
