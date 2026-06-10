"""
knowledge_layer.py - JSON-backed reusable knowledge loader.

The analyst bundle is deal-specific memory. Files under knowledge/observations
are reviewed learning candidates. Files under knowledge/patterns are the small,
distilled runtime layer that can be loaded cheaply on every analysis.

This module intentionally does not load raw observations at runtime. That keeps
the prompt and deterministic logic bounded: many observations become a few
evidence-scored patterns.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


KNOWLEDGE_DIR = Path("knowledge")
PATTERNS_DIR = KNOWLEDGE_DIR / "patterns"

PATTERN_FILES = {
    "model_patterns": "model_patterns.json",
    "metric_patterns": "metric_patterns.json",
    "business_plan_patterns": "business_plan_patterns.json",
}


class KnowledgePatternError(ValueError):
    """Raised when a runtime knowledge pattern is malformed."""


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_pattern_schema(pattern_type: str, payload: dict[str, Any]) -> list[str]:
    """
    Validate the minimal schema needed for safe runtime use.

    The validator is intentionally lightweight: it catches malformed active
    patterns without turning knowledge files into a rigid database migration.
    """
    errors: list[str] = []
    if not isinstance(payload, dict):
        return [f"{pattern_type}: payload must be an object"]
    if "version" not in payload:
        errors.append(f"{pattern_type}: missing version")

    if pattern_type == "model_patterns":
        for idx, pattern in enumerate(payload.get("patterns", [])):
            prefix = f"{pattern_type}.patterns[{idx}]"
            for field in ("pattern_id", "status", "model_type", "signals"):
                if field not in pattern:
                    errors.append(f"{prefix}: missing {field}")
    elif pattern_type == "metric_patterns":
        for midx, metric in enumerate(payload.get("metrics", [])):
            mprefix = f"{pattern_type}.metrics[{midx}]"
            if "metric" not in metric:
                errors.append(f"{mprefix}: missing metric")
            for ridx, rule in enumerate(metric.get("rules", [])):
                rprefix = f"{mprefix}.rules[{ridx}]"
                for field in ("rule_id", "status", "pattern", "confidence", "evidence_count"):
                    if field not in rule:
                        errors.append(f"{rprefix}: missing {field}")
    elif pattern_type == "business_plan_patterns":
        for idx, pattern in enumerate(payload.get("patterns", [])):
            prefix = f"{pattern_type}.patterns[{idx}]"
            for field in ("pattern_id", "status", "property_type", "signals"):
                if field not in pattern:
                    errors.append(f"{prefix}: missing {field}")
    return errors


@lru_cache(maxsize=1)
def load_knowledge_layer(base_dir: Path | str = KNOWLEDGE_DIR) -> dict[str, Any]:
    """
    Load the distilled reusable knowledge layer.

    Missing pattern files return an empty section so local development remains
    forgiving, but malformed JSON should raise loudly.
    """
    root = Path(base_dir)
    patterns_dir = root / "patterns"
    loaded: dict[str, Any] = {}
    for key, filename in PATTERN_FILES.items():
        path = patterns_dir / filename
        loaded[key] = _load_json(path) if path.exists() else {}
        errors = validate_pattern_schema(key, loaded[key]) if loaded[key] else []
        if errors:
            raise KnowledgePatternError("; ".join(errors))
    return {
        "knowledge_dir": str(root),
        "patterns": loaded,
    }


def load_active_patterns(base_dir: Path | str = KNOWLEDGE_DIR) -> dict[str, Any]:
    """
    Load only runtime-eligible active patterns.

    Raw observations are never read here. Candidate/inactive patterns remain in
    the catalogs as evidence and review material, but cannot influence runtime.
    """
    layer = load_knowledge_layer(base_dir)
    patterns = layer["patterns"]

    model_payload = patterns.get("model_patterns", {})
    metric_payload = patterns.get("metric_patterns", {})
    business_payload = patterns.get("business_plan_patterns", {})

    active_metric_groups: list[dict[str, Any]] = []
    for metric in metric_payload.get("metrics", []):
        active_rules = [
            rule for rule in metric.get("rules", [])
            if rule.get("status") == "active"
        ]
        if active_rules:
            active_metric_groups.append({
                **{k: v for k, v in metric.items() if k != "rules"},
                "rules": active_rules,
            })

    return {
        "model_patterns": [
            p for p in model_payload.get("patterns", [])
            if p.get("status") == "active"
        ],
        "metric_patterns": active_metric_groups,
        "business_plan_patterns": [
            p for p in business_payload.get("patterns", [])
            if p.get("status") == "active"
        ],
    }


def _rule_contradiction_rate(rule: dict[str, Any]) -> float:
    evidence_count = int(rule.get("evidence_count") or 0)
    contradiction_count = int(rule.get("contradiction_count") or 0)
    total = evidence_count + contradiction_count
    return contradiction_count / total if total else 0.0


def active_metric_rules(metric_name: str | None = None) -> list[dict[str, Any]]:
    """
    Return active metric rules, optionally filtered by metric name.

    Candidate rules are intentionally excluded from runtime use. Promotion is a
    separate QC step driven by evidence count, contradiction rate, and review.
    """
    layer = load_knowledge_layer()
    metric_patterns = layer["patterns"].get("metric_patterns", {})
    policy = metric_patterns.get("promotion_policy", {})
    max_contradiction_rate = float(policy.get("max_contradiction_rate") or 1.0)

    out: list[dict[str, Any]] = []
    for metric in metric_patterns.get("metrics", []):
        if metric_name and metric.get("metric") != metric_name:
            continue
        for rule in metric.get("rules", []):
            if rule.get("status") != "active":
                continue
            if _rule_contradiction_rate(rule) > max_contradiction_rate:
                continue
            out.append({
                "metric": metric.get("metric"),
                "canonical_unit": metric.get("canonical_unit"),
                **rule,
            })
    return out


def knowledge_summary() -> dict[str, Any]:
    """Return a compact count summary for diagnostics/UI."""
    layer = load_knowledge_layer()
    patterns = layer["patterns"]

    model_patterns = patterns.get("model_patterns", {}).get("patterns", [])
    business_patterns = patterns.get("business_plan_patterns", {}).get("patterns", [])
    metric_groups = patterns.get("metric_patterns", {}).get("metrics", [])
    metric_rules = [
        rule
        for metric in metric_groups
        for rule in metric.get("rules", [])
    ]

    return {
        "model_patterns": len(model_patterns),
        "metric_groups": len(metric_groups),
        "metric_rules": len(metric_rules),
        "active_metric_rules": len(active_metric_rules()),
        "business_plan_patterns": len(business_patterns),
    }
