"""Every default in the compiled form must validate against its own schema.

Catches defaults the form's submit validation rejects (e.g. ``null`` in a ``type: string``
field inside a union row), which surface in Desktop as misleading "required" errors on
fields that are already set. Runs against the committed compiled ``rjsf.json``.
"""

import json
from pathlib import Path

import jsonschema
import pytest

RJSF = (
    Path(__file__).resolve().parents[1]
    / "ecoscope-workflows-occupancy-model-workflow"
    / "ecoscope_workflows_occupancy_model_workflow"
    / "rjsf.json"
)


# Platform-owned fields whose null default is filled by a Desktop custom widget.
KNOWN_WIDGET_FILLED = {"time_range.properties.timezone"}


def _walk(node, path):
    if isinstance(node, dict):
        if "default" in node and not path.startswith("$defs") and path not in KNOWN_WIDGET_FILLED:
            yield path, node
        for k, v in node.items():
            if k in ("default", "examples"):
                continue
            yield from _walk(v, f"{path}.{k}" if path else k)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")


def _cases():
    if not RJSF.exists():
        return []
    rjsf = json.loads(RJSF.read_text())
    return [(p, n, rjsf.get("$defs", {})) for p, n in _walk(rjsf.get("properties", {}), "")]


@pytest.mark.skipif(not RJSF.exists(), reason="workflow not compiled")
@pytest.mark.parametrize("path,node,defs", _cases(), ids=[c[0] for c in _cases()])
def test_form_default_validates(path, node, defs):
    schema = {k: v for k, v in node.items() if k != "default"}
    schema["$defs"] = defs
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(node["default"]))
    assert not errors, f"{path}: default {node['default']!r} fails: {[e.message[:200] for e in errors]}"
