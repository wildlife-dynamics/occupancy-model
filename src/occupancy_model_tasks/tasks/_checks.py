"""Input checks surfaced in the Model Summary."""

from collections import Counter
from typing import Annotated, Any

from ecoscope.platform.annotations import AnyDataFrame
from ecoscope.platform.tasks.io._earthranger import CombinedPatrolAndEventsParams
from pydantic import Field
from pydantic.json_schema import SkipJsonSchema
from wt_registry import register
from wt_task import SkippedDependencyFallback


def _none_if_skipped(obj: Any) -> Any:
    from wt_task.skip import SkipSentinel

    return None if isinstance(obj, SkipSentinel) else obj


@register()
def check_target_event_types(
    combined_params: Annotated[
        CombinedPatrolAndEventsParams,
        Field(description="Patrol and event type selection.", exclude=True),
    ],
    patrols_df: Annotated[
        SkipJsonSchema[None] | AnyDataFrame,
        Field(description="Patrols (from get_patrols_from_combined_params).", exclude=True),
        SkippedDependencyFallback(_none_if_skipped),
    ] = None,
) -> dict:
    """Count the event types recorded on the selected patrols, and how many of them match
    the requested target types. A typo'd or display-name event type (``bird_sighting``
    instead of ``bird_sighting_rep``) otherwise just looks like "no events".
    """
    requested = list(combined_params.event_types or [])
    counts: Counter = Counter()
    if patrols_df is not None and len(patrols_df) and "patrol_segments" in patrols_df:
        for segments in patrols_df["patrol_segments"]:
            for seg in segments if segments is not None else []:
                events = seg.get("events") if hasattr(seg, "get") else None
                for ev in events if events is not None else []:  # may be a numpy array
                    counts[ev.get("event_type")] += 1
    matched = sum(n for t, n in counts.items() if not requested or t in requested)
    return {
        "requested": requested,
        "matched": int(matched),
        "available": {str(t): int(n) for t, n in counts.most_common()},
        "unknown": [t for t in requested if t not in counts],
    }


def event_type_warnings(check: dict | None) -> list[str]:
    if not check or not check.get("requested"):
        return []
    available = check.get("available", {})
    listing = ", ".join(f"{t} ({n})" for t, n in list(available.items())[:10]) or "none"
    if check.get("matched", 0) == 0:
        return [
            f"None of the requested event types ({', '.join(check['requested'])}) were recorded on the "
            f"selected patrols. Event types on these patrols: {listing}. Enter event types as their "
            "EarthRanger value (e.g. snare_rep), not the display name."
        ]
    if check.get("unknown"):
        return [f"Event types not found on the selected patrols: {', '.join(check['unknown'])}."]
    return []
