"""Skip conditions used by the occupancy-model spec."""

from typing import Any

from wt_registry import register


@register()
def all_dependencies_skipped(*args: Any) -> bool:
    """True when every argument is a skip: a SkipSentinel, a ``(key, SkipSentinel)`` pair,
    or a keyed iterable made only of those (checked recursively, so nesting depth doesn't matter). Unlike ``all_keyed_iterables_are_skips`` it accepts a
    wholly skipped keyed iterable, and it skips on no arguments at all -- so a
    ``groupbykey`` over map layers skips (keeping the widget placeholder) only when every
    layer is missing, and still combines whatever layers exist otherwise.
    """
    from wt_task.skip import SkipSentinel

    def skipped(a: Any) -> bool:
        if isinstance(a, SkipSentinel):
            return True
        if isinstance(a, tuple) and len(a) == 2:  # (key, value) pair of a keyed iterable
            return skipped(a[1])
        if isinstance(a, list):  # a keyed iterable (an empty one counts as skipped)
            return all(skipped(x) for x in a)
        return False

    return all(skipped(a) for a in args)
