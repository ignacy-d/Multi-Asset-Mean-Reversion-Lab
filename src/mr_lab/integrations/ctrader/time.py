"""Conversion boundary for Python and pythonnet UTC timestamp values."""

from __future__ import annotations

from datetime import UTC, datetime


def to_python_utc(value: object) -> datetime:
    """Copy Python or pythonnet ``System.DateTime`` values into Python UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("platform timestamp must be timezone-aware")
        return value.astimezone(UTC)
    try:
        return datetime(
            int(value.Year),
            int(value.Month),
            int(value.Day),
            int(value.Hour),
            int(value.Minute),
            int(value.Second),
            int(value.Millisecond) * 1000,
            tzinfo=UTC,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("unsupported native platform timestamp") from exc
