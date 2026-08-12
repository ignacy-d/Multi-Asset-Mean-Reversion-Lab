"""Conservative UTC-aligned resampling of completed fixed-duration bars."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mr_lab.data.dataset import validate_dataset
from mr_lab.data.models import Bar, DataContractError, DatasetMetadata, Timeframe


@dataclass(frozen=True, slots=True)
class IncompleteWindow:
    """A UTC target window omitted because expected source bars were absent."""

    open_time: datetime
    close_time: datetime
    observed_bars: int
    expected_bars: int


@dataclass(frozen=True, slots=True)
class ResampleResult:
    """Completed output bars plus explicitly reported incomplete windows."""

    bars: tuple[Bar, ...]
    incomplete_windows: tuple[IncompleteWindow, ...]


def _seconds(timeframe: Timeframe) -> int:
    return int(timeframe.duration.total_seconds())


def _aligned_open(value: datetime, timeframe: Timeframe) -> datetime:
    """Floor a UTC timestamp to a duration boundary anchored at Unix epoch."""
    seconds = _seconds(timeframe)
    epoch_seconds = int(value.timestamp())
    return datetime.fromtimestamp(epoch_seconds - epoch_seconds % seconds, tz=UTC)


def resample_bars(
    bars: Iterable[Bar], target: Timeframe, metadata: DatasetMetadata
) -> ResampleResult:
    """Aggregate complete contiguous source windows into a larger timeframe."""
    source_bars = tuple(bars)
    validate_dataset(source_bars, metadata)
    if metadata.native_timeframe is None:
        raise DataContractError("resampling requires a native source timeframe")
    source = metadata.native_timeframe
    source_seconds = _seconds(source)
    target_seconds = _seconds(target)
    if target_seconds <= source_seconds or target_seconds % source_seconds:
        raise DataContractError("target timeframe must be a larger integer multiple")

    expected = target_seconds // source_seconds
    windows: dict[datetime, list[Bar]] = defaultdict(list)
    for bar in source_bars:
        if bar.close_time - bar.open_time != source.duration:
            raise DataContractError("source bar interval must match its timeframe")
        if _aligned_open(bar.open_time, source) != bar.open_time:
            raise DataContractError("source bar is not UTC timeframe-aligned")
        windows[_aligned_open(bar.open_time, target)].append(bar)

    output: list[Bar] = []
    incomplete: list[IncompleteWindow] = []
    for window_open in sorted(windows):
        components = windows[window_open]
        window_close = window_open + target.duration
        expected_opens = {
            window_open + timedelta(seconds=index * source_seconds)
            for index in range(expected)
        }
        observed_opens = {bar.open_time for bar in components}
        if len(components) != expected or observed_opens != expected_opens:
            incomplete.append(
                IncompleteWindow(window_open, window_close, len(components), expected)
            )
            continue

        first, last = components[0], components[-1]
        volumes = [bar.volume for bar in components]
        volume = (
            None
            if first.volume is None
            else sum(value for value in volumes if value is not None)
        )
        output.append(
            Bar(
                instrument=first.instrument,
                timeframe=target,
                open_time=window_open,
                close_time=window_close,
                available_at=max(
                    window_close, *(bar.available_at for bar in components)
                ),
                open=first.open,
                high=max(bar.high for bar in components),
                low=min(bar.low for bar in components),
                close=last.close,
                price_basis=first.price_basis,
                volume=volume,
                volume_type=first.volume_type,
            )
        )
    return ResampleResult(tuple(output), tuple(incomplete))
