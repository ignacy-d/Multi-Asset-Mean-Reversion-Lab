"""Conservative resampling on Unix-epoch-aligned fixed UTC windows."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from mr_lab.data.dataset import validate_dataset
from mr_lab.data.models import Bar, DataContractError, Timeframe, VolumeSemantics

_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


@dataclass(frozen=True, slots=True)
class IncompleteWindow:
    """A target window skipped because it lacks the expected source intervals."""

    open_time: datetime
    close_time: datetime
    expected_components: int
    observed_components: int


@dataclass(frozen=True, slots=True)
class ResamplingResult:
    """Completed resampled bars and explicitly skipped windows."""

    bars: tuple[Bar, ...]
    incomplete_windows: tuple[IncompleteWindow, ...]


def _aligned_open(timestamp: datetime, duration: timedelta) -> datetime:
    elapsed = timestamp - _EPOCH
    windows = elapsed // duration
    return _EPOCH + windows * duration


def resample_bars(
    bars: list[Bar] | tuple[Bar, ...], target_timeframe: Timeframe
) -> ResamplingResult:
    """Aggregate complete source bars into epoch-aligned fixed UTC windows."""
    report = validate_dataset(bars)
    if not isinstance(target_timeframe, Timeframe):
        raise DataContractError("target_timeframe must be a Timeframe")
    if not report.bars:
        return ResamplingResult((), ())
    source_duration = report.bars[0].timeframe.duration
    target_duration = target_timeframe.duration
    if target_duration <= source_duration:
        raise DataContractError("target timeframe must be larger than source timeframe")
    if target_duration % source_duration != timedelta(0):
        raise DataContractError("target duration must be an integer source multiple")

    expected = target_duration // source_duration
    grouped: dict[datetime, list[Bar]] = {}
    for bar in report.bars:
        window_open = _aligned_open(bar.open_time, target_duration)
        window_close = window_open + target_duration
        if bar.open_time < window_open or bar.close_time > window_close:
            raise DataContractError("source bar crosses a target interval boundary")
        grouped.setdefault(window_open, []).append(bar)

    output: list[Bar] = []
    incomplete: list[IncompleteWindow] = []
    for window_open, components in grouped.items():
        window_close = window_open + target_duration
        contiguous = all(
            bar.open_time == window_open + index * source_duration
            and bar.close_time == window_open + (index + 1) * source_duration
            for index, bar in enumerate(components)
        )
        if len(components) != expected or not contiguous:
            incomplete.append(
                IncompleteWindow(window_open, window_close, expected, len(components))
            )
        else:
            first, last = components[0], components[-1]
            volume = (
                None
                if first.volume_semantics is VolumeSemantics.NONE
                else sum(bar.volume for bar in components if bar.volume is not None)
            )
            output.append(
                Bar(
                    instrument=first.instrument,
                    timeframe=target_timeframe,
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
                    volume_semantics=first.volume_semantics,
                )
            )
    return ResamplingResult(tuple(output), tuple(incomplete))
