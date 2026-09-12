"""Deterministic conversion of alpha evidence into explicit opportunities."""

from collections import defaultdict

from mr_lab.portfolio.contracts import OpportunitySet, OpportunityStatus
from mr_lab.portfolio.identity import stable_id
from mr_lab.portfolio.registry import AlphaModuleRegistry
from mr_lab.signals import AlphaSignal


class OpportunityInvariantError(ValueError):
    """Raised when evidence identities collide or cannot be handled safely."""


class OpportunityEngine:
    def __init__(self, registry: AlphaModuleRegistry) -> None:
        self._registry = registry

    def build(self, signals: tuple[AlphaSignal, ...]) -> tuple[OpportunitySet, ...]:
        unique: dict[tuple[str, str], AlphaSignal] = {}
        registered = []
        for signal in signals:
            spec = self._registry.get(signal.module_id)
            identity = (signal.module_id, signal.source_event_id)
            previous = unique.get(identity)
            if previous is not None:
                if previous != signal:
                    raise OpportunityInvariantError(
                        f"alpha event identity collision: {identity!r}"
                    )
                continue
            unique[identity] = signal
            registered.append((spec, signal))

        groups = defaultdict(list)
        for spec, signal in registered:
            key = (
                spec.opportunity_family_id,
                spec.sleeve_id,
                signal.instrument,
                signal.timestamp,
                spec.enabled,
            )
            groups[key].append(signal)

        opportunities = []
        for key, members in groups.items():
            family, sleeve, instrument, timestamp, enabled = key
            ordered = tuple(
                sorted(members, key=lambda x: (x.module_id, x.source_event_id))
            )
            directions = tuple(sorted({signal.direction for signal in ordered}))
            if not enabled:
                status = OpportunityStatus.DISABLED
                direction = directions[0] if len(directions) == 1 else None
            elif len(directions) > 1:
                status = OpportunityStatus.CONFLICT
                direction = None
            else:
                status = OpportunityStatus.ACTIONABLE
                direction = directions[0]
            opportunity_id = stable_id(
                "opportunity", family, sleeve, instrument, timestamp, enabled
            )
            opportunities.append(
                OpportunitySet(
                    opportunity_id,
                    family,
                    sleeve,
                    instrument,
                    timestamp,
                    direction,
                    ordered,
                    status,
                    directions if status is OpportunityStatus.CONFLICT else (),
                )
            )
        return tuple(sorted(opportunities, key=lambda item: item.opportunity_id))
