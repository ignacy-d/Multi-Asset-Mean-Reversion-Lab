"""Explicit registration of alpha modules and their portfolio grouping."""

from dataclasses import dataclass

from mr_lab.portfolio.contracts import require_text


class RegistryInvariantError(ValueError):
    """Raised when module configuration cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class AlphaModuleSpec:
    module_id: str
    sleeve_id: str
    opportunity_family_id: str
    enabled: bool = True

    def __post_init__(self) -> None:
        require_text("module_id", self.module_id)
        require_text("sleeve_id", self.sleeve_id)
        require_text("opportunity_family_id", self.opportunity_family_id)


class AlphaModuleRegistry:
    """Immutable-by-convention registry reconstructed from explicit configuration."""

    def __init__(self, specs: tuple[AlphaModuleSpec, ...]) -> None:
        by_module: dict[str, AlphaModuleSpec] = {}
        family_sleeves: dict[str, str] = {}
        for spec in specs:
            if spec.module_id in by_module:
                raise RegistryInvariantError(f"duplicate module_id: {spec.module_id}")
            existing = family_sleeves.setdefault(
                spec.opportunity_family_id, spec.sleeve_id
            )
            if existing != spec.sleeve_id:
                raise RegistryInvariantError(
                    "an opportunity family cannot span sleeves: "
                    f"{spec.opportunity_family_id}"
                )
            by_module[spec.module_id] = spec
        self._by_module = by_module

    def get(self, module_id: str) -> AlphaModuleSpec:
        try:
            return self._by_module[module_id]
        except KeyError as error:
            raise RegistryInvariantError(
                f"unregistered alpha module: {module_id}"
            ) from error
