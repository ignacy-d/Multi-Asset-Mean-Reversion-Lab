"""Typed, fail-closed provenance contract for empirical research runs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import cast

from mr_lab.spec_lock import STUDY_ID_PATTERN

RUN_MANIFEST_SCHEMA_VERSION = "mr-lab-run-manifest-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
GIT_REVISION_PATTERN = re.compile(r"[0-9a-f]{40,64}\Z")
EXECUTION_STATUSES = frozenset(("PLANNED", "RUNNING", "SUCCEEDED", "FAILED"))
RESULT_CLASSIFICATIONS = frozenset(("PASS", "KILL", "INCONCLUSIVE"))
type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]
type FrozenJsonValue = (
    JsonScalar | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue]
)


class RunManifestError(ValueError):
    """Raised when empirical-run provenance is incomplete or malformed."""


def _required_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RunManifestError(f"{field} must be a non-empty string")
    return value


def _validate_json(value: object, field: str) -> JsonValue:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RunManifestError(f"{field} contains a non-finite number")
        return value
    if isinstance(value, list):
        return [_validate_json(item, field) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: _validate_json(item, field) for key, item in value.items()}
    raise RunManifestError(f"{field} must contain only JSON values")


def _freeze_json(value: JsonValue) -> FrozenJsonValue:
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in value.items()}
        )
    return value


def _thaw_json(value: object) -> JsonValue:
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise RunManifestError("parameters must be a string-keyed mapping")
        return {cast(str, key): _thaw_json(item) for key, item in value.items()}
    return _validate_json(value, "parameters")


@dataclass(frozen=True)
class AuthenticatedDataIdentity:
    """One explicitly named and authenticated empirical input."""

    name: str
    identity: str
    path: str

    def __post_init__(self) -> None:
        _required_string(self.name, "authenticated_data.name")
        _required_string(self.identity, "authenticated_data.identity")
        _required_string(self.path, "authenticated_data.path")

    def to_dict(self) -> dict[str, str]:
        return {"identity": self.identity, "name": self.name, "path": self.path}


@dataclass(frozen=True)
class GeneratedArtifact:
    """One generated artifact bound to its exact bytes."""

    path: str
    sha256: str

    def __post_init__(self) -> None:
        _required_string(self.path, "generated_artifacts.path")
        if not SHA256_PATTERN.fullmatch(self.sha256):
            raise RunManifestError(
                "generated_artifacts.sha256 must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class RunManifest:
    """Complete provenance supplied by a future authenticated empirical runner."""

    study_id: str
    frozen_spec_sha256: str
    git_revision: str
    registry_identity: str
    registry_path: str
    authenticated_data: tuple[AuthenticatedDataIdentity, ...]
    runner_command: tuple[str, ...]
    parameters: Mapping[str, JsonValue]
    random_seed: int | None
    execution_status: str
    result_classification: str | None
    generated_artifacts: tuple[GeneratedArtifact, ...]
    schema_version: str = RUN_MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_MANIFEST_SCHEMA_VERSION:
            raise RunManifestError("unsupported run-manifest schema version")
        if not STUDY_ID_PATTERN.fullmatch(_required_string(self.study_id, "study_id")):
            raise RunManifestError("invalid study_id")
        if not SHA256_PATTERN.fullmatch(
            _required_string(self.frozen_spec_sha256, "frozen_spec_sha256")
        ):
            raise RunManifestError("frozen_spec_sha256 must be a lowercase SHA-256")
        if not GIT_REVISION_PATTERN.fullmatch(
            _required_string(self.git_revision, "git_revision")
        ):
            raise RunManifestError("git_revision must be a full lowercase commit SHA")
        _required_string(self.registry_identity, "registry_identity")
        _required_string(self.registry_path, "registry_path")
        if (
            not isinstance(self.authenticated_data, tuple)
            or not self.authenticated_data
        ):
            raise RunManifestError(
                "authenticated_data must be a non-empty tuple of identities"
            )
        if not all(
            isinstance(item, AuthenticatedDataIdentity)
            for item in self.authenticated_data
        ):
            raise RunManifestError(
                "authenticated_data must contain authenticated identity objects"
            )
        for field in ("name", "identity", "path"):
            values = [getattr(item, field) for item in self.authenticated_data]
            if len(values) != len(set(values)):
                raise RunManifestError(f"authenticated_data contains duplicate {field}")
        if (
            not isinstance(self.runner_command, tuple)
            or not self.runner_command
            or any(
                not isinstance(part, str) or not part for part in self.runner_command
            )
        ):
            raise RunManifestError(
                "runner_command must be a non-empty tuple of arguments"
            )
        if not isinstance(self.parameters, Mapping) or any(
            not isinstance(key, str) for key in self.parameters
        ):
            raise RunManifestError("parameters must be a string-keyed mapping")
        validated_parameters = _thaw_json(self.parameters)
        assert isinstance(validated_parameters, dict)
        object.__setattr__(self, "parameters", _freeze_json(validated_parameters))
        if self.random_seed is not None and type(self.random_seed) is not int:
            raise RunManifestError("random_seed must be an integer or null")
        if self.execution_status not in EXECUTION_STATUSES:
            raise RunManifestError("unsupported execution_status")
        if self.result_classification is not None and (
            self.result_classification not in RESULT_CLASSIFICATIONS
        ):
            raise RunManifestError("unsupported result_classification")
        if self.execution_status == "SUCCEEDED" and self.result_classification is None:
            raise RunManifestError("a successful run requires a result classification")
        if (
            self.execution_status != "SUCCEEDED"
            and self.result_classification is not None
        ):
            raise RunManifestError("only a successful run may have a classification")
        if not isinstance(self.generated_artifacts, tuple):
            raise RunManifestError("generated_artifacts must be a tuple")
        if self.execution_status == "SUCCEEDED" and not self.generated_artifacts:
            raise RunManifestError("a successful run requires generated artifacts")
        if not all(
            isinstance(item, GeneratedArtifact) for item in self.generated_artifacts
        ):
            raise RunManifestError(
                "generated_artifacts must contain generated artifact objects"
            )
        artifact_paths = [item.path for item in self.generated_artifacts]
        if len(artifact_paths) != len(set(artifact_paths)):
            raise RunManifestError("generated_artifacts contains duplicate paths")

    def to_dict(self) -> dict[str, object]:
        return {
            "authenticated_data": [item.to_dict() for item in self.authenticated_data],
            "execution_status": self.execution_status,
            "frozen_spec_sha256": self.frozen_spec_sha256,
            "generated_artifacts": [
                item.to_dict() for item in self.generated_artifacts
            ],
            "git_revision": self.git_revision,
            "parameters": _thaw_json(self.parameters),
            "random_seed": self.random_seed,
            "registry_identity": self.registry_identity,
            "registry_path": self.registry_path,
            "result_classification": self.result_classification,
            "runner_command": list(self.runner_command),
            "schema_version": self.schema_version,
            "study_id": self.study_id,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> RunManifest:
        required = {
            "authenticated_data",
            "execution_status",
            "frozen_spec_sha256",
            "generated_artifacts",
            "git_revision",
            "parameters",
            "random_seed",
            "registry_identity",
            "registry_path",
            "result_classification",
            "runner_command",
            "schema_version",
            "study_id",
        }
        if set(value) != required:
            missing = sorted(required - set(value))
            extra = sorted(set(value) - required)
            raise RunManifestError(
                f"manifest fields mismatch; missing={missing}, extra={extra}"
            )
        data_values = value["authenticated_data"]
        artifact_values = value["generated_artifacts"]
        command = value["runner_command"]
        parameters = value["parameters"]
        if not isinstance(data_values, list) or not all(
            isinstance(item, dict) and set(item) == {"identity", "name", "path"}
            for item in data_values
        ):
            raise RunManifestError("authenticated_data has invalid structure")
        if not isinstance(artifact_values, list) or not all(
            isinstance(item, dict) and set(item) == {"path", "sha256"}
            for item in artifact_values
        ):
            raise RunManifestError("generated_artifacts has invalid structure")
        if not isinstance(command, list):
            raise RunManifestError("runner_command must be a list")
        if not isinstance(parameters, dict):
            raise RunManifestError("parameters must be an object")
        return cls(
            study_id=_required_string(value["study_id"], "study_id"),
            frozen_spec_sha256=_required_string(
                value["frozen_spec_sha256"], "frozen_spec_sha256"
            ),
            git_revision=_required_string(value["git_revision"], "git_revision"),
            registry_identity=_required_string(
                value["registry_identity"], "registry_identity"
            ),
            registry_path=_required_string(value["registry_path"], "registry_path"),
            authenticated_data=tuple(
                AuthenticatedDataIdentity(
                    name=_required_string(item["name"], "authenticated_data.name"),
                    identity=_required_string(
                        item["identity"], "authenticated_data.identity"
                    ),
                    path=_required_string(item["path"], "authenticated_data.path"),
                )
                for item in data_values
            ),
            runner_command=tuple(command),
            parameters=parameters,
            random_seed=value["random_seed"],  # type: ignore[arg-type]
            execution_status=_required_string(
                value["execution_status"], "execution_status"
            ),
            result_classification=(
                None
                if value["result_classification"] is None
                else _required_string(
                    value["result_classification"], "result_classification"
                )
            ),
            generated_artifacts=tuple(
                GeneratedArtifact(
                    path=_required_string(item["path"], "generated_artifacts.path"),
                    sha256=_required_string(
                        item["sha256"], "generated_artifacts.sha256"
                    ),
                )
                for item in artifact_values
            ),
            schema_version=_required_string(value["schema_version"], "schema_version"),
        )

    @classmethod
    def from_json(cls, payload: str) -> RunManifest:
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise RunManifestError("run manifest is not valid JSON") from error
        if not isinstance(value, dict):
            raise RunManifestError("run manifest must be a JSON object")
        return cls.from_dict(value)

    @classmethod
    def read(cls, path: Path) -> RunManifest:
        try:
            return cls.from_json(path.read_text(encoding="utf-8"))
        except OSError as error:
            raise RunManifestError(f"cannot read run manifest: {path}") from error
