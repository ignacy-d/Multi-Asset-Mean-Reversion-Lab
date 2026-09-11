# ruff: noqa: E501
"""Authenticated frozen-2024 triage of already-computed Stage 4B evidence.

Performance is computed only for exact strategy/exit cells.  Family and
execution views contain descriptive robustness and opportunity evidence; they
never manufacture a portfolio or choose a representative trade outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import sqlite3
import statistics
import tempfile
import time
from collections import defaultdict
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

from mr_lab.ornstein_uhlenbeck import frozen_ou_eligibility_spec
from mr_lab.stage4b import STAGE4B_METHODOLOGY_ID
from mr_lab.stage4c import CostProfile, net_pips

SCHEMA = "stage4-discovery-atlas-v2"
METHODOLOGY = "exact-cell-performance-family-plateau-v1"
FROZEN_START, FROZEN_END = "2024-01-01", "2024-12-31"
HEADLINE_COSTS = (("mean", 0.0), ("p75", 0.1), ("p90", 0.25), ("p95", 0.5))
SETUP_FIELDS = (
    "instrument",
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "lookback",
    "signal_threshold",
    "entry_mode",
    "filter_family",
    "filter_spec_id",
    "process_spec_id",
)
EXIT_FIELDS = ("tp_target_fraction", "sl_extension_fraction", "time_stop_minutes")
CELL_FIELDS = (*SETUP_FIELDS, *EXIT_FIELDS)
EXECUTION_FIELDS = (
    "instrument",
    "signal_timeframe",
    "session",
    "direction",
    "entry_mode",
)
HYPOTHESIS_FIELDS = (
    "signal_timeframe",
    "session",
    "direction",
    "benchmark_family",
    "lookback",
    "signal_threshold",
    "entry_mode",
    "filter_family",
    "filter_spec_id",
    "process_spec_id",
    *EXIT_FIELDS,
)
OUTPUTS = (
    "setup-family-matrix.csv",
    "execution-family-matrix.csv",
    "cross-asset-hypotheses.csv",
    "cost-robustness.csv",
    "overlap-with-module-a.csv",
    "candidate-shortlist.csv",
    "report.md",
)
RAW_FILES = ("candidate-events.jsonl", "trades.jsonl", "execution-audit.json")

LOGGER = logging.getLogger(__name__)


class DiscoveryAtlasError(ValueError):
    """Evidence violates the frozen atlas input contract."""


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _reject_sealed(path):
    # Must precede exists/stat/open/rglob for every user-supplied evidence path.
    if "2025" in str(path):
        raise DiscoveryAtlasError("sealed-period paths are forbidden")


def _jsonl(path):
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise DiscoveryAtlasError(
                    f"malformed authenticated row {number}"
                ) from error


def _line_count(path):
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _load_registry(path):
    registry = json.loads(path.read_text())
    if registry.get("registry_schema_version") != "stage-4a-2024-corpus-registry-v1":
        raise DiscoveryAtlasError("unexpected frozen registry schema")
    return registry


def _manifest_identity(manifest):
    fields = (
        "source_commit_sha",
        "stage4b_methodology_id",
        "instrument",
        "registry_identity",
        "corpus_id",
        "assembled_dataset_id",
        "source_workflow_run_id",
        "source_artifact_id",
        "source_artifact_name",
        "source_mode",
        "source_acquisition_commit_sha",
        "filter_family",
        "filter_spec_id",
        "process_spec_id",
        "shard_count",
        "full_candidate_count",
        "full_group_keys",
    )
    return tuple(_json(manifest.get(field)) for field in fields)


def _manifest_identity_key(manifest):
    """Return a compact key for the exact canonical logical-run identity."""
    canonical_identity = _json(_manifest_identity(manifest)).encode()
    return hashlib.sha256(canonical_identity).hexdigest()


def _authenticate_run(
    records, registry, registry_sha, role, database, run_number, profile
):
    """Authenticate one complete logical run, never indexes across runs."""
    records.sort(key=lambda item: item[1].get("shard_index", -1))
    reference = records[0][1]
    count = reference.get("shard_count")
    if not isinstance(count, int) or count < 1:
        raise DiscoveryAtlasError("invalid shard count")
    if {manifest.get("shard_index") for _, manifest in records} != set(range(count)):
        raise DiscoveryAtlasError("incomplete per-run shard indexes")
    identity = _manifest_identity(reference)
    if any(_manifest_identity(manifest) != identity for _, manifest in records):
        raise DiscoveryAtlasError("logical-run shard provenance mismatch")
    if reference.get("stage4b_methodology_id") != STAGE4B_METHODOLOGY_ID:
        raise DiscoveryAtlasError("Stage 4B methodology mismatch")
    if len(reference.get("source_commit_sha", "")) != 40:
        raise DiscoveryAtlasError("missing Stage 4B source commitment")
    if reference.get("registry_identity") != registry_sha:
        raise DiscoveryAtlasError("registry commitment mismatch")
    instrument = reference.get("instrument")
    expected = registry.get("instruments", {}).get(instrument)
    if not expected or expected.get("verification_status") != "verified":
        raise DiscoveryAtlasError("instrument lacks verified frozen registry evidence")
    source_mode = expected.get("source_mode", "github-artifact")
    if reference.get("source_mode", "github-artifact") != source_mode:
        raise DiscoveryAtlasError("registry source_mode mismatch")
    for field in (
        "corpus_id",
        "assembled_dataset_id",
        "source_workflow_run_id",
        "source_artifact_id",
        "source_artifact_name",
        "source_acquisition_commit_sha",
    ):
        if reference.get(field) != expected.get(field):
            raise DiscoveryAtlasError(f"registry {field} mismatch")
    if (expected.get("requested_start_date"), expected.get("requested_end_date")) != (
        FROZEN_START,
        FROZEN_END,
    ):
        raise DiscoveryAtlasError("registry is not frozen 2024")
    full_groups = reference.get("full_group_keys")
    combined_groups = [
        group for _, manifest in records for group in manifest.get("group_keys", [])
    ]
    if not isinstance(full_groups, list) or combined_groups != full_groups:
        raise DiscoveryAtlasError("incomplete or overlapping full group universe")
    if sum(
        manifest.get("shard_candidate_count", -1) for _, manifest in records
    ) != reference.get("full_candidate_count"):
        raise DiscoveryAtlasError("incomplete candidate coverage")

    database.execute("DELETE FROM events")
    commitments, process_ids = [], set()
    candidate_count = trade_count = 0
    for directory, manifest in records:
        hashes, counts = manifest.get("file_sha256", {}), manifest.get("row_counts", {})
        for name in RAW_FILES:
            path = directory / name
            if hashes.get(name) != _sha(path):
                raise DiscoveryAtlasError(f"shard raw hash mismatch: {name}")
            if name.endswith("jsonl") and counts.get(name) != _line_count(path):
                raise DiscoveryAtlasError(f"shard raw row-count mismatch: {name}")
        audit = json.loads((directory / "execution-audit.json").read_text())
        for field in (
            "stage4b_methodology_id",
            "source_commit_sha",
            "instrument",
            "corpus_id",
            "assembled_dataset_id",
            "filter_family",
            "filter_spec_id",
        ):
            if audit.get(field) != reference.get(field):
                raise DiscoveryAtlasError(f"execution-audit identity mismatch: {field}")
        process_ids.add(audit.get("process_spec_id"))
        if reference.get("process_spec_id") is not None and audit.get(
            "process_spec_id"
        ) != reference.get("process_spec_id"):
            raise DiscoveryAtlasError("execution-audit process provenance mismatch")
        for event in _jsonl(directory / "candidate-events.jsonl"):
            signal = event["signal"]
            if (
                signal.get("instrument") != instrument
                or signal.get("source_corpus_id") != reference["corpus_id"]
                or signal.get("assembled_dataset_id")
                != reference["assembled_dataset_id"]
            ):
                raise DiscoveryAtlasError("candidate provenance mismatch")
            event_id = event["candidate_event_id"]
            try:
                database.execute(
                    "INSERT INTO events VALUES (?, ?)",
                    (event_id, _time(signal["signal_timestamp"]).isoformat()),
                )
            except sqlite3.IntegrityError as error:
                raise DiscoveryAtlasError(
                    "duplicate candidate across shards"
                ) from error
            candidate_count += 1
        commitments.append(
            {
                "role": role,
                "manifest_sha256": _sha(directory / "shard-manifest.json"),
                "raw_sha256": {name: hashes[name] for name in RAW_FILES},
            }
        )
    if candidate_count != reference["full_candidate_count"]:
        raise DiscoveryAtlasError("candidate rows do not cover declared universe")
    if len(process_ids) != 1:
        raise DiscoveryAtlasError("logical-run process provenance mismatch")
    reference = reference | {"process_spec_id": next(iter(process_ids))}
    provenance = tuple(
        reference.get(field)
        for field in ("filter_family", "filter_spec_id", "process_spec_id")
    )
    run_key = _manifest_identity_key(reference)
    insert_sql = (
        "INSERT INTO trades(role, run_key, candidate_id, cell_key, setup_key, "
        "execution_family, execution_key, timestamp, gross_pips, "
        "benchmark_family, lookback, net_mean, net_p75, "
        "net_p90, net_p95, module_candidate) "
        "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
    )
    pending = []

    def flush_pending():
        if not pending:
            return
        try:
            database.executemany(insert_sql, pending)
        except sqlite3.IntegrityError as error:
            raise DiscoveryAtlasError("duplicate trade identity") from error
        pending.clear()

    event_timestamps = dict(
        database.execute("SELECT candidate_event_id, timestamp FROM events")
    )
    for directory, _manifest in records:
        for row in _jsonl(directory / "trades.jsonl"):
            if not row.get("complete"):
                continue
            timestamp = event_timestamps.get(row.get("candidate_event_id"))
            if timestamp is None or row.get("instrument") != reference["instrument"]:
                raise DiscoveryAtlasError("trade provenance mismatch")
            row.update(
                timestamp=datetime.fromisoformat(timestamp),
                role=role,
                process_spec_id=provenance[2],
                execution_key=(
                    row["instrument"],
                    timestamp,
                    row["session"],
                    row["direction"],
                    row["entry_mode"],
                ),
            )
            cell = tuple(row.get(field) for field in CELL_FIELDS)
            setup = cell[: len(SETUP_FIELDS)]
            execution = tuple(row[field] for field in EXECUTION_FIELDS)
            scenario_values = [
                _scenario_values([row], profile, statistic, slippage)[0]
                for statistic, slippage in HEADLINE_COSTS
            ]
            pending.append(
                (
                    role,
                    run_key,
                    row["candidate_event_id"],
                    _json(cell),
                    _json(setup),
                    _json(execution),
                    _json(row["execution_key"]),
                    timestamp,
                    float(row["gross_return_pips_adverse_first"]),
                    row["benchmark_family"],
                    row["lookback"],
                    *scenario_values,
                    int(_module_candidate(row)),
                )
            )
            if len(pending) == 1000:
                flush_pending()
            trade_count += 1
    flush_pending()
    database.commit()
    LOGGER.info(
        "authenticated run %d: shards=%d candidate_rows=%d trade_rows_spooled=%d",
        run_number,
        len(records),
        candidate_count,
        trade_count,
    )
    return reference, commitments, candidate_count, trade_count


def _time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DiscoveryAtlasError("timestamps must be timezone-aware")
    parsed = parsed.astimezone(UTC)
    if parsed.year != 2024:
        raise DiscoveryAtlasError("row outside frozen 2024 period")
    return parsed


def _collect(
    directories, registry, registry_sha, role, database, profile, run_offset=0
):
    directories = tuple(map(Path, directories))
    for directory in directories:
        _reject_sealed(directory)
    manifests = []
    for directory in directories:
        path = directory / "shard-manifest.json"
        if not path.is_file():
            raise DiscoveryAtlasError("every raw input requires shard-manifest.json")
        manifests.append((directory, json.loads(path.read_text())))
    logical = defaultdict(list)
    for directory, manifest in manifests:
        logical[_manifest_identity(manifest)].append((directory, manifest))
    commitments, provenance = [], set()
    candidates = trades = 0
    for number, records in enumerate(logical.values(), run_offset + 1):
        manifest, run_commitments, run_candidates, run_trades = _authenticate_run(
            records, registry, registry_sha, role, database, number, profile
        )
        provenance.add(
            tuple(
                manifest.get(field)
                for field in ("filter_family", "filter_spec_id", "process_spec_id")
            )
        )
        commitments.extend(run_commitments)
        candidates += run_candidates
        trades += run_trades
    return commitments, provenance, candidates, trades, len(logical)


def _write(path, rows):
    fields = (
        sorted({field for row in rows for field in row}) if rows else ["schema_version"]
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _pf(values):
    gains, losses = sum(v for v in values if v > 0), -sum(v for v in values if v < 0)
    return gains / losses if losses else (math.inf if gains else 0.0)


def _cell_metrics(rows, values):
    ordered = sorted(
        zip(rows, values, strict=True),
        key=lambda item: (item[0]["timestamp"], item[0]["candidate_event_id"]),
    )
    monthly, observations = defaultdict(float), defaultdict(int)
    streak = longest = 0
    for row, value in ordered:
        monthly[row["timestamp"].month] += value
        observations[row["timestamp"].month] += 1
        streak = streak + 1 if value < 0 else 0
        longest = max(longest, streak)
    month_values = [monthly.get(month, 0.0) for month in range(1, 13)]
    return {
        "trade_count": len(rows),
        "trades_per_month": len(rows) / 12,
        "candidate_count": len({row["candidate_event_id"] for row in rows}),
        "expectancy": statistics.fmean(values),
        "total_pips": sum(values),
        "profit_factor": _pf(values),
        "win_rate": sum(v > 0 for v in values) / len(values),
        "positive_months": sum(monthly[m] > 0 for m in observations),
        "negative_months": sum(monthly[m] < 0 for m in observations),
        "no_trade_months": 12 - len(observations),
        "worst_month": min(month_values),
        "best_month": max(month_values),
        "monthly_mean": statistics.fmean(month_values),
        "monthly_standard_deviation": statistics.pstdev(month_values),
        "max_losing_streak": longest,
    }


def _scenario_values(rows, profile, statistic, slippage):
    result = []
    for row in rows:
        spread, commission = profile.costs(row["instrument"], row["session"], statistic)
        result.append(
            net_pips(
                float(row["gross_return_pips_adverse_first"]),
                row["instrument"],
                spread,
                slippage,
                commission,
            )
        )
    return result


def _module_candidate(row):
    return (
        row["signal_timeframe"] == "15m"
        and row["session"] == "london"
        and row["direction"] == "SHORT"
        and row["benchmark_family"] in ("vwap", "vwap-canonical-m1")
        and row["lookback"] in (20, 40)
        and row["entry_mode"] == "immediate"
    )


def _create_spool(path):
    database = sqlite3.connect(path)
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute(
        "CREATE TABLE events(candidate_event_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL)"
    )
    database.execute(
        "CREATE TABLE trades(seq INTEGER PRIMARY KEY, role TEXT NOT NULL, run_key TEXT NOT NULL, "
        "candidate_id TEXT NOT NULL, cell_key TEXT NOT NULL, setup_key TEXT NOT NULL, "
        "execution_family TEXT NOT NULL, execution_key TEXT NOT NULL, "
        "timestamp TEXT NOT NULL, "
        "gross_pips REAL NOT NULL, benchmark_family TEXT NOT NULL, lookback NOT NULL, "
        "net_mean REAL NOT NULL, net_p75 REAL NOT NULL, net_p90 REAL NOT NULL, "
        "net_p95 REAL NOT NULL, "
        "module_candidate INTEGER NOT NULL, "
        "UNIQUE(role, run_key, candidate_id, cell_key))"
    )

    return database


def _index_spool(database):
    """Add read-path indexes after bulk ingestion has completed."""
    database.execute(
        "CREATE INDEX trades_cell ON trades(role, cell_key, timestamp, candidate_id)"
    )
    database.execute(
        "CREATE INDEX trades_setup ON trades(role, setup_key, execution_key)"
    )
    database.execute(
        "CREATE INDEX trades_execution ON trades(role, execution_family, execution_key)"
    )
    database.execute(
        "CREATE INDEX trades_overlap ON trades(role, module_candidate, execution_key)"
    )


class _ExactFloatSum:
    """Incremental, correctly rounded equivalent of ``math.fsum`` for finite values."""

    def __init__(self):
        self.numerator = 0
        self.exponent = 0

    def add(self, value):
        numerator, denominator = float(value).as_integer_ratio()
        exponent = denominator.bit_length() - 1
        if exponent > self.exponent:
            self.numerator <<= exponent - self.exponent
            self.exponent = exponent
        self.numerator += numerator << (self.exponent - exponent)

    def value(self):
        return float(Fraction(self.numerator, 1 << self.exponent))


def _new_cell_metric_states():
    scenarios = ("gross", *(statistic for statistic, _ in HEADLINE_COSTS))
    return scenarios, {
        name: {
            "exact_sum": _ExactFloatSum(),
            "total": 0.0,
            "gains": 0.0,
            "losses": 0.0,
            "wins": 0,
            "monthly": defaultdict(float),
            "observed": set(),
            "streak": 0,
            "longest": 0,
        }
        for name in scenarios
    }


def _finalize_cell_metrics(scenarios, states, count, candidate_count):
    results = {}
    for name in scenarios:
        state = states[name]
        monthly = state["monthly"]
        observed = state["observed"]
        month_values = [monthly.get(month, 0.0) for month in range(1, 13)]
        losses = state["losses"]
        gains = state["gains"]
        results[name] = {
            "trade_count": count,
            "trades_per_month": count / 12,
            "candidate_count": candidate_count,
            "expectancy": state["exact_sum"].value() / count,
            "total_pips": state["total"],
            "profit_factor": gains / losses if losses else (math.inf if gains else 0.0),
            "win_rate": state["wins"] / count,
            "positive_months": sum(monthly[m] > 0 for m in observed),
            "negative_months": sum(monthly[m] < 0 for m in observed),
            "no_trade_months": 12 - len(observed),
            "worst_month": min(month_values),
            "best_month": max(month_values),
            "monthly_mean": statistics.fmean(month_values),
            "monthly_standard_deviation": statistics.pstdev(month_values),
            "max_losing_streak": state["longest"],
        }
    return results


def _stream_spooled_cell_metrics(database, *, started=None, progress_rows=1_000_000):
    """Yield every evidence cell from one chronological, bounded-memory cursor."""
    scan_started = started if started is not None else time.monotonic()
    query = (
        "SELECT cell_key, candidate_id, timestamp, gross_pips, net_mean, net_p75, "
        "net_p90, net_p95 FROM trades WHERE role='evidence' "
        "ORDER BY cell_key, timestamp, candidate_id"
    )
    current_key = None
    scenarios = states = None
    candidates = set()
    count = rows_processed = cells_finalized = 0
    for row in database.execute(query):
        cell_key, candidate_id, timestamp, *values = row
        if current_key is not None and cell_key != current_key:
            cells_finalized += 1
            yield (
                current_key,
                _finalize_cell_metrics(scenarios, states, count, len(candidates)),
            )
            scenarios, states = _new_cell_metric_states()
            candidates = set()
            count = 0
        elif current_key is None:
            scenarios, states = _new_cell_metric_states()
        current_key = cell_key
        candidates.add(candidate_id)
        count += 1
        rows_processed += 1
        month = datetime.fromisoformat(timestamp).month
        for name, value in zip(scenarios, values, strict=True):
            state = states[name]
            state["exact_sum"].add(value)
            state["total"] += value
            if value > 0:
                state["gains"] += value
                state["wins"] += 1
            elif value < 0:
                state["losses"] -= value
            state["monthly"][month] += value
            state["observed"].add(month)
            state["streak"] = state["streak"] + 1 if value < 0 else 0
            state["longest"] = max(state["longest"], state["streak"])
        if progress_rows and rows_processed % progress_rows == 0:
            LOGGER.info(
                "cell_rows_processed=%d cells_finalized=%d elapsed_seconds=%.3f",
                rows_processed,
                cells_finalized,
                time.monotonic() - scan_started,
            )
    if current_key is not None:
        cells_finalized += 1
        yield (
            current_key,
            _finalize_cell_metrics(scenarios, states, count, len(candidates)),
        )
    LOGGER.info(
        "cell_rows_processed=%d cells_finalized=%d elapsed_seconds=%.3f",
        rows_processed,
        cells_finalized,
        time.monotonic() - scan_started,
    )


def _stream_setup_counts(database):
    """Yield setup-level distinct counts while retaining only one setup's keys."""
    query = (
        "SELECT setup_key, candidate_id, execution_key FROM trades "
        "WHERE role='evidence' ORDER BY setup_key, execution_key, candidate_id"
    )
    current = None
    candidates, executions = set(), set()
    for setup_key, candidate_id, execution_key in database.execute(query):
        if current is not None and setup_key != current:
            yield current, len(candidates), len(executions)
            candidates, executions = set(), set()
        current = setup_key
        candidates.add(candidate_id)
        executions.add(execution_key)
    if current is not None:
        yield current, len(candidates), len(executions)


def build_atlas(
    stage4b_dirs,
    output_dir,
    profile_path,
    registry_path=Path("configs/stage4a-2024-corpus-registry.json"),
    *,
    module_a_dirs=(),
    spool_dir=None,
):
    started = time.monotonic()
    stage4b_dirs, module_a_dirs = tuple(stage4b_dirs), tuple(module_a_dirs)
    for path in (*stage4b_dirs, *module_a_dirs):
        _reject_sealed(path)
    registry_path, profile_path = Path(registry_path), Path(profile_path)
    registry_sha, profile_sha = _sha(registry_path), _sha(profile_path)
    registry, profile = _load_registry(registry_path), CostProfile.load(profile_path)
    if spool_dir is not None:
        spool_dir = Path(spool_dir)
        _reject_sealed(spool_dir)
        spool_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="discovery-atlas-", dir=spool_dir
    ) as temporary:
        database = _create_spool(Path(temporary) / "atlas.sqlite3")
        try:
            commitments, evidence_provenance, candidate_count, trade_count, runs = (
                _collect(
                    stage4b_dirs,
                    registry,
                    registry_sha,
                    "evidence",
                    database,
                    profile,
                )
            )
            (
                module_commitments,
                module_provenance,
                module_candidates,
                module_trades,
                module_runs,
            ) = (
                _collect(
                    module_a_dirs,
                    registry,
                    registry_sha,
                    "module_a",
                    database,
                    profile,
                    runs,
                )
                if module_a_dirs
                else ([], set(), 0, 0, 0)
            )
            commitments.extend(module_commitments)
            _index_spool(database)
            LOGGER.info(
                "authenticated_shards=%d candidate_rows_processed=%d "
                "trade_rows_spooled=%d elapsed_seconds=%.3f",
                len(commitments),
                candidate_count + module_candidates,
                trade_count + module_trades,
                time.monotonic() - started,
            )
            if not trade_count:
                raise DiscoveryAtlasError("no complete authenticated evidence trades")
            if module_trades:
                spec = frozen_ou_eligibility_spec("frozen-ou-crossasset-v1")
                expected_module = (
                    "ornstein-uhlenbeck",
                    spec.filter_spec_id,
                    spec.process_spec.process_spec_id,
                )
                if module_provenance != {expected_module}:
                    raise DiscoveryAtlasError(
                        "Module A role requires exact frozen-ou-crossasset-v1 provenance"
                    )

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            cell_keys = [
                row[0]
                for row in database.execute(
                    "SELECT DISTINCT cell_key FROM trades WHERE role='evidence' ORDER BY cell_key"
                )
            ]
            cost_rows, cell_results = [], {}
            for position, (encoded_key, metric_bundle) in enumerate(
                _stream_spooled_cell_metrics(database, started=started), 1
            ):
                key = tuple(json.loads(encoded_key))
                gross_metrics = metric_bundle["gross"]
                for statistic, slippage in HEADLINE_COSTS:
                    scenario = f"{statistic}+{slippage:g}"
                    metrics = metric_bundle[statistic]
                    cell_results[(encoded_key, scenario)] = metrics
                    cost_rows.append(
                        dict(zip(CELL_FIELDS, key, strict=True))
                        | {
                            "cost_scenario": scenario,
                            "spread_statistic": statistic,
                            "slippage_pips": slippage,
                            **{
                                f"gross_{name}": value
                                for name, value in gross_metrics.items()
                            },
                            **{f"net_{name}": value for name, value in metrics.items()},
                            **metrics,
                        }
                    )
                if position == 1 or position % 100 == 0 or position == len(cell_keys):
                    LOGGER.info(
                        "cell_groups_processed=%d/%d elapsed_seconds=%.3f",
                        position,
                        len(cell_keys),
                        time.monotonic() - started,
                    )
            _write(output_dir / OUTPUTS[3], cost_rows)

            family_cells = defaultdict(list)
            for encoded in cell_keys:
                key = tuple(json.loads(encoded))
                family_cells[_json(key[: len(SETUP_FIELDS)])].append(encoded)
            setup_counts = {
                key: (candidates, opportunities)
                for key, candidates, opportunities in _stream_setup_counts(database)
            }
            setup_rows = []
            for encoded_family, exits in sorted(family_cells.items()):
                family = tuple(json.loads(encoded_family))
                candidates, opportunities = setup_counts[encoded_family]
                base = dict(zip(SETUP_FIELDS, family, strict=True)) | {
                    "exit_cell_count": len(exits),
                    "candidate_count": candidates,
                    "unique_execution_opportunities": opportunities,
                    "trades_per_month": opportunities / 12,
                }
                for statistic, slippage in HEADLINE_COSTS:
                    scenario = f"{statistic}+{slippage:g}"
                    expectancies = [
                        cell_results[(key, scenario)]["expectancy"] for key in exits
                    ]
                    positive = sum(value > 0 for value in expectancies)
                    prefix = scenario.replace("+", "_").replace(".", "p")
                    base.update(
                        {
                            f"{prefix}_positive_exit_cells": positive,
                            f"{prefix}_positive_fraction": positive / len(expectancies),
                            f"{prefix}_min_expectancy": min(expectancies),
                            f"{prefix}_median_expectancy": statistics.median(
                                expectancies
                            ),
                            f"{prefix}_max_expectancy": max(expectancies),
                        }
                    )
                setup_rows.append(base)
            _write(output_dir / OUTPUTS[0], setup_rows)

            execution_rows = []
            for (encoded,) in database.execute(
                "SELECT DISTINCT execution_family FROM trades WHERE role='evidence' ORDER BY execution_family"
            ):
                key = tuple(json.loads(encoded))
                where = "role='evidence' AND execution_family=?"
                parameters = (encoded,)
                execution_rows.append(
                    dict(zip(EXECUTION_FIELDS, key, strict=True))
                    | {
                        "unique_execution_opportunities": database.execute(
                            f"SELECT COUNT(DISTINCT execution_key) FROM trades WHERE {where}",
                            parameters,
                        ).fetchone()[0],
                        "family_membership_count": database.execute(
                            f"SELECT COUNT(DISTINCT setup_key) FROM trades WHERE {where}",
                            parameters,
                        ).fetchone()[0],
                        "benchmark_variants_firing": "|".join(
                            row[0]
                            for row in database.execute(
                                f"SELECT DISTINCT benchmark_family FROM trades WHERE {where} ORDER BY 1",
                                parameters,
                            )
                        ),
                        "lookbacks_firing": "|".join(
                            str(row[0])
                            for row in database.execute(
                                f"SELECT DISTINCT lookback FROM trades WHERE {where} ORDER BY 1",
                                parameters,
                            )
                        ),
                    }
                )
            _write(output_dir / OUTPUTS[1], execution_rows)

            hypotheses = defaultdict(dict)
            for encoded in cell_keys:
                cell = tuple(json.loads(encoded))
                hypothesis = tuple(
                    cell[CELL_FIELDS.index(f)] for f in HYPOTHESIS_FIELDS
                )
                hypotheses[_json(hypothesis)][cell[0]] = encoded
            cross_rows, shortlist = [], []
            for index, encoded_hypothesis in enumerate(sorted(hypotheses), 1):
                hypothesis = tuple(json.loads(encoded_hypothesis))
                instruments = hypotheses[encoded_hypothesis]
                scenario_positive, per_instrument = {}, {}
                for instrument, cell in sorted(instruments.items()):
                    per_instrument[instrument] = {}
                    for statistic, slippage in HEADLINE_COSTS:
                        scenario = f"{statistic}+{slippage:g}"
                        expectancy = cell_results[(cell, scenario)]["expectancy"]
                        per_instrument[instrument][scenario] = expectancy
                        scenario_positive[scenario] = scenario_positive.get(
                            scenario, 0
                        ) + (expectancy > 0)
                tested = len(instruments)
                base = dict(zip(HYPOTHESIS_FIELDS, hypothesis, strict=True))
                for instrument, evidence_by_cost in per_instrument.items():
                    count = database.execute(
                        "SELECT COUNT(*) FROM trades WHERE role='evidence' AND cell_key=?",
                        (instruments[instrument],),
                    ).fetchone()[0]
                    row = base | {
                        "instrument": instrument,
                        "instrument_observation_count": count,
                        "instruments_tested": tested,
                        "instruments_with_sufficient_observations": "not_assessed_no_preregistered_threshold",
                        "cross_asset_support": _json(per_instrument),
                    }
                    for scenario, positive in scenario_positive.items():
                        prefix = scenario.replace("+", "_").replace(".", "p")
                        row[f"{prefix}_instruments_positive"] = positive
                        row[f"{prefix}_instruments_negative_or_zero"] = (
                            tested - positive
                        )
                        row[f"{prefix}_instrument_expectancy"] = evidence_by_cost[
                            scenario
                        ]
                    cross_rows.append(row)
                shortlist.append(
                    {
                        "hypothesis_id": f"H{index:04d}",
                        **base,
                        "evidence_status": "descriptive_only_no_preregistered_triage_thresholds",
                        "cross_asset_support": _json(per_instrument),
                        "frequency": "see cross-asset-hypotheses.csv",
                        "positive_months": "see cost-robustness.csv",
                        "plateau_breadth": "see setup-family-matrix.csv",
                        "cost_survival": _json(scenario_positive),
                        "module_a_overlap": "see overlap-with-module-a.csv",
                    }
                )
            _write(output_dir / OUTPUTS[2], cross_rows)
            _write(output_dir / OUTPUTS[5], shortlist)

            overlap_rows = []
            actual_available = bool(module_trades)
            for encoded_family in sorted(family_cells):
                family = tuple(json.loads(encoded_family))

                def overlap_count(role, negate=False, family_key=encoded_family):
                    operator = "NOT EXISTS" if negate else "EXISTS"
                    return database.execute(
                        "SELECT COUNT(DISTINCT family.execution_key) FROM trades family "
                        "WHERE family.role='evidence' AND family.setup_key=? AND "
                        f"{operator} (SELECT 1 FROM trades compared WHERE "
                        "compared.role=? AND compared.module_candidate=1 AND "
                        "compared.execution_key=family.execution_key)",
                        (family_key, role),
                    ).fetchone()[0]

                opportunities = database.execute(
                    "SELECT COUNT(DISTINCT execution_key) FROM trades "
                    "WHERE role='evidence' AND setup_key=?",
                    (encoded_family,),
                ).fetchone()[0]
                raw_intersection = overlap_count("evidence")
                module_intersection = overlap_count("module_a")
                unique_setup = overlap_count("module_a", negate=True)
                module_only = database.execute(
                    "SELECT COUNT(DISTINCT module.execution_key) FROM trades module "
                    "WHERE module.role='module_a' AND module.module_candidate=1 AND "
                    "NOT EXISTS (SELECT 1 FROM trades family WHERE family.role='evidence' "
                    "AND family.setup_key=? AND family.execution_key=module.execution_key)",
                    (encoded_family,),
                ).fetchone()[0]
                overlap_rows.append(
                    dict(zip(SETUP_FIELDS, family, strict=True))
                    | {
                        "raw_vwap_signal_family_intersection_count": raw_intersection,
                        "raw_vwap_signal_family_overlap_rate": raw_intersection
                        / opportunities,
                        "actual_frozen_module_a_available": actual_available,
                        "actual_frozen_module_a_intersection_count": module_intersection,
                        "actual_frozen_module_a_overlap_rate": module_intersection
                        / opportunities,
                        "unique_to_setup_vs_module_a": unique_setup,
                        "module_a_only_count": module_only,
                    }
                )
            _write(output_dir / OUTPUTS[4], overlap_rows)
            (output_dir / OUTPUTS[6]).write_text(
                "# Frozen 2024 discovery atlas\n\nExact cells retain performance. Families report exit-plateau evidence; execution deduplication reports opportunities only. No automatic triage threshold or ranking is applied.\n"
            )
            audit = {
                "schema_version": SCHEMA,
                "atlas_methodology_version": METHODOLOGY,
                "research_period": {"start": FROZEN_START, "end": FROZEN_END},
                "stage4b_methodology_id": STAGE4B_METHODOLOGY_ID,
                "registry_sha256": registry_sha,
                "cost_profile_sha256": profile_sha,
                "input_shards": commitments,
                "filter_process_provenance": sorted(
                    (
                        *[(a, b, c, "evidence") for a, b, c in evidence_provenance],
                        *[(a, b, c, "module_a") for a, b, c in module_provenance],
                    )
                ),
                "output_sha256": {name: _sha(output_dir / name) for name in OUTPUTS},
            }
            (output_dir / "execution-audit.json").write_text(_json(audit) + "\n")
            return audit
        finally:
            database.close()


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(
        description="Report authenticated frozen-2024 Stage 4 families"
    )
    parser.add_argument(
        "--stage4b-dir",
        action="append",
        type=Path,
        required=True,
        help="complete baseline/evidence shard set (repeat per shard)",
    )
    parser.add_argument(
        "--module-a-dir",
        action="append",
        type=Path,
        default=[],
        help="separate frozen-OU Module A shard set used only for overlap",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--spool-dir",
        type=Path,
        help="parent directory for the automatically cleaned temporary SQLite spool",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("configs/stage4a-2024-corpus-registry.json"),
    )
    parser.add_argument(
        "--cost-profile",
        type=Path,
        default=Path("configs/stage4c-ftmo-cost-profile-v1.json"),
    )
    args = parser.parse_args(argv)
    build_atlas(
        args.stage4b_dir,
        args.output_dir,
        args.cost_profile,
        args.registry,
        module_a_dirs=args.module_a_dir,
        spool_dir=args.spool_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
