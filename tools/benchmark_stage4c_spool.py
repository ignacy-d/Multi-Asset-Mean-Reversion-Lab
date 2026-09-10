"""Deterministic microbenchmark for the Stage 4C source/spool data path."""

import hashlib
import json
import random
import resource
import sqlite3
import tempfile
import time
from pathlib import Path

N = 300_000
G = 400
random.seed(417)
root = Path(tempfile.mkdtemp(prefix="s4c-bench-", dir="/tmp"))
src = root / "trades.jsonl"
order = list(range(N))
random.shuffle(order)
with src.open("wb") as f:
    for i in order:
        group = json.dumps([f"g{i % G:04}", "15m", i % 7], separators=(",", ":"))
        ident = json.dumps([group, f"e{i:08}"], separators=(",", ":"))
        row = json.dumps(
            {
                "candidate_event_id": f"e{i:08}",
                "complete": True,
                "group": group,
                "gross_return_pips_adverse_first": (i % 31) - 15,
                "padding": "x" * 160,
            },
            separators=(",", ":"),
        )
        f.write(json.dumps([group, ident, row], separators=(",", ":")).encode() + b"\n")


def baseline(db):
    t = time.perf_counter()
    with src.open("rb") as source_stream:
        hashlib.file_digest(source_stream, "sha256").hexdigest()
    passes = 1
    c = sqlite3.connect(db)
    c.execute("pragma journal_mode=off")
    c.execute("pragma synchronous=off")
    c.execute(
        "create table trades(group_key text not null,identity text primary key,"
        "row_json text not null) without rowid"
    )
    with src.open("rb") as f:
        for line in f:
            c.execute("insert into trades values(?,?,?)", json.loads(line))
    c.commit()
    passes += 1
    for _ in c.execute(
        "select group_key,row_json from trades order by group_key,identity"
    ):
        pass
    c.close()
    return time.perf_counter() - t, db.stat().st_size, passes


def optimized(db):
    t = time.perf_counter()
    h = hashlib.sha256()
    passes = 1
    c = sqlite3.connect(db)
    c.execute("pragma journal_mode=off")
    c.execute("pragma synchronous=off")
    c.execute(
        "create table trades(group_key text not null,identity text not null,"
        "row_json text not null,primary key(group_key,identity)) without rowid"
    )
    c.execute("begin")
    batch = []
    with src.open("rb") as f:
        for line in f:
            h.update(line)
            batch.append(json.loads(line))
            if len(batch) == 10000:
                c.executemany("insert into trades values(?,?,?)", batch)
                batch = []
    if batch:
        c.executemany("insert into trades values(?,?,?)", batch)
    c.commit()
    for _ in c.execute(
        "select group_key,row_json from trades order by group_key,identity"
    ):
        pass
    c.close()
    return time.perf_counter() - t, db.stat().st_size, passes


for name, fn in [("baseline", baseline), ("optimized", optimized)]:
    elapsed, size, passes = fn(root / f"{name}.sqlite3")
    throughput = N / elapsed
    size_mib = size / 1024 / 1024
    print(
        name,
        f"wall={elapsed:.3f}s rows/s={throughput:.0f} "
        f"size={size_mib:.1f}MiB passes={passes}",
    )
print(
    "source",
    f"{src.stat().st_size / 1024 / 1024:.1f}MiB",
    "peak_rss",
    f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.1f}MiB",
    "root",
    root,
)
