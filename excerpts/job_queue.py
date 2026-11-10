"""Postgres-backed job queue: idempotent enqueue, SKIP LOCKED claims, leases and retries.

Excerpt from the Video Studio module `studio.jobs.queue` (plus the lease keeper from `studio.jobs.worker`
and the `jobs` table from the schema).

Every pipeline step (research, script, factcheck, voice, render, stitch, archive, ...) is one row in
`jobs`. Workers are systemd template units (`worker@1`, `worker@2`, ...) that loop on `claim()`.

Removed for the excerpt: event emission (`emit(...)` calls after each state change), secret redaction
of stored error text, `has_pending`, `hold`, the `_stale` helper that logs a refused late result, and the lease keeper's
fallback when its own connection cannot be opened (the original then runs the job without renewals).
Logic and SQL are otherwise unchanged. Needs psycopg 3 and a Postgres database; no unit test here.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from uuid import UUID

import psycopg
from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  type text NOT NULL,
  video_id uuid REFERENCES videos(id) ON DELETE CASCADE,
  scene_id uuid REFERENCES scenes(id) ON DELETE CASCADE,
  inputs jsonb NOT NULL DEFAULT '{}',
  output jsonb,
  idempotency_key text UNIQUE NOT NULL,
  status text NOT NULL DEFAULT 'queued'
    CHECK (status IN ('queued','running','done','failed','held','cancelled')),
  attempts int NOT NULL DEFAULT 0,
  max_attempts int NOT NULL DEFAULT 3,
  error text,
  priority int NOT NULL DEFAULT 100,
  run_after timestamptz NOT NULL DEFAULT now(),
  lease_until timestamptz,
  worker_id text,
  started_at timestamptz,
  finished_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS jobs_claim_idx ON jobs (status, priority, run_after);
"""


@dataclass(frozen=True)
class Job:
    """A claimed unit of work (`attempts` is 1 on the first run)."""

    id: UUID
    type: str
    video_id: UUID | None
    scene_id: UUID | None
    inputs: dict
    attempts: int
    max_attempts: int


def enqueue(conn: psycopg.Connection, type: str, *, idempotency_key: str, inputs: dict | None = None,
            video_id: UUID | None = None, scene_id: UUID | None = None, priority: int = 100,
            delay_s: int = 0, max_attempts: int = 3) -> UUID:
    """Queue a job once per idempotency key; a duplicate returns the existing job's id.

    A stage that fans out (one render job per shot, say) can be re-run after a crash and enqueue the
    same keys again without creating duplicate work or duplicate spend.
    """
    row = conn.execute(
        "INSERT INTO jobs (type, idempotency_key, inputs, video_id, scene_id, priority, run_after, max_attempts)"
        " VALUES (%s,%s,%s,%s,%s,%s, now() + make_interval(secs => %s), %s)"
        " ON CONFLICT (idempotency_key) DO NOTHING RETURNING id",
        (type, idempotency_key, Jsonb(inputs or {}), video_id, scene_id, priority, delay_s, max_attempts),
    ).fetchone()
    if row is None:
        existing = conn.execute("SELECT id FROM jobs WHERE idempotency_key=%s", (idempotency_key,)).fetchone()
        return existing["id"]
    return row["id"]


def claim(conn: psycopg.Connection, worker_id: str, *, types: list[str] | None = None,
          lease_s: int = 1800) -> Job | None:
    """Atomically claim the next due job (lowest priority number first) and lease it.

    `FOR UPDATE SKIP LOCKED` lets any number of workers poll the same table: a row another worker is
    claiming is skipped instead of waited on, so two workers never get the same job.
    """
    with conn.transaction():
        row = conn.execute(
            "SELECT id FROM jobs WHERE status='queued' AND run_after <= now()"
            " AND (%s::text[] IS NULL OR type = ANY(%s::text[]))"
            " ORDER BY priority, run_after, created_at LIMIT 1 FOR UPDATE SKIP LOCKED",
            (types, types),
        ).fetchone()
        if row is None:
            return None
        job = conn.execute(
            "UPDATE jobs SET status='running', attempts=attempts+1, worker_id=%s, started_at=now(),"
            " lease_until=now() + make_interval(secs => %s) WHERE id=%s"
            " RETURNING id, type, video_id, scene_id, inputs, attempts, max_attempts",
            (worker_id, lease_s, row["id"]),
        ).fetchone()
    return Job(**job)


# Fence for every result-reporting call: only the current run of a running job may change it.
# A worker whose lease lapsed (and whose job was re-claimed as attempt n+1) can still finish late;
# its complete()/fail() then matches no row and changes nothing.
_OWNED = " WHERE id=%(id)s AND status='running' AND (%(attempt)s::int IS NULL OR attempts=%(attempt)s::int)"
_RETRY = "(%(retryable)s AND attempts < max_attempts)"
_FAIL_SQL = (
    f"UPDATE jobs SET status=CASE WHEN {_RETRY} THEN 'queued' ELSE 'failed' END,"
    " error=%(error)s, lease_until=NULL, worker_id=NULL,"
    f" run_after=CASE WHEN {_RETRY} THEN now() + make_interval(secs => 30 * power(2, attempts - 1))"
    " ELSE now() END,"
    f" finished_at=CASE WHEN {_RETRY} THEN NULL ELSE now() END"
    + _OWNED
    + " RETURNING status, attempts, video_id, scene_id, type"
)


def heartbeat(conn: psycopg.Connection, job_id: UUID, lease_s: int = 1800, *, attempt: int | None = None) -> None:
    """Extend a running job's lease (no-op if the job is no longer running on `attempt`)."""
    conn.execute(
        "UPDATE jobs SET lease_until=now() + make_interval(secs => %(lease)s)" + _OWNED + " RETURNING id",
        {"lease": lease_s, "id": job_id, "attempt": attempt},
    )


def complete(conn: psycopg.Connection, job_id: UUID, output: dict | None = None, *,
             attempt: int | None = None) -> None:
    """Mark a running job done with optional output; a stale caller changes nothing."""
    conn.execute(
        "UPDATE jobs SET status='done', output=%(output)s, finished_at=now(), lease_until=NULL"
        + _OWNED + " RETURNING video_id, scene_id, type",
        {"output": Jsonb(output or {}), "id": job_id, "attempt": attempt},
    )


def fail(conn: psycopg.Connection, job_id: UUID, error: str, *, retryable: bool = True,
         attempt: int | None = None) -> str:
    """Record a failure in one statement; requeue with exponential backoff while attempts remain.

    Returns "queued" if it will retry, else "failed". A stale caller gets "queued" and changes nothing,
    so it never fails work it no longer owns.
    """
    params = {"retryable": retryable, "error": error[:2000], "id": job_id, "attempt": attempt}
    row = conn.execute(_FAIL_SQL, params).fetchone()
    return "queued" if row is None else row["status"]


def recover_expired(conn: psycopg.Connection) -> int:
    """Requeue running jobs whose lease expired (worker died, power cut); fail those out of attempts."""
    rows = conn.execute(
        "UPDATE jobs SET status=CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,"
        " error=CASE WHEN attempts >= max_attempts THEN 'lease expired' ELSE error END,"
        " finished_at=CASE WHEN attempts >= max_attempts THEN now() ELSE finished_at END,"
        " worker_id=NULL, lease_until=NULL"
        " WHERE status='running' AND lease_until < now()"
        " RETURNING id, status, video_id, scene_id, type"
    ).fetchall()
    return sum(r["status"] == "queued" for r in rows)


# --- lease keeper (from studio.jobs.worker) -------------------------------------------------------
# Long renders (a headless Blender shot, a full stitch) can outlive the lease. A daemon thread renews it
# on its OWN connection, so a renewal never joins, or gets rolled back with, a transaction the stage
# handler holds open on the worker's shared connection.

def _renew_lease(conn, job: Job, lease_s: int, every_s: float, stop: threading.Event) -> None:
    """Thread body: extend the lease every `every_s` seconds until `stop` is set, then close `conn`."""
    with conn:
        while not stop.wait(every_s):
            try:
                heartbeat(conn, job.id, lease_s, attempt=job.attempts)
            except Exception:  # noqa: BLE001 - a renewal failure must never kill the worker
                pass  # (original records a worker.lease_error event here)


@contextmanager
def lease_keeper(open_conn, job: Job, *, lease_s: int = 1800, every_s: float = 600):
    """Keep a running job's lease alive in a daemon thread for the duration of the block."""
    keeper_conn = open_conn()
    stop = threading.Event()
    keeper = threading.Thread(target=_renew_lease, args=(keeper_conn, job, lease_s, every_s, stop),
                              name=f"lease-{job.id}", daemon=True)
    keeper.start()
    try:
        yield
    finally:
        stop.set()
        keeper.join(timeout=60)
