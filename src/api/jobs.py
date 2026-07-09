"""Lightweight in-process job orchestration for the analysis pipeline.

The API process owns the jobs intentionally: this keeps local deployments and
the demo dashboard dependency-free.  The public snapshot model makes the
lifecycle observable while keeping subprocess handles private.
"""
from __future__ import annotations

import asyncio
import uuid
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence


CommandRunner = Callable[[Sequence[str], Path], Awaitable[Dict[str, Any]]]
TERMINAL_STATUSES = {"succeeded", "failed", "cancelled"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass
class PipelineJob:
    id: str
    target_date: str
    skip_scrape: bool
    skip_ai: bool
    status: str = "queued"
    created_at: str = ""
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    returncode: Optional[int] = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    error: Optional[str] = None

    def snapshot(self) -> Dict[str, Any]:
        return asdict(self)


async def run_subprocess(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
    """Run one pipeline command without a shell and retain bounded diagnostics."""
    process = await asyncio.create_subprocess_exec(
        *command,
        cwd=str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await process.communicate()
    except asyncio.CancelledError:
        if process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=3)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
        raise
    return {
        "returncode": process.returncode,
        "stdout_tail": (stdout or b"")[-4000:].decode("utf-8", errors="replace"),
        "stderr_tail": (stderr or b"")[-2000:].decode("utf-8", errors="replace"),
    }


class PipelineJobManager:
    """Bounded in-memory registry with explicit job lifecycle transitions."""

    def __init__(
        self,
        *,
        runner: CommandRunner = run_subprocess,
        max_jobs: int = 50,
        max_concurrent: int = 1,
    ) -> None:
        self._runner = runner
        self._max_jobs = max(1, max_jobs)
        self._semaphore = asyncio.Semaphore(max(1, max_concurrent))
        self._jobs: "OrderedDict[str, PipelineJob]" = OrderedDict()
        self._tasks: Dict[str, asyncio.Task[None]] = {}

    def submit(
        self,
        *,
        target_date: str,
        skip_scrape: bool,
        skip_ai: bool,
        command: Sequence[str],
        cwd: Path,
    ) -> Dict[str, Any]:
        self._prune()
        if len(self._jobs) >= self._max_jobs:
            raise RuntimeError("job queue is full")
        job = PipelineJob(
            id=uuid.uuid4().hex,
            target_date=target_date,
            skip_scrape=skip_scrape,
            skip_ai=skip_ai,
            created_at=_utc_now(),
        )
        self._jobs[job.id] = job
        task = asyncio.create_task(self._run(job, tuple(command), cwd))
        self._tasks[job.id] = task
        task.add_done_callback(lambda _: self._tasks.pop(job.id, None))
        return job.snapshot()

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        return job.snapshot() if job else None

    def list(self, *, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        jobs = reversed(self._jobs.values())
        if status is not None:
            jobs = (job for job in jobs if job.status == status)
        return [job.snapshot() for _, job in zip(range(limit), jobs)]

    def cancel(self, job_id: str) -> Optional[Dict[str, Any]]:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.status not in TERMINAL_STATUSES:
            job.status = "cancelled"
            job.finished_at = _utc_now()
            task = self._tasks.get(job_id)
            if task is not None:
                task.cancel()
        return job.snapshot()

    async def shutdown(self) -> None:
        active = list(self._tasks.values())
        for job_id in list(self._tasks):
            self.cancel(job_id)
        if active:
            await asyncio.gather(*active, return_exceptions=True)

    async def _run(
        self,
        job: PipelineJob,
        command: Sequence[str],
        cwd: Path,
    ) -> None:
        try:
            async with self._semaphore:
                if job.status == "cancelled":
                    return
                job.status = "running"
                job.started_at = _utc_now()
                result = await self._runner(command, cwd)
                job.returncode = int(result.get("returncode", 1))
                job.stdout_tail = str(result.get("stdout_tail", ""))
                job.stderr_tail = str(result.get("stderr_tail", ""))
                job.status = "succeeded" if job.returncode == 0 else "failed"
        except asyncio.CancelledError:
            job.status = "cancelled"
            raise
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
        finally:
            job.finished_at = _utc_now()

    def _prune(self) -> None:
        while len(self._jobs) >= self._max_jobs:
            removable = next(
                (job_id for job_id, job in self._jobs.items()
                 if job.status in TERMINAL_STATUSES),
                None,
            )
            if removable is None:
                break
            self._jobs.pop(removable, None)
