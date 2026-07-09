"""API contract tests for observable background pipeline jobs."""
from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Sequence

from fastapi.testclient import TestClient

from src.api import server
from src.api.jobs import PipelineJobManager


def _wait_for_status(
    client: TestClient,
    job_id: str,
    expected: str,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        response = client.get(f"/v6/jobs/{job_id}")
        assert response.status_code == 200
        job = response.json()
        if job["status"] == expected:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {expected}")


def test_run_endpoint_returns_accepted_job_and_tracks_success(monkeypatch) -> None:
    commands: list[Sequence[str]] = []

    async def successful_runner(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
        commands.append(command)
        await asyncio.sleep(0)
        return {"returncode": 0, "stdout_tail": "report ready", "stderr_tail": ""}

    manager = PipelineJobManager(runner=successful_runner)
    monkeypatch.setattr(server, "PIPELINE_JOBS", manager)

    with TestClient(server.app) as client:
        response = client.post(
            "/v6/run",
            json={"target_date": "2026-06-12", "skip_scrape": True, "skip_ai": True},
        )
        assert response.status_code == 202
        accepted = response.json()
        assert accepted["target_date"] == "2026-06-12"
        assert accepted["status"] in {"queued", "running", "succeeded"}
        assert len(accepted["id"]) == 32

        completed = _wait_for_status(client, accepted["id"], "succeeded")
        assert completed["returncode"] == 0
        assert completed["stdout_tail"] == "report ready"
        assert completed["started_at"] is not None
        assert completed["finished_at"] is not None

        listed = client.get("/v6/jobs", params={"status": "succeeded"}).json()
        assert [job["id"] for job in listed] == [accepted["id"]]
        assert "--skip-scrape" in commands[0]
        assert "--skip-ai" in commands[0]


def test_running_job_can_be_cancelled(monkeypatch) -> None:
    async def slow_runner(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
        await asyncio.Event().wait()
        return {"returncode": 0}

    manager = PipelineJobManager(runner=slow_runner)
    monkeypatch.setattr(server, "PIPELINE_JOBS", manager)

    with TestClient(server.app) as client:
        response = client.post("/v6/run", json={"target_date": "2026-06-12"})
        job_id = response.json()["id"]

        cancelled = client.delete(f"/v6/jobs/{job_id}")
        assert cancelled.status_code == 200
        assert cancelled.json()["status"] == "cancelled"
        assert _wait_for_status(client, job_id, "cancelled")["finished_at"] is not None


def test_job_manager_queues_resource_intensive_runs(monkeypatch) -> None:
    async def slow_runner(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
        await asyncio.Event().wait()
        return {"returncode": 0}

    manager = PipelineJobManager(runner=slow_runner, max_concurrent=1)
    monkeypatch.setattr(server, "PIPELINE_JOBS", manager)

    with TestClient(server.app) as client:
        first = client.post("/v6/run", json={"target_date": "2026-06-12"}).json()
        _wait_for_status(client, first["id"], "running")
        second = client.post("/v6/run", json={"target_date": "2026-06-13"}).json()

        assert client.get(f"/v6/jobs/{second['id']}").json()["status"] == "queued"
        assert client.delete(f"/v6/jobs/{second['id']}").json()["status"] == "cancelled"
        assert client.delete(f"/v6/jobs/{first['id']}").json()["status"] == "cancelled"


def test_job_endpoints_validate_filters_and_missing_ids(monkeypatch) -> None:
    manager = PipelineJobManager()
    monkeypatch.setattr(server, "PIPELINE_JOBS", manager)

    with TestClient(server.app) as client:
        assert client.get("/v6/jobs/missing").status_code == 404
        assert client.delete("/v6/jobs/missing").status_code == 404
        assert client.get("/v6/jobs", params={"status": "unknown"}).status_code == 422
        assert client.get("/v6/jobs", params={"limit": 0}).status_code == 422
        page = client.get("/ops")
        assert page.status_code == 200
        assert "Pipeline Operations" in page.text


def test_job_endpoint_rejects_work_when_registry_is_full(monkeypatch) -> None:
    async def slow_runner(command: Sequence[str], cwd: Path) -> Dict[str, Any]:
        await asyncio.Event().wait()
        return {"returncode": 0}

    manager = PipelineJobManager(runner=slow_runner, max_jobs=1)
    monkeypatch.setattr(server, "PIPELINE_JOBS", manager)

    with TestClient(server.app) as client:
        accepted = client.post("/v6/run", json={"target_date": "2026-06-12"})
        assert accepted.status_code == 202
        rejected = client.post("/v6/run", json={"target_date": "2026-06-13"})
        assert rejected.status_code == 429
        assert rejected.json()["detail"] == "job queue is full"
