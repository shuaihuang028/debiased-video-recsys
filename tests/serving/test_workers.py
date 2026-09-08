from __future__ import annotations

from short_video_recsys.serving.workers import WorkerPool


class FakeWorker:
    def __init__(self, value: int):
        self.value = value
        self.closed = False

    def call(self, payload, timeout=30.0):
        return {"worker": self.value, "payload": payload, "timeout": timeout}

    def close(self):
        self.closed = True


def test_worker_pool_leases_and_closes_workers() -> None:
    first = FakeWorker(1)
    second = FakeWorker(2)
    pool = WorkerPool([first, second])

    assert pool.call({"x": 1})["worker"] == 1
    assert pool.call({"x": 2})["worker"] == 2
    pool.close()

    assert first.closed and second.closed
