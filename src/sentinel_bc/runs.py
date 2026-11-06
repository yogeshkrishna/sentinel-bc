"""Isolated local experiments and one cooperative training worker per registry."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import uuid

from .config import ROOT, load_config
from .data import load_training_data
from .inference import ActiveModel, Engine

TRAINING_LOCK = ROOT / "runtime/.studio-training.lock"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        for attempt in range(20):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                # Windows readers can briefly hold a file without delete sharing.
                # Keep the previous complete JSON visible and retry the rename.
                if attempt == 19:
                    raise
                time.sleep(.025)
    finally:
        temporary.unlink(missing_ok=True)


def pid_alive(pid):
    if os.name == "nt":
        import ctypes
        api = ctypes.windll.kernel32
        api.OpenProcess.restype = ctypes.c_void_p
        api.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        api.CloseHandle.argtypes = [ctypes.c_void_p]
        handle = api.OpenProcess(0x100000, False, int(pid))
        if not handle:
            return False
        try:
            return api.WaitForSingleObject(handle, 0) == 258
        finally:
            api.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class RunManager:
    def __init__(self, root=None, benchmark_artifacts=None):
        self.root = Path(root or ROOT / "runtime/training").resolve()
        for protected in (ROOT, ROOT / "artifacts", ROOT / "results"):
            if self.root == protected or protected != ROOT and protected in self.root.parents:
                raise ValueError("Choose a run registry outside the benchmark directories")
        self.root.mkdir(parents=True, exist_ok=True)
        self.benchmark = Path(benchmark_artifacts or ROOT / "artifacts/full").resolve()
        self.active = ActiveModel()
        self._mutex = threading.RLock()
        self.processes = {}
        self.notice = "No active model. Train a dataset or activate a completed run."
        saved = self.root / "active.json"
        preferred = json.loads(saved.read_text())["run_id"] if saved.exists() else "benchmark"
        try:
            self.activate(preferred)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            self.notice = f"No usable active model: {exc}"
            if preferred != "benchmark":
                try:
                    self.activate("benchmark")
                    self.notice = "Previous run unavailable. The preserved benchmark is active."
                except (ValueError, FileNotFoundError, KeyError):
                    pass

    def paths(self, run_id):
        if run_id == "benchmark":
            return self.benchmark, self.benchmark.parent.parent / "results" / self.benchmark.name
        directory = self.root / run_id
        if directory.parent != self.root or not run_id.startswith("run_"):
            raise ValueError("Unknown run ID")
        record = json.loads((directory / "run.json").read_text())
        return directory / "artifacts" / record["mode"], directory / "results" / record["mode"]

    def status(self, run_id):
        directory = self.root / run_id
        if directory.parent != self.root or not run_id.startswith("run_"):
            raise ValueError("Unknown run ID")
        path = directory / "run.json"
        record = json.loads(path.read_text())
        process = self.processes.get(run_id)
        if record["status"] in ("QUEUED", "RUNNING", "CANCELLING"):
            exited = process is not None and process.poll() is not None
            if exited or not pid_alive(record["pid"]):
                record.update(status="INTERRUPTED", phase="Worker interrupted", error="Training process ended before completion. Start a new run.", finished_at=now())
                atomic_json(path, record)
                self._release_lock(run_id)
        record["elapsed_seconds"] = round(time.time() - record["started_unix"], 1) if "finished_at" not in record else record.get("duration_seconds", 0)
        return record

    def _release_lock(self, run_id):
        lock = TRAINING_LOCK
        if lock.exists() and json.loads(lock.read_text())["run_id"] == run_id:
            lock.unlink()

    def start(self, mode="quick", data=None, seed=None, epochs=None):
        config = deepcopy(load_config())
        if mode not in ("quick", "full"):
            raise ValueError("Choose Quick or Full training")
        if seed is not None:
            if int(seed) != seed or not 0 <= seed <= 2147483000:
                raise ValueError("Random seed must be an integer between 0 and 2147483000")
            config["seed"] = int(seed)
        if epochs is not None:
            if int(epochs) != epochs or not 2 <= epochs <= 200:
                raise ValueError("Epoch budget must be an integer between 2 and 200")
            config[mode]["epochs"] = int(epochs)
        dataset = load_training_data(data, config, mode)
        with self._mutex:
            lock = TRAINING_LOCK
            lock.parent.mkdir(parents=True, exist_ok=True)
            if lock.exists():
                try:
                    owner = json.loads(lock.read_text())
                except json.JSONDecodeError as exc:
                    raise ValueError("Another training worker is starting. Try again shortly.") from exc
                if pid_alive(owner["pid"]):
                    raise ValueError("A training job is already running. Wait for it or request cancellation.")
                old_record = Path(owner["directory"]) / "run.json"
                if old_record.exists():
                    interrupted = json.loads(old_record.read_text())
                    if interrupted["status"] in ("QUEUED", "RUNNING", "CANCELLING"):
                        interrupted.update(status="INTERRUPTED", phase="Worker interrupted", error="The previous process exited before completion.", finished_at=now())
                        atomic_json(old_record, interrupted)
                self._release_lock(owner["run_id"])
            run_id = "run_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8]
            # Exclusive creation also serializes separate CLI and app processes.
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError as exc:
                raise ValueError("Another training job just started") from exc
            with os.fdopen(descriptor, "w") as stream:
                json.dump({"run_id": run_id, "pid": os.getpid(), "directory": str(self.root / run_id)}, stream)
            directory = self.root / run_id
            directory.mkdir(exist_ok=False)
            record = {"run_id": run_id, "status": "QUEUED", "phase": "Preparing worker", "mode": mode,
                      "created_at": now(), "started_unix": time.time(), "pid": os.getpid(),
                      "dataset": dataset.provenance, "config": dataset.config, "detail": ""}
            try:
                if data is not None:
                    # Store canonical validated rows, never an external upload path.
                    snapshot = dataset.x.copy()
                    snapshot["label"] = dataset.y
                    snapshot.to_csv(directory / "dataset.csv", index=False)
                atomic_json(directory / "run.json", record)
                atomic_json(directory / "config.json", dataset.config)
                with (directory / "worker.log").open("w", encoding="utf-8") as output:
                    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
                    process = subprocess.Popen([sys.executable, "-m", "sentinel_bc.runs", "--worker", str(directory)],
                        cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, creationflags=flags)
                self.processes[run_id] = process
                record["pid"] = process.pid
                # The worker does not begin until its ownership is recorded.
                atomic_json(directory / "run.json", record)
                atomic_json(lock, {"run_id": run_id, "pid": process.pid, "directory": str(directory)})
                (directory / "ready").touch()
            except Exception as exc:
                record.update(status="FAILED", error=str(exc), finished_at=now())
                atomic_json(directory / "run.json", record)
                self._release_lock(run_id)
                raise
        return run_id

    def cancel(self, run_id):
        record = self.status(run_id)
        if record["status"] in ("QUEUED", "RUNNING", "CANCELLING"):
            (self.root / run_id / "cancel.request").touch()
            # The worker owns status updates; avoid racing a completion write.
            return "Cancellation requested. The worker stops at its next model or epoch boundary."
        return "This job is no longer running."

    def validate_run(self, run_id, load_engine=True):
        artifacts, results = self.paths(run_id)
        if run_id != "benchmark":
            record = self.status(run_id)
            if record["status"] != "COMPLETED":
                raise ValueError("Only successfully completed runs can be activated")
            for name, digest in record["files"].items():
                path = (self.root / run_id / name).resolve()
                if self.root / run_id not in path.parents or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                    raise ValueError("Run outputs are missing or modified")
        manifest = json.loads((artifacts / "manifest.json").read_text())
        metrics = json.loads((results / "metrics.json").read_text())
        experiment = json.loads((results / "experiment.json").read_text())
        if metrics["version"] != manifest["version"] or experiment != manifest:
            raise ValueError("Model and evaluation belong to different runs")
        if run_id != "benchmark" and manifest["dataset"]["fingerprint"] != record["dataset"]["fingerprint"]:
            raise ValueError("Dataset fingerprint does not match the experiment")
        return (Engine(artifacts) if load_engine else manifest), metrics, results

    def completed(self):
        rows = []
        for run_id in ["benchmark"] + sorted([p.name for p in self.root.glob("run_*") if p.is_dir()], reverse=True):
            try:
                metadata, metrics, _ = self.validate_run(run_id, load_engine=False)
                rows.append({"run_id": run_id, "mode": metadata["mode"], "model": metadata["selected_model"],
                             "dataset": metadata.get("dataset", {}).get("source", "Historical WDBC benchmark"),
                             "version": metadata["version"], "accuracy": metrics["test"]["accuracy"]})
            except (ValueError, KeyError, FileNotFoundError):
                continue
        return rows

    def activate(self, run_id):
        engine, metrics, directory = self.validate_run(run_id)
        with self.active.lock:
            atomic_json(self.root / "active.json", {"run_id": run_id})
            self.active.set(run_id, engine, metrics, directory)
            self.notice = f"Active: {run_id} · {engine.metadata['selected_model']} · {engine.metadata['version']}"
        return self.notice


def worker(directory):
    from .training import train, TrainingCancelled
    directory = Path(directory).resolve()
    for _ in range(100):
        if (directory / "ready").exists():
            break
        time.sleep(.1)
    record = json.loads((directory / "run.json").read_text())
    def progress(phase, detail=""):
        record.update(status="RUNNING", phase=phase, detail=detail)
        atomic_json(directory / "run.json", record)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"time": now(), "phase": phase, "detail": detail}) + "\n")
    try:
        config = json.loads((directory / "config.json").read_text())
        data = directory / "dataset.csv" if (directory / "dataset.csv").exists() else None
        artifacts = train(record["mode"], directory, data=data, config=config, progress=progress,
                          cancelled=lambda: (directory / "cancel.request").exists())
        engine = Engine(artifacts)
        results = directory / "results" / record["mode"]
        metrics = json.loads((results / "metrics.json").read_text())
        if metrics["version"] != engine.metadata["version"] or engine.metadata["dataset"]["fingerprint"] != record["dataset"]["fingerprint"]:
            raise ValueError("Completed output did not match the requested dataset and model")
        record.update(status="COMPLETED", phase="Training complete", model_version=engine.metadata["version"],
                      selected_model=engine.metadata["selected_model"], summary={"test": metrics["test"], "selective": metrics["selective"]},
                      files={str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for base in (directory / "artifacts", directory / "results") for p in base.rglob("*") if p.is_file()})
    except TrainingCancelled as exc:
        record.update(status="CANCELLED", phase="Cancelled", error=str(exc))
    except Exception as exc:
        import traceback
        traceback.print_exc()
        record.update(status="FAILED", phase="Training failed", error=str(exc))
    finally:
        record.update(finished_at=now(), duration_seconds=round(time.time()-record["started_unix"], 2))
        atomic_json(directory / "run.json", record)
        lock = TRAINING_LOCK
        if lock.exists() and json.loads(lock.read_text())["run_id"] == record["run_id"]:
            lock.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True)
    worker(parser.parse_args().worker)
