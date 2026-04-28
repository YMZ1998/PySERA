import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from config import (
    CONFUSION_MATRIX_PLOT_PATH,
    FEATURE_METADATA_PATH,
    MODEL_OPTIONS,
    PLOT_OUTPUT_DIR,
    ROC_PLOT_PATH,
    SCRIPT_DIR,
    get_runtime_config,
    update_runtime_config,
)


APP_DIR = Path(SCRIPT_DIR)
TEMPLATES_DIR = APP_DIR / "templates"
STATIC_DIR = APP_DIR / "static"
ARTIFACT_ROOT = APP_DIR
STEP_FILES = {
    "step1": APP_DIR / "step1.py",
    "step2": APP_DIR / "step2.py",
    "step3": APP_DIR / "step3.py",
}

app = FastAPI(title="PySERA Pipeline Console")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/artifacts", StaticFiles(directory=ARTIFACT_ROOT), name="artifacts")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

jobs = {}
jobs_lock = threading.Lock()
active_job_id = None


def iso_now():
    return datetime.now().isoformat(timespec="seconds")


def get_relative_artifact_url(path_value):
    path = Path(path_value)
    if not path.exists():
        return None
    relative_path = path.relative_to(ARTIFACT_ROOT).as_posix()
    return f"/artifacts/{relative_path}?ts={int(path.stat().st_mtime)}"


def load_feature_metadata():
    if not os.path.exists(FEATURE_METADATA_PATH):
        return None
    with open(FEATURE_METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def build_config_summary():
    runtime_config = get_runtime_config()
    return {
        "data_path": runtime_config["data_path"],
        "class_config": runtime_config["class_config"],
        "test_ratio": runtime_config["test_ratio"],
        "random_state": runtime_config["random_state"],
        "use_ttest": runtime_config["use_ttest"],
        "use_muse": runtime_config["use_muse"],
        "max_muse_features": runtime_config["max_muse_features"],
        "model_name": runtime_config["model_name"],
    }


def build_artifact_summary():
    metadata = load_feature_metadata()
    summary = {
        "selected_features_count": 0,
        "selected_features_path": None,
        "roc_plot_url": get_relative_artifact_url(ROC_PLOT_PATH),
        "confusion_matrix_plot_url": get_relative_artifact_url(CONFUSION_MATRIX_PLOT_PATH),
    }
    if metadata:
        summary["selected_features_count"] = len(metadata.get("selected_features", []))
        summary["selected_features_path"] = FEATURE_METADATA_PATH
        summary["train_selected_path"] = metadata.get("train_selected_path")
        summary["test_selected_path"] = metadata.get("test_selected_path")
    return summary


def snapshot_job(job_id):
    with jobs_lock:
        job = jobs[job_id]
        return {
            "id": job["id"],
            "step": job["step"],
            "status": job["status"],
            "started_at": job["started_at"],
            "finished_at": job.get("finished_at"),
            "return_code": job.get("return_code"),
            "logs": "".join(job["logs"]),
            "artifacts": build_artifact_summary(),
        }


def append_job_log(job_id, line):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append(line)


def mark_job_state(job_id, **updates):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id].update(updates)


def run_step_job(job_id, step_name):
    global active_job_id

    mark_job_state(job_id, status="running")
    script_path = STEP_FILES[step_name]
    process = subprocess.Popen(
        [sys.executable, str(script_path)],
        cwd=str(APP_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    try:
        if process.stdout is not None:
            for line in process.stdout:
                append_job_log(job_id, line)
        return_code = process.wait()
        status = "completed" if return_code == 0 else "failed"
        mark_job_state(
            job_id,
            status=status,
            return_code=return_code,
            finished_at=iso_now(),
        )
    except Exception as exc:
        append_job_log(job_id, f"\n[runner] {exc}\n")
        mark_job_state(
            job_id,
            status="failed",
            return_code=-1,
            finished_at=iso_now(),
        )
    finally:
        with jobs_lock:
            active_job_id = None


def ensure_no_active_job():
    with jobs_lock:
        if active_job_id is not None and jobs.get(active_job_id, {}).get("status") == "running":
            raise HTTPException(status_code=409, detail="Another step is still running.")


@app.get("/")
def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "request": request,
            "config_summary": build_config_summary(),
            "artifact_summary": build_artifact_summary(),
            "model_options": MODEL_OPTIONS,
            "steps": list(STEP_FILES.keys()),
        },
    )


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/api/config")
def api_config():
    config_summary = build_config_summary()
    config_summary["model_options"] = MODEL_OPTIONS
    return JSONResponse(config_summary)


@app.post("/api/config")
async def api_update_config(request: Request):
    ensure_no_active_job()
    payload = await request.json()
    updates = {}

    if "use_ttest" in payload:
        updates["use_ttest"] = bool(payload["use_ttest"])
    if "use_muse" in payload:
        updates["use_muse"] = bool(payload["use_muse"])
    if "max_muse_features" in payload:
        updates["max_muse_features"] = int(payload["max_muse_features"])
    if "test_ratio" in payload:
        updates["test_ratio"] = float(payload["test_ratio"])
    if "random_state" in payload:
        updates["random_state"] = int(payload["random_state"])
    if "model_name" in payload:
        model_name = str(payload["model_name"])
        if model_name not in MODEL_OPTIONS:
            raise HTTPException(status_code=400, detail="Unsupported model_name.")
        updates["model_name"] = model_name

    if "data_path" in payload:
        updates["data_path"] = str(payload["data_path"])

    if not updates:
        raise HTTPException(status_code=400, detail="No valid config fields were provided.")

    if updates.get("test_ratio", 0.15) <= 0 or updates.get("test_ratio", 0.15) >= 1:
        raise HTTPException(status_code=400, detail="test_ratio must be between 0 and 1.")
    if updates.get("max_muse_features", 1) <= 0:
        raise HTTPException(status_code=400, detail="max_muse_features must be positive.")

    config_summary = update_runtime_config(updates)
    config_summary["model_options"] = MODEL_OPTIONS
    return JSONResponse(config_summary)


@app.get("/api/artifacts")
def api_artifacts():
    return JSONResponse(build_artifact_summary())


@app.get("/api/jobs/latest")
def api_latest_job():
    with jobs_lock:
        if not jobs:
            return JSONResponse({"job": None})
        latest_job_id = max(jobs, key=lambda job_id: jobs[job_id]["created_order"])
    return JSONResponse({"job": snapshot_job(latest_job_id)})


@app.get("/api/jobs/{job_id}")
def api_job(job_id: str):
    with jobs_lock:
        if job_id not in jobs:
            raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(snapshot_job(job_id))


@app.post("/api/run/{step_name}")
def api_run_step(step_name: str):
    global active_job_id

    if step_name not in STEP_FILES:
        raise HTTPException(status_code=404, detail="Unknown step.")

    ensure_no_active_job()
    job_id = uuid.uuid4().hex
    with jobs_lock:
        created_order = len(jobs) + 1
        jobs[job_id] = {
            "id": job_id,
            "step": step_name,
            "status": "queued",
            "started_at": iso_now(),
            "finished_at": None,
            "return_code": None,
            "logs": [f"[runner] Starting {step_name}...\n"],
            "created_order": created_order,
        }
        active_job_id = job_id

    thread = threading.Thread(target=run_step_job, args=(job_id, step_name), daemon=True)
    thread.start()
    return JSONResponse(snapshot_job(job_id))
