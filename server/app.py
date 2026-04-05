"""
server/app.py  — L.I.S.T.E.N. FastAPI Backend
Receives .wav uploads from edge nodes, queues them, runs the pipeline,
and exposes results to the Streamlit frontend via /get_results.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import time
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from server.model_manager import ModelManager

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
BUFFER_DIR = BASE_DIR / "audio_buffer"
BUFFER_DIR.mkdir(parents=True, exist_ok=True)

# ── Global state ───────────────────────────────────────────────────────────
node_state: dict = {}
registry = ModelManager()

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND WORKER
# ─────────────────────────────────────────────────────────────────────────────


async def _worker(app_instance: FastAPI):
    print("[WORKER] Pipeline queue is spinning...")
    while True:
        item = await app_instance.state.audio_queue.get()
        file_path, node_id, node_type, node_type_id, project_id = item

        try:
            if node_id in node_state:
                node_state[node_id]["status"] = "processing"

            result = await asyncio.to_thread(
                registry.process_pipeline,
                file_path,
                project_id,
                node_type,
            )

            if node_id in node_state:
                if "error" in result:
                    err_msg = result["error"]
                    node_state[node_id]["status"] = f"error: {err_msg[:80]}"
                    print(f"[PIPELINE ERROR] {node_id}: {err_msg}")
                else:
                    node_state[node_id]["latest_result"] = result
                    node_state[node_id]["status"] = "idle"
                    anomaly_flag = (
                        "ANOMALY" if result.get("is_anomaly") else "✅ normal"
                    )
                    print(
                        f"[RESULT] {node_id} | {result.get('machine_type','?')} "
                        f"id_{result.get('machine_id','?')} | "
                        f"MSE={result.get('mse_score',0):.5f} "
                        f"thr={result.get('threshold',0):.5f} | {anomaly_flag}"
                    )

        except Exception as e:
            print(f"[WORKER EXCEPTION] {node_id}: {e}")
            if node_id in node_state:
                node_state[node_id]["status"] = f"error: {str(e)[:80]}"

        finally:
            # Clean up the temporary buffer file
            try:
                if os.path.exists(file_path):
                    # Small delay so the OS releases file handles on Windows
                    await asyncio.sleep(2)
                    os.remove(file_path)
            except Exception:
                pass
            app_instance.state.audio_queue.task_done()


# ─────────────────────────────────────────────────────────────────────────────
# APP LIFESPAN
# ─────────────────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.audio_queue = asyncio.Queue()
    worker_task = asyncio.create_task(_worker(app))
    yield
    worker_task.cancel()
    try:
        await worker_task
    except asyncio.CancelledError:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="L.I.S.T.E.N. API",
    description="Latent Inference of Sequential Temporal Energy Networks",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    """Quick liveness probe."""
    return {
        "status": "ok",
        "nodes_active": len(node_state),
        "queue_size": app.state.audio_queue.qsize(),
    }


@app.post("/upload_audio/{project_id}/{node_type}/{node_type_id}/{node_id}")
async def upload_audio(
    project_id: str,
    node_type: str,
    node_type_id: str,
    node_id: str,
    file: UploadFile = File(...),
):
    """
    Receive a WAV file from an edge node, buffer it, and enqueue for processing.

    URL params
    ----------
    project_id   : 'edge' or 'gpu'
    node_type    : machine type reported by the node (fan | pump | slider | valve)
    node_type_id : machine ID reported by the node (00 | 02 | 04 | 06)
    node_id      : unique identifier for this physical node
    """
    if project_id not in ("edge", "gpu"):
        raise HTTPException(
            status_code=400, detail="project_id must be 'edge' or 'gpu'"
        )

    # Register the node if we haven't seen it before
    if node_id not in node_state:
        node_state[node_id] = {
            "project_id": project_id,
            "node_type": node_type,
            "node_type_id": node_type_id,
            "status": "connected",
            "latest_result": None,
        }
        print(
            f"[REGISTER] New node: {node_id}  [{project_id.upper()}]  {node_type}/id_{node_type_id}"
        )

    # Write the file to the buffer directory with a unique name
    suffix = Path(file.filename or "upload.wav").suffix or ".wav"
    unique_filename = f"{node_id}_{int(time.time() * 1000)}{suffix}"
    file_location = BUFFER_DIR / unique_filename

    with open(file_location, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    node_state[node_id]["status"] = "queued"
    await app.state.audio_queue.put(
        (str(file_location), node_id, node_type, node_type_id, project_id)
    )

    return {
        "status": "queued_for_processing",
        "queue_depth": app.state.audio_queue.qsize(),
    }


@app.post("/classify")
async def classify_audio(file: UploadFile = File(...)):
    """Synchronous endpoint to classify a single uploaded WAV file."""
    temp_filename = f"temp_classify_{int(time.time() * 1000)}.wav"
    temp_path = BUFFER_DIR / temp_filename

    # 1. BULLETPROOF FILE SAVING
    # Use await file.read() to guarantee we capture the full byte payload
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # 2. RAW AUDIO LOAD (Bypass Noise Reduction)
        # We must classify on raw audio. If we subtract the wrong machine's
        # noise profile right now, we will destroy the acoustic features.
        import librosa

        # Load the audio at 16kHz to match the MIMII dataset standards
        y, sr = librosa.load(str(temp_path), sr=16000)

        # 3. Extract features & classify
        audio_features = registry._audio_features(y, sr)
        machine_type, machine_id = registry._classify(audio_features)

        return {"predicted_label": f"{machine_type}{machine_id}"}

    except Exception as e:
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # 4. Clean up
        if temp_path.exists():
            os.remove(temp_path)


@app.get("/get_results")
async def get_results():
    """Return the current state of all registered nodes."""
    return node_state


@app.delete("/reset")
async def reset():
    """Clear all node state (useful for demos)."""
    node_state.clear()
    return {"status": "cleared"}
