"""
server/edge_node.py  — L.I.S.T.E.N. MIMII Dataset Streamer

Scans raw_data/ for the MIMII dataset folder structure, discovers every
machine instance, and spawns one virtual "node" per instance that periodically
POSTs real .wav files to the FastAPI backend.

MIMII expected layout
─────────────────────
raw_data/
  {snr}_{machine_type}/          e.g. 6_dB_fan
    {machine_type}/              e.g. fan
      id_{XX}/                   e.g. id_00
        normal/
          00000000.wav …
        abnormal/
          00000000.wav …

Usage
─────
  # Auto-discover all machines and stream:
  python -m server.edge_node

  # Stream only one machine type:
  python -m server.edge_node --type fan

  # Control anomaly injection rate (0.0 = all normal, 1.0 = all abnormal):
  python -m server.edge_node --anomaly_rate 0.15

  # Control interval between sends (seconds):
  python -m server.edge_node --interval 5
"""

from __future__ import annotations

import argparse
import random
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

SERVER_URL = "http://localhost:8000/upload_audio"
RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "raw_data"

VALID_TYPES = {"fan", "pump", "slider", "valve"}
VALID_IDS = {"00", "02", "04", "06"}

# Valve has no id_06 in the classifier output space — keep consistent
VALVE_VALID_IDS = {"00", "02", "04"}

# Edge nodes stream 8-feature models; GPU nodes stream 3-ch Mel models.
# We alternate assignments so the dashboard shows both node types.
_NODE_COUNTER = 0
_LOCK = threading.Lock()


def _next_project() -> str:
    global _NODE_COUNTER
    with _LOCK:
        proj = "edge" if _NODE_COUNTER % 2 == 0 else "gpu"
        _NODE_COUNTER += 1
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────


def discover_machines(raw_data: Path, filter_type: Optional[str] = None) -> list[dict]:
    """
    Walk raw_data/ and return a list of machine instance dicts:
      {"machine_type": str, "machine_id": str, "normal": [Path], "abnormal": [Path]}
    """
    machines = []

    if not raw_data.exists():
        print(f"[WARN] raw_data directory not found: {raw_data}")
        print("[WARN] Using dummy WAV generator — no real files found.")
        return []

    for snr_dir in sorted(raw_data.iterdir()):
        if not snr_dir.is_dir():
            continue

        # Parse e.g. "6_dB_fan" or "-6_dB_pump"
        parts = snr_dir.name.split("_")
        if len(parts) < 3:
            continue

        # Last token is machine type
        machine_type = parts[-1].lower()
        if machine_type not in VALID_TYPES:
            continue
        if filter_type and machine_type != filter_type.lower():
            continue

        machine_dir = snr_dir / machine_type
        if not machine_dir.exists():
            continue

        for id_dir in sorted(machine_dir.iterdir()):
            if not id_dir.is_dir() or not id_dir.name.startswith("id_"):
                continue

            machine_id = id_dir.name.replace("id_", "")
            if machine_id not in VALID_IDS:
                continue
            if machine_type == "valve" and machine_id not in VALVE_VALID_IDS:
                continue

            normal_files = (
                sorted((id_dir / "normal").glob("*.wav"))
                if (id_dir / "normal").exists()
                else []
            )
            abnormal_files = (
                sorted((id_dir / "abnormal").glob("*.wav"))
                if (id_dir / "abnormal").exists()
                else []
            )

            if not normal_files and not abnormal_files:
                continue

            machines.append(
                {
                    "machine_type": machine_type,
                    "machine_id": machine_id,
                    "normal": normal_files,
                    "abnormal": abnormal_files,
                    "snr": snr_dir.name,
                }
            )
            print(
                f"  [FOUND] {machine_type}/id_{machine_id} "
                f"({len(normal_files)} normal, {len(abnormal_files)} abnormal) "
                f"[{snr_dir.name}]"
            )

    return machines


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY WAV (fallback when no raw_data exists)
# ─────────────────────────────────────────────────────────────────────────────


def _make_dummy_wav(path: Path, machine_type: str = "valve", anomaly: bool = False):
    """Generate a synthetic WAV using only the stdlib — no librosa needed."""
    import wave, struct, math

    sample_rate = 16_000
    duration = 2
    n_samples = sample_rate * duration

    # Fundamental tone + harmonics to make it machine-like
    base_freq = {"fan": 120.0, "pump": 80.0, "slider": 200.0, "valve": 160.0}.get(
        machine_type, 160.0
    )
    harmonics = [1.0, 0.5, 0.25, 0.125]

    if anomaly:
        # Add high-frequency noise burst to simulate a fault
        harmonics.append(0.35)
        base_freq *= 1.5

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            t = i / sample_rate
            value = sum(
                amp * math.sin(2 * math.pi * (n + 1) * base_freq * t)
                for n, amp in enumerate(harmonics)
            )
            if anomaly:
                noise = random.gauss(0, 0.15)
                value = value * 0.7 + noise
            value = max(-1.0, min(1.0, value))
            wf.writeframes(struct.pack("<h", int(32767 * value)))


# ─────────────────────────────────────────────────────────────────────────────
# VIRTUAL NODE (one per machine instance)
# ─────────────────────────────────────────────────────────────────────────────


class VirtualNode(threading.Thread):
    """
    A background thread that continuously streams WAV files for one
    machine instance (machine_type + machine_id).
    """

    def __init__(
        self,
        node_id: str,
        project_id: str,
        machine_type: str,
        machine_id: str,
        normal_files: list[Path],
        abnormal_files: list[Path],
        anomaly_rate: float = 0.1,
        interval: float = 5.0,
        jitter: float = 1.5,
    ):
        super().__init__(daemon=True, name=f"Node-{node_id}")
        self.node_id = node_id
        self.project_id = project_id
        self.machine_type = machine_type
        self.machine_id = machine_id
        self.normal_files = normal_files
        self.abnormal_files = abnormal_files
        self.anomaly_rate = anomaly_rate
        self.interval = interval
        self.jitter = jitter
        self._sent = 0
        self._errors = 0

    def _pick_file(self) -> tuple[Path, bool]:
        """Choose a file to send; returns (path, is_anomaly)."""
        inject_anomaly = random.random() < self.anomaly_rate and bool(
            self.abnormal_files
        )
        if inject_anomaly:
            return random.choice(self.abnormal_files), True
        elif self.normal_files:
            return random.choice(self.normal_files), False
        else:
            return random.choice(self.abnormal_files), True

    def _post(self, wav_path: Path):
        url = (
            f"{SERVER_URL}/{self.project_id}/{self.machine_type}"
            f"/{self.machine_id}/{self.node_id}"
        )
        with open(wav_path, "rb") as fh:
            resp = requests.post(
                url,
                files={"file": (wav_path.name, fh, "audio/wav")},
                timeout=10,
            )
        return resp.status_code

    def run(self):
        # Initial stagger so all nodes don't hit the server simultaneously
        time.sleep(random.uniform(0, self.interval))

        while True:
            try:
                wav_path, is_anomaly = self._pick_file()
                tag = "⚠ ANOMALY" if is_anomaly else "  normal "
                code = self._post(wav_path)
                self._sent += 1
                print(
                    f"[{time.strftime('%H:%M:%S')}] "
                    f"{self.node_id:<16} {self.project_id.upper():4} "
                    f"{self.machine_type}/{self.machine_id}  {tag}  "
                    f"file={wav_path.name}  http={code}"
                )
            except requests.exceptions.ConnectionError:
                self._errors += 1
                if self._errors == 1:
                    print(
                        f"[ERROR] {self.node_id}: Backend offline — "
                        "start Uvicorn with: uvicorn server.app:app --reload"
                    )
            except Exception as e:
                self._errors += 1
                print(f"[ERROR] {self.node_id}: {e.__class__.__name__}: {e}")

            sleep_secs = self.interval + random.uniform(-self.jitter, self.jitter)
            time.sleep(max(1.0, sleep_secs))


# ─────────────────────────────────────────────────────────────────────────────
# DUMMY-MODE NODE (when raw_data doesn't exist)
# ─────────────────────────────────────────────────────────────────────────────


class DummyNode(VirtualNode):
    """Falls back to synthesised WAV files when no dataset is present."""

    _TMP = Path(__file__).resolve().parent / "_tmp_wavs"

    def _pick_file(self) -> tuple[Path, bool]:
        self._TMP.mkdir(exist_ok=True)
        inject = random.random() < self.anomaly_rate
        fname = (
            self._TMP
            / f"{self.machine_type}_{self.machine_id}_{'ab' if inject else 'n'}.wav"
        )
        if not fname.exists():
            _make_dummy_wav(fname, self.machine_type, anomaly=inject)
        return fname, inject


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="L.I.S.T.E.N. — MIMII Dataset Edge Node Streamer"
    )
    parser.add_argument(
        "--raw_data", default=str(RAW_DATA_DIR), help="Path to raw_data/ directory"
    )
    parser.add_argument(
        "--type",
        default=None,
        help="Filter to one machine type (fan|pump|slider|valve)",
    )
    parser.add_argument(
        "--anomaly_rate",
        type=float,
        default=0.12,
        help="Fraction of sends that inject an abnormal file (0-1)",
    )
    parser.add_argument(
        "--interval", type=float, default=6.0, help="Seconds between sends per node"
    )
    parser.add_argument(
        "--max_nodes",
        type=int,
        default=8,
        help="Maximum number of concurrent virtual nodes",
    )
    args = parser.parse_args()

    raw_data = Path(args.raw_data)
    print(f"\n{'='*60}")
    print(f"  L.I.S.T.E.N. — Edge Node Streamer")
    print(f"  raw_data : {raw_data}")
    print(f"  server   : {SERVER_URL}")
    print(f"  anomaly  : {args.anomaly_rate*100:.0f}% injection rate")
    print(f"  interval : {args.interval}s ± 1.5s")
    print(f"{'='*60}\n")

    machines = discover_machines(raw_data, filter_type=args.type)

    if not machines:
        print("[ERROR] No MIMII files found in the raw_data/ directory!")
        print("\tPlease download and extract the real MIMII dataset into raw_data/.")
        print("\tExiting streamer...")
        sys.exit(1)

        # If we made it here, we have real data! Limit to max_nodes and shuffle.
        random.shuffle(machines)
        selected = machines[: args.max_nodes]
        nodes = []

    else:
        # Limit to max_nodes, shuffle so we get variety
        random.shuffle(machines)
        selected = machines[: args.max_nodes]
        nodes = []

        for m in selected:
            proj = _next_project()
            node_id = f"node_{m['machine_type']}_{m['machine_id']}_{proj[:1]}"
            NodeCls = VirtualNode

            node = NodeCls(
                node_id=node_id,
                project_id=proj,
                machine_type=m["machine_type"],
                machine_id=m["machine_id"],
                normal_files=m["normal"],
                abnormal_files=m["abnormal"],
                anomaly_rate=args.anomaly_rate,
                interval=args.interval,
            )
            node.start()
            nodes.append(node)
            print(
                f"  [NODE] {node_id:<24} [{proj.upper()}]  "
                f"{m['machine_type']}/id_{m['machine_id']}  "
                f"({len(m['normal'])} normal, {len(m['abnormal'])} abnormal)"
            )

    print(
        f"\n  {len(machines) if machines else 'DUMMY'} nodes streaming — Ctrl+C to stop\n"
    )

    # Keep the main thread alive; threads are daemons so they auto-exit
    try:
        while True:
            time.sleep(30)
            active = threading.active_count() - 1
            print(
                f"[STATUS] {active} nodes active | " f"time={time.strftime('%H:%M:%S')}"
            )
    except KeyboardInterrupt:
        print("\n[EXIT] Streamer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()
