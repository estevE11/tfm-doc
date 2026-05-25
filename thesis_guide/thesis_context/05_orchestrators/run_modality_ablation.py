#!/usr/bin/env python3
"""
Modality ablation orchestrator (Step A of `thesis_ablation_plan.md`).

Submits 12 CogniAligned training jobs (4 cells x 3 datasets, 5-fold CV inside each),
waits for them to finish, then runs ADReSSo test-set inference for the 4 ADReSSo
cells and aggregates everything into `modality_ablation.csv`.

CSV columns:
  model,dataset,cell,fusion,n_heads,modality,
  total_params,trainable_params,
  val_acc_mean,best_fold_acc,
  val_inference_ms_per_sample,
  test_acc,test_inference_ms_per_sample,note

Usage:
    python3 run_modality_ablation.py

Run me in the background and tail the log file:
    tail -f /home/usuaris/veussd/roger.esteve.sanchez/modality_ablation.log
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

BASE = Path("/home/usuaris/veussd/roger.esteve.sanchez")
COG = BASE / "CogniAligned"
SLURM_LOG_DIR = COG / "logs" / "slurm"
SLURM_LOG_DIR.mkdir(parents=True, exist_ok=True)

STATE_FILE = BASE / "modality_ablation_state.json"
ORCH_LOG = BASE / "modality_ablation.log"
CSV_OUT = BASE / "modality_ablation.csv"

# Sleep interval between sacct polls (seconds)
POLL_INTERVAL = 120

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
CELLS = [
    # (cell_id,        description,                fusion,       textual,  audio,      use_af, tag)
    ("A1_audio",       "Audio only (no text/AF)",  "audioonly",  "",       "wav2vec2", False,  "_ablA1"),
    ("A2_text",        "Text only (no audio/AF)",  "textonly",   "distil", "",         False,  "_ablA2"),
    ("A3_av_mean",     "Audio+text mean fusion",   "mean",       "distil", "wav2vec2", False,  "_ablA3"),
    ("A4_av_mean_af",  "Audio+text+AF mean",       "mean",       "distil", "wav2vec2", True,   "_ablA4"),
]

DATASETS = [
    # (dataset_id, main_entry_py,             config_yaml_template, path_prefix, num_classes, supports_test_inference)
    ("adresso", "modules/main.py",         "ablation/{cell}_adresso.yaml", "adresso", 1, True),
    ("amyloid", "modules/amyloid/main.py", "ablation/{cell}_amyloid.yaml", "amyloid", 1, False),
    ("ppa",     "modules/ppa/main.py",     "ablation/{cell}_ppa.yaml",     "ppa",     3, False),
]

# Mapping from cell_id (with letter prefix) to the config base name in our YAMLs:
#   A1_audio -> a1_audio, A2_text -> a2_text, etc.
def cell_config_base(cell_id: str) -> str:
    return cell_id.lower()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ORCH_LOG), logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------
def submit_training_job(cell_id: str, dataset_id: str) -> Optional[str]:
    """Submit one training job (5-fold CV inside) and return its job_id, or None on failure."""
    cell_base = cell_config_base(cell_id)
    config_rel = f"modules/configs/ablation/{cell_base}_{dataset_id}.yaml"
    config_abs = COG / config_rel
    if not config_abs.exists():
        log.error(f"Missing config: {config_abs}")
        return None
    main_py = None
    for did, mp, _, _, _, _ in DATASETS:
        if did == dataset_id:
            main_py = mp
            break
    if main_py is None:
        log.error(f"Unknown dataset_id: {dataset_id}")
        return None

    job_name = f"abl_{cell_id}_{dataset_id}"
    # NB: sbatch --wrap runs under /bin/sh on this cluster; we therefore
    # avoid `source` and instead call the venv's python directly.
    wrap = (
        f"cd '{COG}' && "
        "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && "
        f"export HF_HOME='{COG}/.cache/huggingface' "
        f"WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && "
        f"mkdir -p \"$HF_HOME\" \"$WANDB_DIR\" '{SLURM_LOG_DIR}' && "
        f"'{COG}/.venv/bin/python' -u {main_py} --config {config_rel}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={SLURM_LOG_DIR}/%x_%j.txt",
        "-A", "veu", "-p", "veu",
        "--cpus-per-task=16", "--mem=64GB", "--gres=gpu:1", "--ntasks=1",
        "--exclude=veuc05,veuc01,veuc11,veuc10",
        "--time=06:00:00",
        "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch failed for {job_name}: {p.stderr.strip()}")
        return None
    out = p.stdout.strip()
    # Expected: "Submitted batch job <id>"
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        log.error(f"Could not parse sbatch output for {job_name}: {out}")
        return None
    jid = m.group(1)
    log.info(f"Submitted {job_name} as job {jid}")
    return jid


def submit_test_inference_job(cell_id: str) -> Optional[str]:
    """Submit ADReSSo test inference for one ablation cell. Returns job_id."""
    cell_base = cell_config_base(cell_id)
    config_rel = f"modules/configs/ablation/{cell_base}_adresso.yaml"
    job_name = f"abl_test_{cell_id}"
    wrap = (
        f"cd '{COG}' && "
        "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        "WANDB_MODE=disabled WANDB_SILENT=true && "
        f"export HF_HOME='{COG}/.cache/huggingface' "
        f"WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && "
        f"'{COG}/.venv/bin/python' -u modules/test.py --config {config_rel}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={SLURM_LOG_DIR}/%x_%j.txt",
        "-A", "veu", "-p", "veu",
        "--cpus-per-task=8", "--mem=32GB", "--gres=gpu:1", "--ntasks=1",
        "--exclude=veuc05,veuc01",
        "--time=01:00:00",
        "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch test inference failed for {job_name}: {p.stderr.strip()}")
        return None
    out = p.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        log.error(f"Could not parse sbatch output: {out}")
        return None
    jid = m.group(1)
    log.info(f"Submitted test inference {job_name} as job {jid}")
    return jid


def job_state(job_id: str) -> str:
    """Return the SLURM state for a job_id. Uses sacct for terminal states,
    falls back to squeue. Returns 'UNKNOWN' if neither yields a result."""
    p = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-o", "State", "-j", job_id],
        capture_output=True, text=True,
    )
    if p.returncode == 0 and p.stdout.strip():
        state = p.stdout.strip().splitlines()[0].split()[0]
        return state
    # Fallback to squeue (job may still be pending and not yet in sacct)
    p = subprocess.run(
        ["squeue", "-h", "-j", job_id, "-o", "%T"],
        capture_output=True, text=True,
    )
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().splitlines()[0]
    return "UNKNOWN"


def wait_for_jobs(job_ids: List[str], label: str) -> Dict[str, str]:
    """Block until every job is in a terminal state. Returns dict job_id -> final state."""
    pending = set(job_ids)
    final_states: Dict[str, str] = {}
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"}
    log.info(f"Waiting on {len(pending)} {label} jobs: {sorted(pending)}")
    while pending:
        time.sleep(POLL_INTERVAL)
        running, queued, done_now = 0, 0, []
        for jid in list(pending):
            state = job_state(jid)
            if any(state.startswith(t) for t in terminal):
                final_states[jid] = state
                pending.discard(jid)
                done_now.append((jid, state))
            elif state == "RUNNING":
                running += 1
            elif state in ("PENDING", "REQUEUED"):
                queued += 1
        if done_now:
            for jid, st in done_now:
                log.info(f"  [{label}] job {jid} -> {st}")
        log.info(f"  [{label}] {len(pending)} still active (running={running}, queued={queued})")
    return final_states


# ---------------------------------------------------------------------------
# Path-name reconstruction (mirrors save_config in utils.py)
# ---------------------------------------------------------------------------
def derive_path_name(cell_id: str, dataset_id: str) -> str:
    """Recreate `path_name` as written by main.py for this cell+dataset.

    main.py prepends `<dataset>_` to path_name before writing logs, so we mirror
    that here to get e.g. `adresso_distil_wav2vec2_P_mean_ablA3_mean`.
    """
    _, _, fusion, textual, audio, _, tag = next(c for c in CELLS if c[0] == cell_id)
    textual_data = f"{textual}_" if textual else ""
    audio_data = f"{audio}_" if audio else ""
    pauses_data = "P_"  # all our configs use pauses=True
    pooling = "mean"
    model_name = f"{textual_data}{audio_data}{pauses_data}{fusion}{tag}"
    base = f"{model_name}_{pooling}"
    return f"{dataset_id}_{base}"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_cross_fold_summary(path: Path) -> Tuple[List[float], Optional[float]]:
    """Return list of per-fold accuracies and the best fold accuracy (max)."""
    accs: List[float] = []
    if not path.exists():
        return accs, None
    with open(path, "r") as f:
        for line in f:
            m = re.match(r"^Fold (\d+): Best Value\s*=\s*([0-9.]+)", line)
            if m:
                try:
                    accs.append(float(m.group(2)))
                except ValueError:
                    pass
    return accs, (max(accs) if accs else None)


def parse_val_inference_ms(log_dir: Path, num_folds: int) -> Optional[float]:
    """Average 'Inference ms/sample' across the LAST eval entry of each fold's train_stats file.

    We take the *last* logged inference time per fold (matches the converged regime
    much better than averaging across noisy early epochs).
    """
    times: List[float] = []
    for fold in range(num_folds):
        stats_path = log_dir / f"train_stats_{fold}.txt"
        if not stats_path.exists():
            continue
        last_val: Optional[float] = None
        with open(stats_path, "r") as f:
            for line in f:
                m = re.match(r"^Inference ms/sample:\s*([0-9.]+)", line)
                if m:
                    try:
                        last_val = float(m.group(1))
                    except ValueError:
                        pass
        if last_val is not None:
            times.append(last_val)
    if not times:
        return None
    return sum(times) / len(times)


def parse_params_from_slurm_log(job_id: str) -> Tuple[Optional[int], Optional[int]]:
    """Look in the slurm output file for 'Total Parameters: X,YYY,ZZZ ...' lines.

    Returns (total_params, trainable_params).
    """
    log_files = list(SLURM_LOG_DIR.glob(f"*_{job_id}.txt"))
    if not log_files:
        return None, None
    total, trainable = None, None
    for lf in log_files:
        try:
            with open(lf, "r") as f:
                for line in f:
                    m = re.match(r"Total Parameters:\s*([0-9,]+)", line)
                    if m and total is None:
                        total = int(m.group(1).replace(",", ""))
                    m = re.match(r"Trainable Parameters:\s*([0-9,]+)", line)
                    if m and trainable is None:
                        trainable = int(m.group(1).replace(",", ""))
                    if total is not None and trainable is not None:
                        return total, trainable
        except OSError:
            continue
    return total, trainable


def parse_test_result_from_slurm_log(job_id: str) -> Dict[str, Optional[float]]:
    """Look for the single-line `TEST_RESULT ...` summary in the slurm output."""
    log_files = list(SLURM_LOG_DIR.glob(f"*_{job_id}.txt"))
    result = {"acc": None, "ms_per_sample": None}
    for lf in log_files:
        try:
            with open(lf, "r") as f:
                for line in f:
                    if line.startswith("TEST_RESULT"):
                        kv = dict(token.split("=", 1) for token in line.strip().split() if "=" in token)
                        try:
                            result["acc"] = float(kv.get("acc", "nan"))
                        except ValueError:
                            pass
                        try:
                            result["ms_per_sample"] = float(kv.get("ms_per_sample", "nan"))
                        except ValueError:
                            pass
                        return result
        except OSError:
            continue
    return result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------
def build_csv(state: dict) -> None:
    """Aggregate results across all training and test-inference jobs into CSV."""
    rows: List[Dict[str, str]] = []
    for cell_id, _, fusion, textual, audio, use_af, _ in CELLS:
        modality_tokens = []
        if audio:
            modality_tokens.append("A")
        if textual:
            modality_tokens.append("T")
        if use_af:
            modality_tokens.append("AF")
        modality = "+".join(modality_tokens) if modality_tokens else "none"
        for dataset_id, _, _, _, num_classes, supports_test in DATASETS:
            key = f"{cell_id}__{dataset_id}"
            job_meta = state.get("train_jobs", {}).get(key, {})
            test_meta = state.get("test_jobs", {}).get(cell_id, {}) if supports_test else {}
            train_job_id = job_meta.get("job_id")
            train_state = job_meta.get("state", "UNKNOWN")

            path_name = derive_path_name(cell_id, dataset_id)
            log_dir = COG / "logs" / path_name
            cross_fold = log_dir / "cross_fold_summary.txt"
            accs, best_fold = parse_cross_fold_summary(cross_fold)
            val_acc_mean = (sum(accs) / len(accs)) if accs else None
            val_ms = parse_val_inference_ms(log_dir, num_folds=5)
            total_params, trainable_params = (None, None)
            if train_job_id:
                total_params, trainable_params = parse_params_from_slurm_log(train_job_id)

            test_acc = test_meta.get("test_acc") if supports_test else None
            test_ms = test_meta.get("test_ms_per_sample") if supports_test else None

            note_parts: List[str] = []
            if train_state and train_state != "COMPLETED":
                note_parts.append(f"train={train_state}")
            if accs and len(accs) < 5:
                note_parts.append(f"folds={len(accs)}/5")
            if not accs:
                note_parts.append("no_folds_parsed")
            if supports_test and test_meta and test_meta.get("state", "UNKNOWN") != "COMPLETED":
                note_parts.append(f"test={test_meta.get('state','UNKNOWN')}")
            note = ";".join(note_parts) if note_parts else ""

            rows.append({
                "model": "CogniAligned",
                "dataset": dataset_id,
                "cell": cell_id,
                "fusion": fusion,
                "n_heads": "1",
                "modality": modality,
                "total_params": str(total_params) if total_params is not None else "",
                "trainable_params": str(trainable_params) if trainable_params is not None else "",
                "val_acc_mean": f"{val_acc_mean:.6f}" if val_acc_mean is not None else "",
                "best_fold_acc": f"{best_fold:.6f}" if best_fold is not None else "",
                "val_inference_ms_per_sample": f"{val_ms:.4f}" if val_ms is not None else "",
                "test_acc": f"{test_acc:.6f}" if isinstance(test_acc, float) else "",
                "test_inference_ms_per_sample": f"{test_ms:.4f}" if isinstance(test_ms, float) else "",
                "note": note,
            })

    columns = [
        "model", "dataset", "cell", "fusion", "n_heads", "modality",
        "total_params", "trainable_params",
        "val_acc_mean", "best_fold_acc",
        "val_inference_ms_per_sample",
        "test_acc", "test_inference_ms_per_sample",
        "note",
    ]
    with open(CSV_OUT, "w") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(r.get(c, "") for c in columns) + "\n")
    log.info(f"Wrote {CSV_OUT} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def main() -> None:
    state = load_state()
    state.setdefault("train_jobs", {})  # key: "<cell>__<dataset>" -> {job_id, state}
    state.setdefault("test_jobs", {})   # key: "<cell>" -> {job_id, state, test_acc, test_ms_per_sample}
    state["started_at"] = state.get("started_at", datetime.now().isoformat(timespec="seconds"))

    # ─── Phase 1: training submissions ─────────────────────────────────────
    log.info("=== Phase 1: submitting 12 training jobs ===")
    for cell_id, *_ in CELLS:
        for dataset_id, *_ in DATASETS:
            key = f"{cell_id}__{dataset_id}"
            if key in state["train_jobs"] and state["train_jobs"][key].get("job_id"):
                log.info(f"  skip {key}: already submitted as job {state['train_jobs'][key]['job_id']}")
                continue
            jid = submit_training_job(cell_id, dataset_id)
            if jid is None:
                state["train_jobs"][key] = {"job_id": None, "state": "SUBMIT_FAILED"}
            else:
                state["train_jobs"][key] = {"job_id": jid, "state": "SUBMITTED"}
            save_state(state)

    # ─── Phase 2: wait for training to finish ──────────────────────────────
    log.info("=== Phase 2: waiting for training jobs to finish ===")
    train_ids = [v["job_id"] for v in state["train_jobs"].values() if v.get("job_id")]
    finals = wait_for_jobs(train_ids, "train")
    for key, meta in state["train_jobs"].items():
        if meta.get("job_id") in finals:
            meta["state"] = finals[meta["job_id"]]
    save_state(state)

    # ─── Phase 3: build interim CSV (in case test inference fails) ────────
    log.info("=== Phase 3: building interim CSV (training results only) ===")
    build_csv(state)

    # ─── Phase 4: ADReSSo test inference ──────────────────────────────────
    log.info("=== Phase 4: submitting ADReSSo test inference for 4 cells ===")
    for cell_id, *_ in CELLS:
        if cell_id in state["test_jobs"] and state["test_jobs"][cell_id].get("job_id"):
            log.info(f"  skip {cell_id}: already submitted as job {state['test_jobs'][cell_id]['job_id']}")
            continue
        # Only submit test inference if the corresponding ADReSSo training completed.
        train_key = f"{cell_id}__adresso"
        train_state = state["train_jobs"].get(train_key, {}).get("state", "")
        if train_state != "COMPLETED":
            log.warning(f"  skipping test inference for {cell_id}: ADReSSo training state={train_state}")
            state["test_jobs"][cell_id] = {"job_id": None, "state": "SKIPPED_NO_TRAIN"}
            save_state(state)
            continue
        jid = submit_test_inference_job(cell_id)
        if jid is None:
            state["test_jobs"][cell_id] = {"job_id": None, "state": "SUBMIT_FAILED"}
        else:
            state["test_jobs"][cell_id] = {"job_id": jid, "state": "SUBMITTED"}
        save_state(state)

    # ─── Phase 5: wait for test inference ─────────────────────────────────
    log.info("=== Phase 5: waiting for ADReSSo test inference jobs ===")
    test_ids = [v["job_id"] for v in state["test_jobs"].values() if v.get("job_id")]
    if test_ids:
        finals = wait_for_jobs(test_ids, "test")
        for cell_id, meta in state["test_jobs"].items():
            if meta.get("job_id") in finals:
                meta["state"] = finals[meta["job_id"]]
                # Parse TEST_RESULT line from slurm log
                parsed = parse_test_result_from_slurm_log(meta["job_id"])
                meta["test_acc"] = parsed.get("acc")
                meta["test_ms_per_sample"] = parsed.get("ms_per_sample")
        save_state(state)

    # ─── Phase 6: final CSV ───────────────────────────────────────────────
    log.info("=== Phase 6: building final CSV ===")
    build_csv(state)
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    log.info(f"DONE — modality ablation finished. CSV: {CSV_OUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted by user; state saved to %s — re-run to resume.", STATE_FILE)
        sys.exit(130)
