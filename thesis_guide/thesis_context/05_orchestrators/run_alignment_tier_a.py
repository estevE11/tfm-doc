#!/usr/bin/env python3
"""Step C Tier A: token_uniform + token_char × bicross + mamba × 3 datasets."""
from __future__ import annotations

import argparse
import csv
import json
import logging
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
CFG_DIR = COG / "modules" / "configs" / "ablation" / "tier_a"
TEST_DIR = COG / "modules" / "configs" / "ablation" / "tier_a_test"
SLURM = COG / "logs" / "slurm"
STATE_FILE = BASE / "alignment_tier_a_state.json"
ORCH_LOG = BASE / "alignment_tier_a.log"
CSV_OUT = BASE / "alignment_tier_a.csv"
STEP_B = BASE / "step_b_results.csv"

POLL = 90
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}

ALIGNMENTS = [
    ("token_token_uniform", "pilot_tokuniform", "_tokalign", "_ablCtA_tokuni"),
    ("token_token_char", "pilot_tokchar", "_tokalignchar", "_ablCtA_tokchar"),
]
FUSIONS = ("bicross", "mamba")
DATASETS = [
    ("adresso", "modules/main.py"),
    ("amyloid", "modules/amyloid/main.py"),
    ("ppa", "modules/ppa/main.py"),
]

MAMBA_EXTRA = {"mamba_d_state": 16, "mamba_d_conv": 4, "mamba_expand": 2}
FUSION_HEADS = {"bicross": 8, "mamba": 12}


def setup_log(quiet: bool) -> logging.Logger:
    handlers = [logging.FileHandler(ORCH_LOG)]
    if not quiet:
        handlers.append(logging.StreamHandler())
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=handlers)
    return logging.getLogger(__name__)


log = setup_log(quiet=False)


def job_state(jid: str) -> str:
    p = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "State", "-j", jid], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().splitlines()[0].split()[0]
    p = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True)
    return p.stdout.strip().splitlines()[0] if p.returncode == 0 and p.stdout.strip() else "UNKNOWN"


def is_terminal(st: str) -> bool:
    return any(st.startswith(t) for t in TERMINAL)


def sbatch_wrap(job_name: str, wrap: str, mem: str = "64GB", hours: str = "08:00:00") -> Optional[str]:
    SLURM.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sbatch", f"--job-name={job_name}", f"--output={SLURM}/%x_%j.txt",
        "-A", "veu", "-p", "veu", "--cpus-per-task=16", f"--mem={mem}", "--gres=gpu:1",
        "--ntasks=1", "--exclude=veuc05,veuc01,veuc11,veuc10", f"--time={hours}", "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch {job_name}: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    return m.group(1) if m else None


def cell_key(dataset: str, align: str, fusion: str) -> str:
    return f"{dataset}_{align}_{fusion}"


def experiment_tag(align_tag: str, fusion: str) -> str:
    return f"{align_tag}_{fusion}"


def log_dir(dataset: str, fusion: str, tag: str) -> Path:
    return COG / "logs" / f"{dataset}_distil_wav2vec2_P_{fusion}{tag}_mean"


def cell_complete(dataset: str, fusion: str, tag: str) -> bool:
    s = log_dir(dataset, fusion, tag) / "cross_fold_summary.txt"
    if not s.exists():
        return False
    with open(s) as f:
        return sum(1 for line in f if line.startswith("Fold ")) >= 5


def config_path(stem: str, dataset: str) -> Path:
    return CFG_DIR / f"{stem}_{dataset}.yaml"


def ensure_configs() -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for align, pilot_stem, suffix, align_tag in ALIGNMENTS:
        for fusion in FUSIONS:
            for dataset, _ in DATASETS:
                stem = f"tier_a_{pilot_stem.split('_', 1)[1]}_{fusion}"
                out = config_path(stem, dataset)
                if out.exists():
                    continue
                src = COG / "modules" / "configs" / "ablation" / f"{pilot_stem}_{dataset}.yaml"
                with open(src) as f:
                    cfg = yaml.safe_load(f)
                cfg.setdefault("data", {})["alignment_suffix"] = suffix
                cfg.setdefault("model", {})
                cfg["model"]["fusion"] = fusion
                cfg["model"]["n_heads"] = FUSION_HEADS[fusion]
                cfg["model"]["experiment_tag"] = experiment_tag(align_tag, fusion)
                if fusion == "mamba":
                    cfg["model"].update(MAMBA_EXTRA)
                else:
                    for k in MAMBA_EXTRA:
                        cfg["model"].pop(k, None)
                with open(out, "w") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False)


def submit_train(dataset: str, stem: str) -> Optional[str]:
    main_py = next(m for d, m in DATASETS if d == dataset)
    cfg = f"modules/configs/ablation/tier_a/{stem}_{dataset}.yaml"
    name = f"ablCtA_{stem}_{dataset}"[:64]
    wrap = (
        f"cd '{COG}' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        f"HF_HOME='{COG}/.cache/huggingface' WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && '{COG}/.venv/bin/python' -u {main_py} --config {cfg}"
    )
    jid = sbatch_wrap(name, wrap, hours="08:00:00")
    if jid:
        log.info(f"train {name} -> {jid}")
    return jid


def submit_test(stem: str) -> Optional[str]:
    src = config_path(stem, "adresso")
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_prefix"] = "adresso"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    dst = TEST_DIR / src.name
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    rel = dst.relative_to(COG)
    name = f"test_{stem}_adresso"[:64]
    wrap = (
        f"cd '{COG}' && export PYTHONUNBUFFERED=1 WANDB_MODE=disabled "
        f"HF_HOME='{COG}/.cache/huggingface' PYTHONPATH='{COG}/modules' && "
        f"'{COG}/.venv/bin/python' -u modules/test.py --config {rel}"
    )
    jid = sbatch_wrap(name, wrap, mem="32GB", hours="02:00:00")
    if jid:
        log.info(f"test {name} -> {jid}")
    return jid


def parse_summary(dataset: str, fusion: str, tag: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = log_dir(dataset, fusion, tag) / "cross_fold_summary.txt"
    if not p.exists():
        return None, None, None
    accs = []
    with open(p) as f:
        for line in f:
            m = re.match(r"^Fold \d+: Best Value\s*=\s*([0-9.]+)", line)
            if m:
                accs.append(float(m.group(1)))
    if not accs:
        return None, None, None
    times = []
    d = log_dir(dataset, fusion, tag)
    for fold in range(5):
        sp = d / f"train_stats_{fold}.txt"
        if not sp.exists():
            continue
        last = None
        with open(sp) as f:
            for line in f:
                m = re.match(r"^Inference ms/sample:\s*([0-9.]+)", line)
                if m:
                    last = float(m.group(1))
        if last is not None:
            times.append(last)
    return sum(accs) / len(accs), max(accs), (sum(times) / len(times) if times else None)


def parse_test_slurm(jid: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not jid:
        return None, None
    for lf in SLURM.glob(f"*_{jid}.txt"):
        with open(lf) as f:
            for line in f:
                if line.startswith("TEST_RESULT"):
                    kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                    return (
                        float(kv["acc"]) if "acc" in kv else None,
                        float(kv["ms_per_sample"]) if "ms_per_sample" in kv else None,
                    )
    return None, None


def step_b_row(dataset: str, fusion: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    if not STEP_B.exists():
        return None, None, None, None
    with open(STEP_B) as f:
        for row in csv.DictReader(f):
            if row.get("model") == "CogniAligned" and row.get("dataset") == dataset and row.get("arch") == fusion:
                def fn(k):
                    try:
                        return float(row[k]) if row.get(k) else None
                    except ValueError:
                        return None
                return fn("val_acc_mean"), fn("best_fold_acc"), fn("val_inference_ms_per_sample"), fn("test_acc")
    return None, None, None, None


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(s: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def all_train_done() -> bool:
    for _, _, _, align_tag in ALIGNMENTS:
        for fusion in FUSIONS:
            for dataset, _ in DATASETS:
                if not cell_complete(dataset, fusion, experiment_tag(align_tag, fusion)):
                    return False
    return True


def tests_ready(state: dict) -> bool:
    for align, _, _, align_tag in ALIGNMENTS:
        for fusion in FUSIONS:
            if not cell_complete("adresso", fusion, experiment_tag(align_tag, fusion)):
                return False
            tkey = f"test_adresso_{cell_key('adresso', align, fusion)}"
            tj = state.get("test_jobs", {}).get(tkey, {})
            if tj.get("test_acc") is not None:
                continue
            jid = tj.get("job_id")
            if jid:
                if not is_terminal(job_state(jid)):
                    return False
                acc, _ = parse_test_slurm(jid)
                if acc is not None:
                    tj["test_acc"] = acc
                    continue
                if job_state(jid).startswith("COMPLETED"):
                    return False
            else:
                return False
    return True


def submit_pending(state: dict) -> List[str]:
    ensure_configs()
    active = []
    state.setdefault("train_jobs", {})
    state.setdefault("test_jobs", {})

    for align, pilot_stem, _, align_tag in ALIGNMENTS:
        for fusion in FUSIONS:
            stem = f"tier_a_{pilot_stem.split('_', 1)[1]}_{fusion}"
            tag = experiment_tag(align_tag, fusion)
            for dataset, _ in DATASETS:
                key = cell_key(dataset, align, fusion)
                if cell_complete(dataset, fusion, tag):
                    continue
                meta = state["train_jobs"].get(key, {})
                jid = meta.get("job_id")
                if jid and not is_terminal(job_state(jid)):
                    active.append(jid)
                    continue
                if cell_complete(dataset, fusion, tag):
                    continue
                if jid and is_terminal(job_state(jid)):
                    log.warning(f"retry train {key} (was {job_state(jid)})")
                new = submit_train(dataset, stem)
                state["train_jobs"][key] = {"job_id": new, "stem": stem}
                if new:
                    active.append(new)

    if all_train_done():
        for align, pilot_stem, _, align_tag in ALIGNMENTS:
            for fusion in FUSIONS:
                stem = f"tier_a_{pilot_stem.split('_', 1)[1]}_{fusion}"
                tkey = f"test_adresso_{cell_key('adresso', align, fusion)}"
                meta = state["test_jobs"].get(tkey, {})
                if meta.get("test_acc") is not None:
                    continue
                jid = meta.get("job_id")
                if jid:
                    if not is_terminal(job_state(jid)):
                        active.append(jid)
                        continue
                    acc, ms = parse_test_slurm(jid)
                    if acc is not None:
                        meta["test_acc"], meta["test_ms"] = acc, ms
                        continue
                    if job_state(jid).startswith("COMPLETED"):
                        log.warning(f"retry test {tkey} (no TEST_RESULT)")
                if not cell_complete("adresso", fusion, experiment_tag(align_tag, fusion)):
                    continue
                new = submit_test(stem)
                state["test_jobs"][tkey] = {"job_id": new}
                if new:
                    active.append(new)

    save_state(state)
    return active


def build_csv(state: dict) -> None:
    cols = [
        "phase", "dataset", "alignment", "fusion", "lr",
        "val_acc_mean", "best_fold_acc", "val_inference_ms_per_sample",
        "test_acc", "test_inference_ms_per_sample", "note",
    ]
    rows = []
    for dataset, _ in DATASETS:
        for fusion in FUSIONS:
            v, b, ms, test_acc = step_b_row(dataset, fusion)
            rows.append({
                "phase": "C-tierA", "dataset": dataset, "alignment": "word_token", "fusion": fusion,
                "lr": "2e-5",
                "val_acc_mean": f"{v:.6f}" if v else "", "best_fold_acc": f"{b:.6f}" if b else "",
                "val_inference_ms_per_sample": f"{ms:.4f}" if ms else "",
                "test_acc": f"{test_acc:.6f}" if test_acc and dataset == "adresso" else "",
                "test_inference_ms_per_sample": "", "note": "reused_stepB",
            })
    for align, pilot_stem, _, align_tag in ALIGNMENTS:
        for fusion in FUSIONS:
            stem = f"tier_a_{pilot_stem.split('_', 1)[1]}_{fusion}"
            tag = experiment_tag(align_tag, fusion)
            for dataset, _ in DATASETS:
                v, b, ms = parse_summary(dataset, fusion, tag)
                tkey = f"test_adresso_{cell_key('adresso', align, fusion)}"
                tj = state.get("test_jobs", {}).get(tkey, {})
                test_acc, test_ms = tj.get("test_acc"), tj.get("test_ms")
                if test_acc is None and tj.get("job_id"):
                    test_acc, test_ms = parse_test_slurm(tj["job_id"])
                note = "" if v is not None else "incomplete"
                rows.append({
                    "phase": "C-tierA", "dataset": dataset, "alignment": align, "fusion": fusion, "lr": "2e-5",
                    "val_acc_mean": f"{v:.6f}" if v else "", "best_fold_acc": f"{b:.6f}" if b else "",
                    "val_inference_ms_per_sample": f"{ms:.4f}" if ms else "",
                    "test_acc": f"{test_acc:.6f}" if test_acc and dataset == "adresso" else "",
                    "test_inference_ms_per_sample": f"{test_ms:.4f}" if test_ms and dataset == "adresso" else "",
                    "note": note,
                })
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    es = CSV_OUT.with_name(CSV_OUT.stem + "_es.csv")
    num_re = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
    with open(CSV_OUT, newline="") as fin, open(es, "w", newline="") as fout:
        r, w = csv.reader(fin), csv.writer(fout, delimiter=";")
        for row in r:
            w.writerow([c.replace(".", ",") if num_re.match(c.strip()) else c for c in row])
    log.info(f"Wrote {CSV_OUT} ({len(rows)} rows)")


def monitor_loop(quiet: bool) -> None:
    global log
    log = setup_log(quiet=quiet)
    log.info("Tier A monitor started")
    while True:
        state = load_state()
        active = submit_pending(state)
        state = load_state()
        for jid in list(active):
            st = job_state(jid)
            if is_terminal(st):
                active.remove(jid)
                log.info(f"job {jid} -> {st}")
        if not active and all_train_done():
            state = load_state()
            submit_pending(state)
            state = load_state()
            for tkey, tj in list(state.get("test_jobs", {}).items()):
                if tj.get("test_acc") is None and tj.get("job_id"):
                    acc, ms = parse_test_slurm(tj["job_id"])
                    if acc is not None:
                        tj["test_acc"], tj["test_ms"] = acc, ms
            save_state(state)
            if tests_ready(state):
                build_csv(state)
                state["finished_at"] = datetime.now().isoformat(timespec="seconds")
                save_state(state)
                log.info("Tier A complete.")
                return
        save_state(state)
        time.sleep(POLL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", action="store_true", help="Poll until done (quiet log to file only with --quiet)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--submit-only", action="store_true")
    args = ap.parse_args()
    global log
    log = setup_log(quiet=args.quiet and args.monitor)
    ensure_configs()
    state = load_state()
    active = submit_pending(state)
    log.info(f"Submitted/tracking {len(active)} jobs")
    if args.submit_only:
        return
    if args.monitor:
        monitor_loop(quiet=True)
    else:
        log.info("Use --monitor for completion loop")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
