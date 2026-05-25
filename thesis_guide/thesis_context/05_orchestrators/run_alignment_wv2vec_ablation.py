#!/usr/bin/env python3
"""Alignment × Wav2Vec layer ablation: CogniAligned + SER, bicross fusion, 3 datasets.

Alignments: word_token, token_token_uniform, token_token_char (proportional).
Wav2Vec: final (last layer), weighted (learnable layer mix).
SER uses segment-level audio (no token alignment); rows are tagged alignment=segment_level.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
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
AD = BASE / "ad-detection"
COG_SLURM = COG / "logs" / "slurm"
AD_SLURM = AD / "wandb"
PRE = COG / "modules" / "preprocess"
CFG_DIR = COG / "modules" / "configs" / "ablation" / "align_wv2vec"
TEST_DIR = COG / "modules" / "configs" / "ablation" / "align_wv2vec_test"
STATE_FILE = BASE / "alignment_wv2vec_state.json"
ORCH_LOG = BASE / "alignment_wv2vec_ablation.log"
CSV_OUT = BASE / "alignment_wv2vec_ablation.csv"
STEP_B_CSV = BASE / "step_b_results.csv"
WV_CSV = BASE / "wav2vec_layer_ablation.csv"
TIER_A_CSV = BASE / "alignment_tier_a.csv"

POLL = 90
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}
FUSION = "bicross"

# (alignment_key, align_suffix, short_name)
ALIGNMENTS = [
    ("word_token", "", "wt"),
    ("token_token_uniform", "_tokalign", "tokuni"),
    ("token_token_char", "_tokalignchar", "tokchar"),
]
# (label, layer_mode, layer_suffix)
WAV2VEC = [
    ("final", "last", ""),
    ("weighted", "weighted", "_wlayer"),
]

DATASETS = [
    ("adresso", "modules/main.py", "modules/configs/thesis_bicross.yaml"),
    ("amyloid", "modules/amyloid/main.py", "modules/configs/amyloid/default.yaml"),
    ("ppa", "modules/ppa/main.py", "modules/configs/ppa/default.yaml"),
]

ADRESSO_TEXT = BASE / "adresso" / "ADReSSo21" / "diagnosis" / "train" / "text"
WAB_TEXT = BASE / "WAB_samples" / "text"

# CogniAligned log experiment tags for cells reused from prior steps
REUSE_COG_TAG = {
    ("word_token", "final"): "",
    ("word_token", "weighted"): "_wlayer",
    ("token_token_uniform", "final"): "_ablCtA_tokuni_bicross",
    ("token_token_char", "final"): "_ablCtA_tokchar_bicross",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ORCH_LOG), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def _load_rsb():
    spec = importlib.util.spec_from_file_location("rsb", BASE / "run_step_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RSB = _load_rsb()


def job_state(jid: str) -> str:
    return RSB.job_state(jid)


def is_terminal(st: str) -> bool:
    return any(st.startswith(t) for t in TERMINAL)


def audio_file_suffix(align_suffix: str, layer_suffix: str) -> str:
    return align_suffix + layer_suffix


def experiment_tag(align_short: str, wv_label: str) -> str:
    wv_short = "wt" if wv_label == "weighted" else "fin"
    return f"_ablCwv_{align_short}_{wv_short}_bicross"


def cog_log_dir(dataset: str, tag: str) -> Path:
    return COG / "logs" / f"{dataset}_distil_wav2vec2_P_{FUSION}{tag}_mean"


def cog_cell_complete(dataset: str, tag: str) -> bool:
    s = cog_log_dir(dataset, tag) / "cross_fold_summary.txt"
    if not s.exists():
        return False
    with open(s) as f:
        return sum(1 for line in f if line.startswith("Fold ")) >= 5


def ser_cell_complete(dataset: str, wv_label: str) -> bool:
    label = "bicrossattn" + ("_wvlast" if wv_label == "final" else "")
    ds = "ppa" if dataset == "ppa" else dataset
    d = AD / "models" / f"thesis_{ds}_{label}"
    if not d.exists():
        return False
    return all((d / f"best_{ds}_model_fold{f}.pt").exists() for f in range(5))


def needs_preprocess(align_key: str, wv_label: str) -> bool:
    if wv_label != "weighted":
        return False
    if align_key == "word_token":
        return False
    return True


def embed_count(root: Path, suffix: str, min_n: int) -> bool:
    if not root.exists():
        return False
    pat = f"*distil_pauses_audio{suffix}.pt"
    return len(list(root.rglob(pat))) >= min_n


def preprocess_ready(align_key: str, wv_label: str) -> bool:
    if not needs_preprocess(align_key, wv_label):
        return True
    _, align_suf, _ = next(a for a in ALIGNMENTS if a[0] == align_key)
    _, _, layer_suf = next(w for w in WAV2VEC if w[0] == wv_label)
    suf = audio_file_suffix(align_suf, layer_suf)
    ad_ok = embed_count(ADRESSO_TEXT, suf, 80)
    wab_ok = embed_count(WAB_TEXT, suf, 150)
    return ad_ok and wab_ok


def needs_train_cog(align_key: str, wv_label: str) -> bool:
    key = (align_key, wv_label)
    if key in REUSE_COG_TAG:
        tag = REUSE_COG_TAG[key]
        return False
    return True


def needs_train_ser(wv_label: str) -> bool:
    return False


def cog_train_tag(align_key: str, wv_label: str) -> str:
    if not needs_train_cog(align_key, wv_label):
        return REUSE_COG_TAG[(align_key, wv_label)]
    _, _, short = next(a for a in ALIGNMENTS if a[0] == align_key)
    return experiment_tag(short, wv_label)


def sbatch_wrap(job_name: str, wrap: str, mem: str = "64GB", hours: str = "08:00:00", gpu: str = "1") -> Optional[str]:
    COG_SLURM.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sbatch", f"--job-name={job_name[:64]}", f"--output={COG_SLURM}/%x_%j.txt",
        "-A", "veu", "-p", "veu", "--cpus-per-task=16", f"--mem={mem}", f"--gres=gpu:{gpu}",
        "--ntasks=1", "--exclude=veuc05,veuc01,veuc11,veuc10", f"--time={hours}", "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch {job_name}: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    return m.group(1) if m else None


def ensure_configs() -> None:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    for align_key, align_suf, short in ALIGNMENTS:
        for wv_label, layer_mode, layer_suf in WAV2VEC:
            if not needs_train_cog(align_key, wv_label):
                continue
            tag = experiment_tag(short, wv_label)
            for dataset, _, base_cfg in DATASETS:
                out = CFG_DIR / f"alignwv_{short}_{wv_label}_{dataset}.yaml"
                if out.exists():
                    continue
                with open(COG / base_cfg) as f:
                    cfg = yaml.safe_load(f)
                cfg.setdefault("data", {})
                cfg.setdefault("model", {})
                cfg["data"]["alignment_suffix"] = align_suf
                cfg["data"]["audio_layer_suffix"] = layer_suf
                cfg["model"]["fusion"] = FUSION
                cfg["model"]["n_heads"] = 8
                cfg["model"]["experiment_tag"] = tag
                cfg["model"]["wav2vec2_layer_mode"] = layer_mode
                cfg["model"]["use_weighted_audio_layers"] = wv_label == "weighted"
                cfg["model"]["audio_num_hidden_layers"] = 12
                if dataset != "adresso":
                    cfg["dataset_prefix"] = dataset
                with open(out, "w") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False)


def submit_preprocess_adresso(align_key: str, test: bool) -> Optional[str]:
    flag = " --test" if test else ""
    name = f"pre_awv_adr_{align_key[:12]}_wt{'t' if test else 'tr'}"[:48]
    wrap = (
        f"cd '{PRE}' && export HF_HOME='{COG}/.cache/huggingface' PYTHONUNBUFFERED=1 && "
        f"'{COG}/.venv/bin/python' -u run_preprocessing.py "
        f"--alignment_mode {align_key} --wav2vec2_layer_mode weighted{flag}"
    )
    return sbatch_wrap(name, wrap, hours="10:00:00")


def submit_preprocess_wab(align_key: str) -> Optional[str]:
    name = f"pre_awv_wab_{align_key[:12]}_wt"[:48]
    wrap = (
        f"cd '{PRE}' && export HF_HOME='{COG}/.cache/huggingface' PYTHONUNBUFFERED=1 && "
        f"'{COG}/.venv/bin/python' -u run_preprocessing_wab.py "
        f"--alignment_mode {align_key} --wav2vec2_layer_mode weighted"
    )
    return sbatch_wrap(name, wrap, hours="12:00:00")


def submit_cog_train(dataset: str, align_key: str, wv_label: str) -> Optional[str]:
    _, _, short = next(a for a in ALIGNMENTS if a[0] == align_key)
    stem = f"alignwv_{short}_{wv_label}"
    main_py = next(m for d, m, _ in DATASETS if d == dataset)
    cfg = f"modules/configs/ablation/align_wv2vec/{stem}_{dataset}.yaml"
    name = f"awv_cog_{short}_{wv_label}_{dataset}"[:48]
    wrap = (
        f"cd '{COG}' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        f"HF_HOME='{COG}/.cache/huggingface' WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && '{COG}/.venv/bin/python' -u {main_py} --config {cfg}"
    )
    return sbatch_wrap(name, wrap)


def submit_cog_test(align_key: str, wv_label: str) -> Optional[str]:
    _, _, short = next(a for a in ALIGNMENTS if a[0] == align_key)
    stem = f"alignwv_{short}_{wv_label}"
    src = CFG_DIR / f"{stem}_adresso.yaml"
    if not src.exists():
        return None
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_prefix"] = "adresso"
    TEST_DIR.mkdir(parents=True, exist_ok=True)
    dst = TEST_DIR / src.name
    with open(dst, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    rel = dst.relative_to(COG)
    name = f"awv_test_{short}_{wv_label}_adr"[:48]
    wrap = (
        f"cd '{COG}' && export PYTHONUNBUFFERED=1 WANDB_MODE=disabled "
        f"HF_HOME='{COG}/.cache/huggingface' PYTHONPATH='{COG}/modules' && "
        f"'{COG}/.venv/bin/python' -u modules/test.py --config {rel}"
    )
    return sbatch_wrap(name, wrap, mem="32GB", hours="02:00:00")


def parse_cog_summary(dataset: str, tag: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = cog_log_dir(dataset, tag) / "cross_fold_summary.txt"
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
    d = cog_log_dir(dataset, tag)
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
    for lf in COG_SLURM.glob(f"*_{jid}.txt"):
        with open(lf) as f:
            for line in f:
                if line.startswith("TEST_RESULT"):
                    kv = dict(t.split("=", 1) for t in line.split() if "=" in t)
                    return (
                        float(kv["acc"]) if "acc" in kv else None,
                        float(kv["ms_per_sample"]) if "ms_per_sample" in kv else None,
                    )
    return None, None


def load_wv_index() -> Dict[Tuple[str, str, str], dict]:
    idx = {}
    if not WV_CSV.exists():
        return idx
    with open(WV_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("fusion") != FUSION:
                continue
            idx[(row["model"], row["dataset"], row["wav2vec_layers"])] = row
    return idx


def load_step_b_index() -> Dict[Tuple[str, str, str], dict]:
    idx = {}
    if not STEP_B_CSV.exists():
        return idx
    with open(STEP_B_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("arch") != FUSION:
                continue
            idx[(row["model"], row["dataset"], row["arch"])] = row
    return idx


def load_tier_a_test(align_key: str) -> Tuple[Optional[float], Optional[float]]:
    if not TIER_A_CSV.exists():
        return None, None
    with open(TIER_A_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("dataset") == "adresso" and row.get("alignment") == align_key and row.get("fusion") == FUSION:
                try:
                    acc = float(row["test_acc"]) if row.get("test_acc") else None
                    ms = float(row["test_inference_ms_per_sample"]) if row.get("test_inference_ms_per_sample") else None
                    return acc, ms
                except ValueError:
                    return None, None
    return None, None


def backfill_reused_tests(state: dict) -> None:
    """Populate test_jobs for cells reused from Step B / WvLayer / Tier A."""
    state.setdefault("test_jobs", {})
    sb = load_step_b_index()
    wv = load_wv_index()
    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            tkey = f"test_CogniAligned_{align_key}_{wv_label}"
            tj = state["test_jobs"].setdefault(tkey, {})
            if tj.get("test_acc") is not None:
                continue
            if align_key == "word_token" and wv_label == "final":
                row = sb.get(("CogniAligned", "adresso", FUSION))
                if row and row.get("test_acc"):
                    tj.update({"test_acc": float(row["test_acc"]), "test_ms": float(row.get("test_inference_ms_per_sample") or 0), "note": "reused_stepB"})
            elif align_key == "word_token" and wv_label == "weighted":
                row = wv.get(("CogniAligned", "adresso", wv_label))
                if row and row.get("test_acc"):
                    tj.update({"test_acc": float(row["test_acc"]), "test_ms": float(row.get("test_inference_ms_per_sample") or 0), "note": "reused_wv_layer"})
            elif wv_label == "final" and align_key in ("token_token_uniform", "token_token_char"):
                acc, ms = load_tier_a_test(align_key)
                if acc is not None:
                    tj.update({"test_acc": acc, "test_ms": ms, "note": "reused_tierA"})


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(s: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def test_acc_ready(state: dict, model: str, align_key: str, wv_label: str) -> bool:
    tj = state.get("test_jobs", {}).get(f"test_{model}_{align_key}_{wv_label}", {})
    if tj.get("test_acc") is not None:
        return True
    if tj.get("job_id"):
        acc, _ = parse_test_slurm(tj["job_id"])
        if acc is not None:
            return True
    if model == "SER":
        row = load_wv_index().get((model, "adresso", wv_label))
        return bool(row and row.get("test_acc"))
    if align_key == "word_token" and wv_label == "final":
        row = load_step_b_index().get((model, "adresso", FUSION))
        return bool(row and row.get("test_acc"))
    if align_key == "word_token" and wv_label == "weighted":
        row = load_wv_index().get((model, "adresso", wv_label))
        return bool(row and row.get("test_acc"))
    if wv_label == "final" and align_key in ("token_token_uniform", "token_token_char"):
        acc, _ = load_tier_a_test(align_key)
        return acc is not None
    return False


def all_done(state: dict) -> bool:
    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            for dataset, _, _ in DATASETS:
                if not cog_cell_complete(dataset, cog_train_tag(align_key, wv_label)):
                    return False
                if not ser_cell_complete(dataset, wv_label):
                    return False
            for model in ("CogniAligned", "SER"):
                if not test_acc_ready(state, model, align_key, wv_label):
                    return False
    return True


def submit_pending(state: dict) -> List[str]:
    ensure_configs()
    active: List[str] = []
    state.setdefault("preprocess_jobs", {})
    state.setdefault("train_jobs", {})
    state.setdefault("test_jobs", {})
    backfill_reused_tests(state)

    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            if not needs_preprocess(align_key, wv_label):
                continue
            if preprocess_ready(align_key, wv_label):
                continue
            for test in (False, True):
                key = f"pre_adr_{align_key}_{wv_label}_{'test' if test else 'train'}"
                meta = state["preprocess_jobs"].get(key, {})
                jid = meta.get("job_id")
                if jid and not is_terminal(job_state(jid)):
                    active.append(jid)
                    continue
                if jid and job_state(jid).startswith("COMPLETED"):
                    continue
                new = submit_preprocess_adresso(align_key, test)
                state["preprocess_jobs"][key] = {"job_id": new}
                if new:
                    active.append(new)
            wkey = f"pre_wab_{align_key}_{wv_label}"
            meta = state["preprocess_jobs"].get(wkey, {})
            jid = meta.get("job_id")
            if jid and not is_terminal(job_state(jid)):
                active.append(jid)
            elif not (jid and job_state(jid).startswith("COMPLETED")):
                new = submit_preprocess_wab(align_key)
                state["preprocess_jobs"][wkey] = {"job_id": new}
                if new:
                    active.append(new)

    pre_done = all(
        preprocess_ready(a, w) or not needs_preprocess(a, w)
        for a, _, _ in ALIGNMENTS
        for w, _, _ in WAV2VEC
    )
    if not pre_done:
        save_state(state)
        return active

    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            tag = cog_train_tag(align_key, wv_label)
            for dataset, _, _ in DATASETS:
                if not needs_train_cog(align_key, wv_label):
                    continue
                key = f"cog_{dataset}_{align_key}_{wv_label}"
                if cog_cell_complete(dataset, tag):
                    continue
                meta = state["train_jobs"].get(key, {})
                jid = meta.get("job_id")
                if jid and not is_terminal(job_state(jid)):
                    active.append(jid)
                    continue
                if jid and job_state(jid).startswith("COMPLETED") and cog_cell_complete(dataset, tag):
                    continue
                new = submit_cog_train(dataset, align_key, wv_label)
                state["train_jobs"][key] = {"job_id": new}
                if new:
                    active.append(new)

    sb = load_step_b_index()
    wv = load_wv_index()
    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            for model in ("CogniAligned", "SER"):
                tkey = f"test_{model}_{align_key}_{wv_label}"
                tj = state["test_jobs"].setdefault(tkey, {})
                if tj.get("test_acc") is not None:
                    continue
                if model == "SER":
                    row = wv.get((model, "adresso", wv_label))
                    if row and row.get("test_acc"):
                        try:
                            tj["test_acc"] = float(row["test_acc"])
                            tj["test_ms"] = float(row["test_inference_ms_per_sample"] or 0)
                            tj["note"] = "reused_wv_layer"
                        except ValueError:
                            pass
                        continue
                elif align_key == "word_token" and wv_label == "final":
                    row = sb.get((model, "adresso", FUSION))
                    if row and row.get("test_acc"):
                        try:
                            tj["test_acc"] = float(row["test_acc"])
                            tj["test_ms"] = float(row.get("test_inference_ms_per_sample") or 0)
                            tj["note"] = "reused_stepB"
                        except ValueError:
                            pass
                        continue
                if model == "CogniAligned" and not cog_cell_complete("adresso", cog_train_tag(align_key, wv_label)):
                    continue
                jid = tj.get("job_id")
                if jid and not is_terminal(job_state(jid)):
                    active.append(jid)
                    continue
                if jid:
                    acc, ms = parse_test_slurm(jid)
                    if acc is not None:
                        tj["test_acc"], tj["test_ms"] = acc, ms
                        continue
                if model == "CogniAligned" and needs_train_cog(align_key, wv_label):
                    new = submit_cog_test(align_key, wv_label)
                    tj["job_id"] = new
                    if new:
                        active.append(new)

    save_state(state)
    return active


def build_csv(state: dict) -> None:
    sb = load_step_b_index()
    wv = load_wv_index()
    cols = [
        "phase", "model", "dataset", "alignment", "wav2vec_layers", "fusion", "lr",
        "val_acc_mean", "best_fold_acc", "val_inference_ms_per_sample",
        "test_acc", "test_inference_ms_per_sample", "note",
    ]
    rows = []
    for align_key, _, _ in ALIGNMENTS:
        for wv_label, _, _ in WAV2VEC:
            for dataset, _, _ in DATASETS:
                for model in ("CogniAligned", "SER"):
                    note = ""
                    align_col = "segment_level" if model == "SER" else align_key
                    v = b = ms = test_acc = test_ms = None
                    if model == "CogniAligned":
                        tag = cog_train_tag(align_key, wv_label)
                        if (align_key, wv_label) == ("word_token", "final"):
                            note = "reused_stepB"
                        elif (align_key, wv_label) == ("word_token", "weighted"):
                            note = "reused_wv_layer"
                        elif (align_key, wv_label) in REUSE_COG_TAG:
                            note = "reused_tierA"
                        v, b, ms = parse_cog_summary(dataset, tag)

                        def _from_row(row: Optional[dict]) -> None:
                            nonlocal v, b, ms
                            if not row:
                                return
                            def fn(k):
                                try:
                                    return float(row[k]) if row.get(k) else None
                                except ValueError:
                                    return None
                            if v is None:
                                v = fn("val_acc_mean")
                            if b is None:
                                b = fn("best_fold_acc")
                            if ms is None:
                                ms = fn("val_inference_ms_per_sample")

                        if note == "reused_stepB":
                            _from_row(sb.get((model, dataset, FUSION)))
                        elif note in ("reused_wv_layer",):
                            _from_row(wv.get((model, dataset, wv_label)))
                    else:
                        wv_row = wv.get((model, dataset, wv_label))
                        if wv_row:
                            note = "reused_wv_layer"
                            def fn(k):
                                try:
                                    return float(wv_row[k]) if wv_row.get(k) else None
                                except ValueError:
                                    return None
                            v, b, ms = fn("val_acc_mean"), fn("best_fold_acc"), fn("val_inference_ms_per_sample")
                    tkey = f"test_{model}_{align_key}_{wv_label}"
                    tj = state.get("test_jobs", {}).get(tkey, {})
                    if dataset == "adresso":
                        if tj.get("test_acc") is not None:
                            test_acc, test_ms = tj.get("test_acc"), tj.get("test_ms")
                        elif tj.get("job_id"):
                            test_acc, test_ms = parse_test_slurm(tj["job_id"])
                        elif model == "SER":
                            wv_row = wv.get((model, "adresso", wv_label))
                            if wv_row and wv_row.get("test_acc"):
                                try:
                                    test_acc = float(wv_row["test_acc"])
                                    test_ms = float(wv_row.get("test_inference_ms_per_sample") or 0)
                                except ValueError:
                                    pass
                    if v is None and not note:
                        note = "incomplete"
                    rows.append({
                        "phase": "AlignWv",
                        "model": model,
                        "dataset": dataset,
                        "alignment": align_col,
                        "wav2vec_layers": wv_label,
                        "fusion": FUSION,
                        "lr": "2e-5" if model == "CogniAligned" else "5e-5",
                        "val_acc_mean": f"{v:.6f}" if v is not None else "",
                        "best_fold_acc": f"{b:.6f}" if b is not None else "",
                        "val_inference_ms_per_sample": f"{ms:.4f}" if ms is not None else "",
                        "test_acc": f"{test_acc:.6f}" if test_acc is not None and dataset == "adresso" else "",
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


def monitor_loop() -> None:
    log.info("Alignment×Wav2Vec monitor started")
    while True:
        state = load_state()
        active = submit_pending(state)
        state = load_state()
        if not active:
            backfill_reused_tests(state)
            for tkey, tj in list(state.get("test_jobs", {}).items()):
                if tj.get("test_acc") is None and tj.get("job_id"):
                    acc, ms = parse_test_slurm(tj["job_id"])
                    if acc is not None:
                        tj["test_acc"], tj["test_ms"] = acc, ms
            save_state(state)
            if all_done(state):
                build_csv(state)
                state["finished_at"] = datetime.now().isoformat(timespec="seconds")
                save_state(state)
                log.info("Alignment×Wav2Vec ablation complete.")
                return
        else:
            for jid in list(active):
                if is_terminal(job_state(jid)):
                    log.info(f"job {jid} -> {job_state(jid)}")
        save_state(state)
        time.sleep(POLL)


def main() -> None:
    ap = argparse.ArgumentParser(description="Alignment × Wav2Vec layer ablation (bicross)")
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--submit-only", action="store_true")
    ap.add_argument("--build-csv", action="store_true")
    args = ap.parse_args()
    ensure_configs()
    state = load_state()
    if args.build_csv:
        build_csv(state)
        return
    active = submit_pending(state)
    log.info(f"tracking {len(active)} jobs")
    if args.submit_only:
        return
    if args.monitor:
        monitor_loop()
    else:
        log.info("Use --monitor to poll until done")


if __name__ == "__main__":
    main()
