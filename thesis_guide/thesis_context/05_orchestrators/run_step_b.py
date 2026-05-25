#!/usr/bin/env python3
"""
Step B — Fusion ablation orchestrator (`thesis_ablation_plan.md`).

Coverage matrix:
    7 architectures × 2 models (SER, CogniAligned) × 3 datasets = 42 training cells
    + 14 ADReSSo test-inference cells (7 archs × 2 models)

Architectures (canonical key → SER seq_to_seq_method / CogniAligned fusion / # heads):
    selfattn      SelfAttention                 selfattn      1
    mhselfattn    MultiHeadAttention            mhselfattn    4
    transformerpe TransformerPosEnc             transformerpe 4
    cross         CrossAttention                cross         4
    crossgated    GatedCrossAttention           crossgated    4
    bicross       BidirectionalCrossAttention   bicross       4
    mamba         MambaSeqToSeq                 mamba         4

The orchestrator:
  1. Audits which cells already have completed training artifacts on disk.
  2. Submits training jobs for the missing cells. SER → 5 fold-jobs each via the
     existing `<task>/run_train_generic.sh` script. CogniAligned → 1 sbatch
     `--wrap` job that runs 5-fold CV inside `main.py`.
  3. Waits for all training jobs to terminate.
  4. Submits 14 ADReSSo test-inference jobs (one per arch × model). For
     CogniAligned it calls modules/test.py with the existing thesis configs
     (after writing a `dataset_prefix: adresso` patched copy so the logs dir
     can be resolved). For SER it calls
     `adresso_classify/test_adresso.py --model_paths fold0..fold4`.
  5. Builds `step_b_results.csv` aggregating per-cell metrics.

Usage:
    python3 run_step_b.py
Run in the background and tail `step_b.log`.
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE        = Path("/home/usuaris/veussd/roger.esteve.sanchez")
AD          = BASE / "ad-detection"
COG         = BASE / "CogniAligned"
COG_SLURM   = COG / "logs" / "slurm"
AD_SLURM    = AD / "wandb"   # SER SLURM logs land here as slurm_<jobname>_<jid>.txt
COG_SLURM.mkdir(parents=True, exist_ok=True)

STATE_FILE = BASE / "step_b_state.json"
ORCH_LOG   = BASE / "step_b.log"
CSV_OUT    = BASE / "step_b_results.csv"

POLL_INTERVAL = 120  # seconds

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
# Experiment matrix
# ---------------------------------------------------------------------------
ARCHS = [
    # (key,           ser_method,                     ser_label,        cog_fusion,   n_heads, is_mamba)
    ("selfattn",      "SelfAttention",                "selfattn",       "selfattn",   1, False),
    ("mhselfattn",    "MultiHeadAttention",           "mhselfattn",     "mhselfattn", 4, False),
    ("transformerpe", "TransformerPosEnc",            "transformerpe",  "transformerpe", 4, False),
    ("cross",         "CrossAttention",               "crossattn",      "cross",      4, False),
    ("crossgated",    "GatedCrossAttention",          "gatedcrossattn", "crossgated", 4, False),
    ("bicross",       "BidirectionalCrossAttention",  "bicrossattn",    "bicross",    4, False),
    ("mamba",         "MambaSeqToSeq",                "mamba",          "mamba",      4, True),
]

DATASETS = [
    # (dataset_key, ser_subdir, ser_script, cog_main_py, cog_cfg_dir, num_classes, dataset_prefix)
    ("adresso", "adresso_classify", "adresso_classify/run_train_generic.sh", "modules/main.py",         "modules/configs",        1, "adresso"),
    ("amyloid", "amyloid",          "amyloid/run_train_generic.sh",           "modules/amyloid/main.py", "modules/configs/amyloid", 1, "amyloid"),
    ("ppa",     "ad_classification","ad_classification/run_train_generic.sh", "modules/ppa/main.py",     "modules/configs/ppa",    3, "ppa"),
]

COG_CFG_MAP = {
    "selfattn":       "thesis_selfattn.yaml",
    "mhselfattn":     "thesis_mhselfattn.yaml",
    "transformerpe":  "thesis_transformerpe.yaml",
    "cross":          "thesis_cross.yaml",
    "crossgated":     "thesis_crossgated.yaml",
    "bicross":        "thesis_bicross.yaml",
    "mamba":          None,  # uses default_mamba.yaml per dataset
}

COG_MAMBA_CFG = {
    "adresso": "modules/configs/default_mamba.yaml",
    "amyloid": "modules/configs/amyloid/default_mamba.yaml",
    "ppa":     "modules/configs/ppa/mamba.yaml",
}

# CogniAligned builds the log-dir name from `model.fusion` in the YAML, so we
# need to map our canonical Step-B arch keys to the *actual* fusion string
# used inside each thesis_*.yaml. The only mismatch is `transformerpe`, whose
# config keeps `fusion: concat` but enables positional encoding on top.
COG_LOG_FUSION = {
    "selfattn":      "selfattn",
    "mhselfattn":    "mhselfattn",
    "transformerpe": "concat",
    "cross":         "cross",
    "crossgated":    "crossgated",
    "bicross":       "bicross",
    "mamba":         "mamba",
}

# ---------------------------------------------------------------------------
# Filesystem audit
# ---------------------------------------------------------------------------
def ser_cell_complete(dataset_key: str, ser_label: str) -> bool:
    """A SER cell is complete iff models/thesis_<ds>_<label>/best_<ds>_model_fold{0..4}.pt all exist."""
    if dataset_key == "ppa":
        ds_label = "ppa"
    else:
        ds_label = dataset_key
    d = AD / "models" / f"thesis_{ds_label}_{ser_label}"
    if not d.exists():
        return False
    for fold in range(5):
        if not (d / f"best_{ds_label}_model_fold{fold}.pt").exists():
            return False
    return True


def cog_cell_complete(dataset_key: str, cog_fusion: str) -> bool:
    fusion_dir = COG_LOG_FUSION.get(cog_fusion, cog_fusion)
    d = COG / "logs" / f"{dataset_key}_distil_wav2vec2_P_{fusion_dir}_mean"
    summary = d / "cross_fold_summary.txt"
    if not summary.exists():
        return False
    with open(summary) as f:
        n = sum(1 for line in f if line.startswith("Fold "))
    return n >= 5


# ---------------------------------------------------------------------------
# SLURM helpers
# ---------------------------------------------------------------------------
def sbatch_submit(cmd: List[str], label: str) -> Optional[str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch failed for {label}: {p.stderr.strip()}")
        return None
    out = p.stdout.strip()
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        log.error(f"Could not parse sbatch output for {label}: {out}")
        return None
    jid = m.group(1)
    log.info(f"Submitted {label} as job {jid}")
    return jid


def submit_ser_training(dataset_key: str, ser_method: str, ser_label: str, n_heads: int, is_mamba: bool, fold: int) -> Optional[str]:
    ds = next(d for d in DATASETS if d[0] == dataset_key)
    script = ds[2]
    exp_tag = f"thesis_{dataset_key}_{ser_label}" if dataset_key != "ppa" else f"thesis_ppa_{ser_label}"
    job_name = f"sb_{dataset_key}_{ser_label}_f{fold}"
    export_kv = {
        "SEQ_METHOD": ser_method,
        "HEADS": str(n_heads),
        "LR": "0.00005",
        "MAX_EPOCHS": "25",
        "LR_PATIENCE": "10",
        "EXP_TAG": exp_tag,
    }
    if is_mamba:
        export_kv.update({
            "MAMBA_N_BLOCKS": "4",
            "MAMBA_D_STATE": "16",
            "MAMBA_D_CONV": "4",
            "MAMBA_EXPAND": "2",
        })
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_kv.items())
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--export={export_str}",
        script,
        str(fold),
    ]
    p = subprocess.run(cmd, cwd=str(AD), capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch SER {job_name} failed: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    if not m:
        log.error(f"sbatch SER {job_name}: unparseable output: {p.stdout.strip()}")
        return None
    jid = m.group(1)
    log.info(f"Submitted {job_name} as job {jid}")
    return jid


def submit_cog_training(dataset_key: str, cog_fusion: str) -> Optional[str]:
    ds = next(d for d in DATASETS if d[0] == dataset_key)
    main_py = ds[3]
    cfg_dir = ds[4]
    if cog_fusion == "mamba":
        cfg_rel = COG_MAMBA_CFG[dataset_key]
    else:
        cfg_rel = f"{cfg_dir}/{COG_CFG_MAP[cog_fusion]}"
    cfg_abs = COG / cfg_rel
    if not cfg_abs.exists():
        log.error(f"Missing CogniAligned config: {cfg_abs}")
        return None
    job_name = f"sb_cog_{dataset_key}_{cog_fusion}"
    wrap = (
        f"cd '{COG}' && "
        "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 && "
        f"export HF_HOME='{COG}/.cache/huggingface' "
        f"WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && "
        f"mkdir -p \"$HF_HOME\" \"$WANDB_DIR\" '{COG_SLURM}' && "
        f"'{COG}/.venv/bin/python' -u {main_py} --config {cfg_rel}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={COG_SLURM}/%x_%j.txt",
        "-A", "veu", "-p", "veu",
        "--cpus-per-task=16", "--mem=64GB", "--gres=gpu:1", "--ntasks=1",
        "--exclude=veuc05,veuc01,veuc11,veuc10",
        "--time=06:00:00",
        "--wrap", wrap,
    ]
    return sbatch_submit(cmd, job_name)


def submit_ser_test_inference(dataset_key: str, arch: dict) -> Optional[str]:
    """Submit ADReSSo test inference for a SER cell using all 5 fold checkpoints."""
    if dataset_key != "adresso":
        return None
    ser_label = arch["ser_label"]
    fold_ckpts = [
        AD / "models" / f"thesis_adresso_{ser_label}" / f"best_adresso_model_fold{fold}.pt"
        for fold in range(5)
    ]
    missing = [p for p in fold_ckpts if not p.exists()]
    if missing:
        log.warning(f"SER test inference {ser_label}: missing checkpoints {[str(p) for p in missing]}")
        return None

    job_name = f"sb_test_ser_{ser_label}"
    ckpts_arg = " ".join(f"'{p}'" for p in fold_ckpts)
    # Training-time arch values that the test-time Classifier must match.
    mamba_args = ""
    if arch["mamba"]:
        mamba_args = " --mamba_n_blocks 4 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2"
    wrap = (
        f"cd '{AD}' && "
        f". {AD}/.venv/bin/activate && "
        "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false && "
        f"python -u adresso_classify/test_adresso.py "
        f"--model_paths {ckpts_arg} "
        f"--exp_tag thesis_adresso_{ser_label} "
        f"--labels_csv '{AD}/task1.csv' "
        "--audio_dir '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/test-dist/audio' "
        "--transcription_dir '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/test-dist' "
        "--batch_size 8 --sample_rate 16000 --padding_type repetition_pad "
        "--window_secs 40 --stride_secs 40 "
        "--speech_feature_extractor WAV2VEC2_XLSR_300M --speech_feature_extractor_output_vectors_dimension 1024 "
        "--text_feature_extractor MODERN_BERT_BASE --text_feature_extractor_output_vectors_dimension 768 "
        "--speech_adapter LinearAdapter --speech_adapter_output_vectors_dimension 256 "
        "--text_adapter LinearAdapter --text_adapter_output_vectors_dimension 256 "
        f"--seq_to_seq_method {arch['ser_method']} --seq_to_seq_heads_number {arch['n_heads']} "
        "--seq_to_seq_input_dropout 0.3 --skip_connections "
        "--seq_to_one_method AttentionPooling --seq_to_one_input_dropout 0.3 "
        "--classifier_hidden_layers 1 --classifier_hidden_layers_width 256 --classifier_layer_drop_out 0.3 "
        "--acoustic_features_path '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/acoustic_features.csv'"
        f"{mamba_args}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={AD_SLURM}/slurm_%x_%j.txt",
        "-A", "veu", "-p", "veu",
        "--cpus-per-task=8", "--mem=32GB", "--gres=gpu:1", "--ntasks=1",
        "--exclude=veuc05,veuc01",
        "--time=02:00:00",
        "--wrap", wrap,
    ]
    return sbatch_submit(cmd, job_name)


def _patched_cog_config_for_test(cog_fusion: str) -> Path:
    """Write a temporary ADReSSo CogniAligned test config with dataset_prefix:'adresso'.

    The default `thesis_*.yaml` configs don't set dataset_prefix, which means
    `test.py.__main__` would look for `logs/<model_name>_mean/` instead of
    `logs/adresso_<model_name>_mean/`. We patch a copy on disk so test.py can
    resolve the right checkpoint dir without a symlink hack.
    """
    if cog_fusion == "mamba":
        src = COG / COG_MAMBA_CFG["adresso"]
    else:
        src = COG / "modules" / "configs" / COG_CFG_MAP[cog_fusion]
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_prefix"] = "adresso"
    out_dir = COG / "modules" / "configs" / "ablation" / "step_b_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"adresso_{cog_fusion}.yaml"
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return out_path


def submit_cog_test_inference(arch: dict) -> Optional[str]:
    cog_fusion = arch["cog_fusion"]
    cfg_path = _patched_cog_config_for_test(cog_fusion)
    cfg_rel = cfg_path.relative_to(COG)
    job_name = f"sb_test_cog_{cog_fusion}"
    wrap = (
        f"cd '{COG}' && "
        "export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        "WANDB_MODE=disabled WANDB_SILENT=true && "
        f"export HF_HOME='{COG}/.cache/huggingface' "
        f"WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && "
        f"'{COG}/.venv/bin/python' -u modules/test.py --config {cfg_rel}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={COG_SLURM}/%x_%j.txt",
        "-A", "veu", "-p", "veu",
        "--cpus-per-task=8", "--mem=32GB", "--gres=gpu:1", "--ntasks=1",
        "--exclude=veuc05,veuc01",
        "--time=01:00:00",
        "--wrap", wrap,
    ]
    return sbatch_submit(cmd, job_name)


# ---------------------------------------------------------------------------
# Job-state helpers (mirrors run_modality_ablation.py)
# ---------------------------------------------------------------------------
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}


def job_state(jid: str) -> str:
    p = subprocess.run(["sacct", "-X", "-n", "-P", "-o", "State", "-j", jid], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().splitlines()[0].split()[0]
    p = subprocess.run(["squeue", "-h", "-j", jid, "-o", "%T"], capture_output=True, text=True)
    if p.returncode == 0 and p.stdout.strip():
        return p.stdout.strip().splitlines()[0]
    return "UNKNOWN"


def wait_for_jobs(job_ids: List[str], label: str) -> Dict[str, str]:
    pending = set(job_ids)
    finals: Dict[str, str] = {}
    log.info(f"Waiting on {len(pending)} {label} jobs.")
    while pending:
        time.sleep(POLL_INTERVAL)
        done_now, running, queued = [], 0, 0
        for jid in list(pending):
            st = job_state(jid)
            if any(st.startswith(t) for t in TERMINAL):
                finals[jid] = st
                pending.discard(jid)
                done_now.append((jid, st))
            elif st == "RUNNING":
                running += 1
            elif st in ("PENDING", "REQUEUED"):
                queued += 1
        if done_now:
            for jid, st in done_now:
                log.info(f"  [{label}] job {jid} -> {st}")
        log.info(f"  [{label}] {len(pending)} still active (running={running}, queued={queued})")
    return finals


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def parse_cog_summary(dataset_key: str, cog_fusion: str) -> Tuple[Optional[float], Optional[float], int]:
    """Return (val_acc_mean, best_fold_acc, n_folds) from cross_fold_summary.txt."""
    fusion_dir = COG_LOG_FUSION.get(cog_fusion, cog_fusion)
    p = COG / "logs" / f"{dataset_key}_distil_wav2vec2_P_{fusion_dir}_mean" / "cross_fold_summary.txt"
    if not p.exists():
        return None, None, 0
    accs: List[float] = []
    with open(p) as f:
        for line in f:
            m = re.match(r"^Fold (\d+): Best Value\s*=\s*([0-9.]+)", line)
            if m:
                try:
                    accs.append(float(m.group(2)))
                except ValueError:
                    pass
    if not accs:
        return None, None, 0
    return sum(accs) / len(accs), max(accs), len(accs)


def parse_cog_val_inference_ms(dataset_key: str, cog_fusion: str) -> Optional[float]:
    fusion_dir = COG_LOG_FUSION.get(cog_fusion, cog_fusion)
    d = COG / "logs" / f"{dataset_key}_distil_wav2vec2_P_{fusion_dir}_mean"
    times = []
    for fold in range(5):
        sp = d / f"train_stats_{fold}.txt"
        if not sp.exists():
            continue
        last_val = None
        with open(sp) as f:
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


def _scan_cog_log_for_params(lf: Path) -> Tuple[Optional[int], Optional[int]]:
    total, trainable = None, None
    try:
        with open(lf) as f:
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
        return None, None
    return total, trainable


# Alias names used historically in CogniAligned slurm-log filenames per arch.
# Some old runs used SER-style names (crossattn, gatedcrossattn, bicrossattn)
# while others used CogniAligned-style fusion names (cross, crossgated, bicross).
_COG_ARCH_ALIASES = {
    "selfattn":      ["selfattn"],
    "mhselfattn":    ["mhselfattn"],
    "transformerpe": ["transformerpe", "concat"],
    "cross":         ["cross", "crossattn"],
    "crossgated":    ["crossgated", "gatedcrossattn"],
    "bicross":       ["bicross", "bicrossattn"],
    "mamba":         ["mamba"],
}


def parse_cog_params(dataset_key: str, arch_key: str, job_id: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """Find Total/Trainable params for a CogniAligned cell.

    Order of preference:
      1. If we have a Step-B `job_id`, read `CogniAligned/logs/slurm/*_<jid>.txt`.
      2. Else, glob CogniAligned/logs/slurm/ for any slurm log whose filename
         contains both the dataset key and one of the arch's name aliases.
    """
    if job_id:
        for lf in COG_SLURM.glob(f"*_{job_id}.txt"):
            total, trainable = _scan_cog_log_for_params(lf)
            if total is not None and trainable is not None:
                return total, trainable
    # Fall back to historical logs.
    candidates: List[Path] = []
    for alias in _COG_ARCH_ALIASES.get(arch_key, [arch_key]):
        candidates.extend(COG_SLURM.glob(f"*{dataset_key}*{alias}*.txt"))
        candidates.extend(COG_SLURM.glob(f"*{alias}*{dataset_key}*.txt"))
    seen: set = set()
    for lf in candidates:
        if str(lf) in seen:
            continue
        seen.add(str(lf))
        total, trainable = _scan_cog_log_for_params(lf)
        if total is not None and trainable is not None:
            return total, trainable
    return None, None


def parse_ser_per_fold(dataset_key: str, ser_label: str) -> Tuple[Optional[float], Optional[float], int, List[Tuple[int, int]], Optional[float]]:
    """Look at all `slurm_{thesis,sb}_<ds>_<label>_f<fold>_<jid>.txt` files and extract:
      - per-fold best Val Acc (last "New best model saved ... Val Acc: X.X%" line)
      - per-fold total/trainable parameters
      - per-fold inference/val_ms_per_sample (from wandb summary line)
    Returns (val_acc_mean, best_fold_acc, n_folds, [(total,trainable)] (best-effort), val_ms_mean).
    """
    if dataset_key == "ppa":
        ds_label = "ppa"
    else:
        ds_label = dataset_key
    per_fold_acc: Dict[int, float] = {}
    per_fold_params: List[Tuple[int, int]] = []
    per_fold_ms: List[float] = []
    # Accept both the original thesis-sweep naming (`slurm_thesis_*`) and the
    # Step-B orchestrator naming (`slurm_sb_*`).
    candidates: List[Path] = []
    for prefix in ("thesis", "sb"):
        candidates.extend(sorted(AD_SLURM.glob(f"slurm_{prefix}_{ds_label}_{ser_label}_f*.txt")))
    pattern = re.compile(rf"slurm_(?:thesis|sb)_{ds_label}_{ser_label}_f(\d+)_\d+\.txt$")
    for lf in candidates:
        m = pattern.search(str(lf))
        if not m:
            continue
        fold = int(m.group(1))
        try:
            with open(lf) as f:
                content = f.read()
        except OSError:
            continue
        # best val acc (last occurrence wins — early-stopping picks the highest)
        best_acc = None
        for am in re.finditer(r"Val Acc:\s*([0-9.]+)%", content):
            try:
                best_acc = float(am.group(1)) / 100.0
            except ValueError:
                pass
        if best_acc is not None:
            # Keep the highest across multiple runs of the same fold (re-submits).
            if (fold not in per_fold_acc) or best_acc > per_fold_acc[fold]:
                per_fold_acc[fold] = best_acc
        # parameters
        total = trainable = None
        for line in content.splitlines():
            m_t = re.match(r"\s*Total parameters:\s*([0-9,]+)", line)
            m_tr = re.match(r"\s*Trainable parameters:\s*([0-9,]+)", line)
            if m_t and total is None:
                total = int(m_t.group(1).replace(",", ""))
            if m_tr and trainable is None:
                trainable = int(m_tr.group(1).replace(",", ""))
        if total is not None and trainable is not None and (total, trainable) not in per_fold_params:
            per_fold_params.append((total, trainable))
        # wandb summary inference/val_ms_per_sample line
        # Lines like "wandb:      inference/val_ms_per_sample 1.0404"
        for sm in re.finditer(r"inference/val_ms_per_sample\s+([0-9.]+)", content):
            try:
                per_fold_ms.append(float(sm.group(1)))
            except ValueError:
                pass
    if not per_fold_acc:
        return None, None, 0, per_fold_params, (sum(per_fold_ms)/len(per_fold_ms)) if per_fold_ms else None
    accs = list(per_fold_acc.values())
    return sum(accs) / len(accs), max(accs), len(accs), per_fold_params, (sum(per_fold_ms)/len(per_fold_ms)) if per_fold_ms else None


def parse_test_result_from_slurm(prefix: str, job_id: Optional[str], in_dir: Path) -> Dict[str, Optional[float]]:
    if not job_id:
        return {"acc": None, "ms_per_sample": None}
    candidates = list(in_dir.glob(f"*_{job_id}.txt"))
    candidates += list(in_dir.glob(f"*{job_id}*.txt"))
    seen = set()
    for lf in candidates:
        if str(lf) in seen:
            continue
        seen.add(str(lf))
        try:
            with open(lf) as f:
                for line in f:
                    if line.startswith("TEST_RESULT"):
                        kv = dict(t.split("=", 1) for t in line.strip().split() if "=" in t)
                        out = {"acc": None, "ms_per_sample": None}
                        try: out["acc"] = float(kv.get("acc", "nan"))
                        except ValueError: pass
                        try: out["ms_per_sample"] = float(kv.get("ms_per_sample", "nan"))
                        except ValueError: pass
                        return out
        except OSError:
            continue
    return {"acc": None, "ms_per_sample": None}


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------
def save_state(s: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# CSV builder
# ---------------------------------------------------------------------------
def build_csv(state: dict) -> None:
    rows: List[Dict[str, str]] = []
    for arch_t in ARCHS:
        akey, ser_method, ser_label, cog_fusion, n_heads, is_mamba = arch_t
        arch = {"key": akey, "ser_method": ser_method, "ser_label": ser_label,
                "cog_fusion": cog_fusion, "n_heads": n_heads, "mamba": is_mamba}
        for dataset_key, _, _, _, _, num_classes, _ in DATASETS:
            # ── SER row ──
            ser_val_mean, ser_best, ser_n, ser_params, ser_val_ms = parse_ser_per_fold(dataset_key, ser_label)
            ser_total, ser_train = (ser_params[0] if ser_params else (None, None))
            ser_test = state.get("test_jobs", {}).get(f"ser_{dataset_key}_{akey}", {})
            note_ser: List[str] = []
            if ser_n and ser_n < 5: note_ser.append(f"folds={ser_n}/5")
            if not ser_n: note_ser.append("no_train_artifacts")
            rows.append({
                "phase": "B",
                "model": "SER",
                "dataset": dataset_key,
                "arch": akey,
                "ser_method": ser_method,
                "cog_fusion": "",
                "n_heads": str(n_heads),
                "lr": "5e-5",
                "total_params": str(ser_total) if ser_total is not None else "",
                "trainable_params": str(ser_train) if ser_train is not None else "",
                "val_acc_mean": f"{ser_val_mean:.6f}" if ser_val_mean is not None else "",
                "best_fold_acc": f"{ser_best:.6f}" if ser_best is not None else "",
                "val_inference_ms_per_sample": f"{ser_val_ms:.4f}" if ser_val_ms is not None else "",
                "test_acc": f"{ser_test.get('test_acc'):.6f}" if isinstance(ser_test.get("test_acc"), float) else "",
                "test_inference_ms_per_sample": f"{ser_test.get('test_ms_per_sample'):.4f}" if isinstance(ser_test.get("test_ms_per_sample"), float) else "",
                "note": ";".join(note_ser),
            })
            # ── CogniAligned row ──
            cog_val_mean, cog_best, cog_n = parse_cog_summary(dataset_key, cog_fusion)
            cog_val_ms = parse_cog_val_inference_ms(dataset_key, cog_fusion)
            train_jid = state.get("train_jobs", {}).get(f"cog_{dataset_key}_{cog_fusion}", {}).get("job_id")
            cog_total, cog_train = parse_cog_params(dataset_key, akey, train_jid)
            cog_test = state.get("test_jobs", {}).get(f"cog_{dataset_key}_{akey}", {})
            note_cog: List[str] = []
            if cog_n and cog_n < 5: note_cog.append(f"folds={cog_n}/5")
            if not cog_n: note_cog.append("no_train_artifacts")
            rows.append({
                "phase": "B",
                "model": "CogniAligned",
                "dataset": dataset_key,
                "arch": akey,
                "ser_method": "",
                "cog_fusion": cog_fusion,
                "n_heads": str(n_heads),
                "lr": "2e-5",
                "total_params": str(cog_total) if cog_total is not None else "",
                "trainable_params": str(cog_train) if cog_train is not None else "",
                "val_acc_mean": f"{cog_val_mean:.6f}" if cog_val_mean is not None else "",
                "best_fold_acc": f"{cog_best:.6f}" if cog_best is not None else "",
                "val_inference_ms_per_sample": f"{cog_val_ms:.4f}" if cog_val_ms is not None else "",
                "test_acc": f"{cog_test.get('test_acc'):.6f}" if isinstance(cog_test.get("test_acc"), float) else "",
                "test_inference_ms_per_sample": f"{cog_test.get('test_ms_per_sample'):.4f}" if isinstance(cog_test.get("test_ms_per_sample"), float) else "",
                "note": ";".join(note_cog),
            })

    columns = [
        "phase", "model", "dataset", "arch", "ser_method", "cog_fusion",
        "n_heads", "lr", "total_params", "trainable_params",
        "val_acc_mean", "best_fold_acc", "val_inference_ms_per_sample",
        "test_acc", "test_inference_ms_per_sample", "note",
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
    state.setdefault("train_jobs", {})  # key: "ser_<ds>_<arch>_f<fold>" or "cog_<ds>_<arch>" -> {job_id, state}
    state.setdefault("test_jobs", {})   # key: "ser_<ds>_<arch>" or "cog_<ds>_<arch>" -> {job_id, state, test_acc, test_ms_per_sample}
    state.setdefault("started_at", datetime.now().isoformat(timespec="seconds"))

    # ─── Phase 1: audit + submit missing training ──────────────────────────
    log.info("=== Phase 1: auditing missing training cells ===")
    train_ids_to_wait: List[str] = []
    TERMINAL_STATES = TERMINAL  # alias for clarity
    for arch_t in ARCHS:
        akey, ser_method, ser_label, cog_fusion, n_heads, is_mamba = arch_t
        for dataset_key, *_ in DATASETS:
            # SER
            if ser_cell_complete(dataset_key, ser_label):
                log.info(f"  SER {dataset_key}/{akey}: complete on disk → reuse")
            else:
                for fold in range(5):
                    fkey = f"ser_{dataset_key}_{akey}_f{fold}"
                    existing = state["train_jobs"].get(fkey, {})
                    if existing.get("job_id"):
                        # On resume, wait for any still-pending state. Terminal-state
                        # jobs are treated as "done" — handled in CSV builder.
                        st = job_state(existing["job_id"])
                        if not any(st.startswith(t) for t in TERMINAL_STATES):
                            log.info(f"  resume: {fkey} job {existing['job_id']} state={st} → waiting")
                            train_ids_to_wait.append(existing["job_id"])
                        else:
                            log.info(f"  resume: {fkey} job {existing['job_id']} already {st}")
                            existing["state"] = st
                        continue
                    jid = submit_ser_training(dataset_key, ser_method, ser_label, n_heads, is_mamba, fold)
                    state["train_jobs"][fkey] = {"job_id": jid, "state": "SUBMITTED" if jid else "SUBMIT_FAILED"}
                    if jid:
                        train_ids_to_wait.append(jid)
                    save_state(state)
            # CogniAligned
            if cog_cell_complete(dataset_key, cog_fusion):
                log.info(f"  CogniAligned {dataset_key}/{akey}: complete on disk → reuse")
            else:
                ckey = f"cog_{dataset_key}_{cog_fusion}"
                existing = state["train_jobs"].get(ckey, {})
                if existing.get("job_id"):
                    st = job_state(existing["job_id"])
                    if not any(st.startswith(t) for t in TERMINAL_STATES):
                        log.info(f"  resume: {ckey} job {existing['job_id']} state={st} → waiting")
                        train_ids_to_wait.append(existing["job_id"])
                    else:
                        log.info(f"  resume: {ckey} job {existing['job_id']} already {st}")
                        existing["state"] = st
                else:
                    jid = submit_cog_training(dataset_key, cog_fusion)
                    state["train_jobs"][ckey] = {"job_id": jid, "state": "SUBMITTED" if jid else "SUBMIT_FAILED"}
                    if jid:
                        train_ids_to_wait.append(jid)
                    save_state(state)

    # ─── Phase 2: wait for training ────────────────────────────────────────
    if train_ids_to_wait:
        log.info(f"=== Phase 2: waiting for {len(train_ids_to_wait)} training jobs ===")
        finals = wait_for_jobs(train_ids_to_wait, "train")
        for key, meta in state["train_jobs"].items():
            if meta.get("job_id") in finals:
                meta["state"] = finals[meta["job_id"]]
        save_state(state)
    else:
        log.info("=== Phase 2: no new training jobs to wait on ===")

    # ─── Phase 3: interim CSV (training only) ──────────────────────────────
    log.info("=== Phase 3: building interim CSV (training only) ===")
    build_csv(state)

    # ─── Phase 4: ADReSSo test inference ───────────────────────────────────
    log.info("=== Phase 4: submitting ADReSSo test inference (14 cells) ===")
    test_ids_to_wait: List[str] = []
    for arch_t in ARCHS:
        akey, ser_method, ser_label, cog_fusion, n_heads, is_mamba = arch_t
        arch = {"key": akey, "ser_method": ser_method, "ser_label": ser_label,
                "cog_fusion": cog_fusion, "n_heads": n_heads, "mamba": is_mamba}
        # SER ADReSSo
        skey = f"ser_adresso_{akey}"
        existing = state["test_jobs"].get(skey, {})
        if existing.get("job_id"):
            st = job_state(existing["job_id"])
            if not any(st.startswith(t) for t in TERMINAL_STATES):
                log.info(f"  resume: {skey} job {existing['job_id']} state={st} → waiting")
                test_ids_to_wait.append(existing["job_id"])
            else:
                existing["state"] = st
                log.info(f"  resume: {skey} job {existing['job_id']} already {st}")
        elif ser_cell_complete("adresso", ser_label):
            jid = submit_ser_test_inference("adresso", arch)
            state["test_jobs"][skey] = {"job_id": jid, "state": "SUBMITTED" if jid else "SUBMIT_FAILED"}
            if jid:
                test_ids_to_wait.append(jid)
            save_state(state)
        else:
            log.warning(f"  skipping SER test {akey}: training not complete on disk")
            state["test_jobs"][skey] = {"job_id": None, "state": "SKIPPED_NO_TRAIN"}
            save_state(state)
        # CogniAligned ADReSSo
        ckey = f"cog_adresso_{akey}"
        existing = state["test_jobs"].get(ckey, {})
        if existing.get("job_id"):
            st = job_state(existing["job_id"])
            if not any(st.startswith(t) for t in TERMINAL_STATES):
                log.info(f"  resume: {ckey} job {existing['job_id']} state={st} → waiting")
                test_ids_to_wait.append(existing["job_id"])
            else:
                existing["state"] = st
                log.info(f"  resume: {ckey} job {existing['job_id']} already {st}")
        elif cog_cell_complete("adresso", cog_fusion):
            jid = submit_cog_test_inference(arch)
            state["test_jobs"][ckey] = {"job_id": jid, "state": "SUBMITTED" if jid else "SUBMIT_FAILED"}
            if jid:
                test_ids_to_wait.append(jid)
            save_state(state)
        else:
            log.warning(f"  skipping CogniAligned test {akey}: training not complete on disk")
            state["test_jobs"][ckey] = {"job_id": None, "state": "SKIPPED_NO_TRAIN"}
            save_state(state)

    # ─── Phase 5: wait for test inference ──────────────────────────────────
    if test_ids_to_wait:
        log.info(f"=== Phase 5: waiting for {len(test_ids_to_wait)} test-inference jobs ===")
        finals = wait_for_jobs(test_ids_to_wait, "test")
        for key, meta in state["test_jobs"].items():
            jid = meta.get("job_id")
            if jid in finals:
                meta["state"] = finals[jid]
                in_dir = AD_SLURM if key.startswith("ser_") else COG_SLURM
                parsed = parse_test_result_from_slurm("TEST_RESULT", jid, in_dir)
                meta["test_acc"] = parsed["acc"]
                meta["test_ms_per_sample"] = parsed["ms_per_sample"]
        save_state(state)
    else:
        log.info("=== Phase 5: no new test-inference jobs to wait on ===")

    # ─── Phase 6: final CSV ────────────────────────────────────────────────
    log.info("=== Phase 6: building final CSV ===")
    # Even for cells without a fresh test job, scan the slurm-log for an
    # existing TEST_RESULT line so that resumed runs pick up prior tests.
    for key, meta in state["test_jobs"].items():
        jid = meta.get("job_id")
        if jid and (meta.get("test_acc") is None):
            in_dir = AD_SLURM if key.startswith("ser_") else COG_SLURM
            parsed = parse_test_result_from_slurm("TEST_RESULT", jid, in_dir)
            meta["test_acc"] = parsed["acc"]
            meta["test_ms_per_sample"] = parsed["ms_per_sample"]
    save_state(state)
    build_csv(state)
    state["finished_at"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)
    log.info(f"DONE — Step B finished. CSV: {CSV_OUT}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Interrupted; state saved → re-run to resume.")
        sys.exit(130)
