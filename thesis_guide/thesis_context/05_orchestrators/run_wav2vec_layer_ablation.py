#!/usr/bin/env python3
"""Wav2Vec layer ablation: CogniAligned + SER × weighted/final × crossgated/bicross/mamba × 3 datasets."""
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
AD = BASE / "ad-detection"
COG = BASE / "CogniAligned"
COG_SLURM = COG / "logs" / "slurm"
AD_SLURM = AD / "wandb"
PRE = COG / "modules" / "preprocess"
CFG_OUT = COG / "modules" / "configs" / "ablation" / "wav2vec_layer"
STATE_FILE = BASE / "wav2vec_layer_state.json"
ORCH_LOG = BASE / "wav2vec_layer_ablation.log"
CSV_OUT = BASE / "wav2vec_layer_ablation.csv"
STEP_B_CSV = BASE / "step_b_results.csv"

POLL = 90
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "CANCELLED+", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}

FUSIONS = [
    ("crossgated", "GatedCrossAttention", "gatedcrossattn"),
    ("bicross", "BidirectionalCrossAttention", "bicrossattn"),
    ("mamba", "MambaSeqToSeq", "mamba"),
]
TECHNIQUES = [
    ("weighted", "weighted", "weighted"),  # label, cog_mode, ser_mode
    ("final", "last", "last"),
]
DATASETS = [
    ("adresso", "adresso_classify/run_train_generic.sh", "modules/main.py", "modules/configs"),
    ("amyloid", "amyloid/run_train_generic.sh", "modules/amyloid/main.py", "modules/configs/amyloid"),
    ("ppa", "ad_classification/run_train_generic.sh", "modules/ppa/main.py", "modules/configs/ppa"),
]
COG_MAMBA = {
    "adresso": "modules/configs/default_mamba.yaml",
    "amyloid": "modules/configs/amyloid/default_mamba.yaml",
    "ppa": "modules/configs/ppa/mamba.yaml",
}
COG_BASE_CFG = {
    "crossgated": "thesis_crossgated.yaml",
    "bicross": "thesis_bicross.yaml",
    "mamba": None,
}
COG_LOG_FUSION = {"crossgated": "crossgated", "bicross": "bicross", "mamba": "mamba", "transformerpe": "concat"}
SER_CACHE_WEIGHTED = AD / "cache" / "ser_embeddings"
SER_CACHE_LAST = AD / "cache" / "ser_embeddings_last"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(ORCH_LOG), logging.StreamHandler()],
)
log = logging.getLogger(__name__)


def _load_step_b():
    spec = importlib.util.spec_from_file_location("rsb", BASE / "run_step_b.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


RSB = _load_step_b()


def job_state(jid: str) -> str:
    return RSB.job_state(jid)


def is_terminal(st: str) -> bool:
    return any(st.startswith(t) for t in TERMINAL)


def cog_tag(technique: str) -> str:
    return "_wlayer" if technique == "weighted" else ""


def ser_exp_label(ser_label: str, technique: str) -> str:
    return ser_label if technique == "weighted" else f"{ser_label}_wvlast"


def cog_log_dir(dataset: str, fusion: str, technique: str) -> Path:
    fusion_dir = COG_LOG_FUSION.get(fusion, fusion)
    tag = cog_tag(technique)
    return COG / "logs" / f"{dataset}_distil_wav2vec2_P_{fusion_dir}{tag}_mean"


def cog_cell_complete(dataset: str, fusion: str, technique: str) -> bool:
    s = cog_log_dir(dataset, fusion, technique) / "cross_fold_summary.txt"
    if not s.exists():
        return False
    with open(s) as f:
        return sum(1 for line in f if line.startswith("Fold ")) >= 5


def ser_cell_complete(dataset: str, ser_label: str, technique: str) -> bool:
    label = ser_exp_label(ser_label, technique)
    if dataset == "ppa":
        ds_label = "ppa"
    else:
        ds_label = dataset
    d = AD / "models" / f"thesis_{ds_label}_{label}"
    if not d.exists():
        return False
    return all((d / f"best_{ds_label}_model_fold{f}.pt").exists() for f in range(5))


def needs_train(model: str, technique: str) -> bool:
    if model == "CogniAligned" and technique == "final":
        return False
    if model == "SER" and technique == "weighted":
        return False
    return True


def wlayer_embed_count(root: Path, min_n: int = 80) -> bool:
    if not root.exists():
        return False
    n = len(list(root.rglob("*distil_pauses_audio_wlayer.pt")))
    return n >= min_n


def ser_cache_ready(technique: str, min_manifest: int = 3000) -> bool:
    cache = SER_CACHE_LAST if technique == "final" else SER_CACHE_WEIGHTED
    manifest = cache / "manifest.csv"
    if not manifest.exists():
        return False
    with open(manifest) as f:
        return sum(1 for _ in f) - 1 >= min_manifest


def submit_ser_preprocess_last() -> Optional[str]:
    """One job: last-layer Wav2Vec features for all SER tasks/folds."""
    name = "pre_ser_wvlast_all"
    wrap = (
        f"cd '{AD}' && . '{AD}/.venv/bin/activate' && "
        f"export PYTHONUNBUFFERED=1 && "
        f"python -u scripts/precompute_ser_embeddings.py --task all "
        f"--wav2vec_layer_mode last "
        f"--cache_dir '{SER_CACHE_LAST}'"
    )
    cmd = [
        "sbatch", f"--job-name={name}", f"--output={AD_SLURM}/slurm_%x_%j.txt",
        "-A", "veu", "-p", "veu", "--cpus-per-task=8", "--mem=32GB", "--gres=gpu:1",
        "--ntasks=1", "--exclude=veuc05,veuc01,veuc11,veuc10", "--time=24:00:00", "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"SER precompute sbatch failed: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    return m.group(1) if m else None


def submit_preprocess() -> List[str]:
    jids = []
    for test in (False, True):
        flag = " --test" if test else ""
        name = f"pre_wvl_adresso_{'test' if test else 'train'}"
        wrap = (
            f"cd '{PRE}' && export HF_HOME='{COG}/.cache/huggingface' PYTHONUNBUFFERED=1 && "
            f"'{COG}/.venv/bin/python' -u run_preprocessing.py "
            f"--wav2vec2_layer_mode weighted{flag}"
        )
        jid = _sbatch(name, wrap, hours="12:00:00")
        if jid:
            jids.append(jid)
    name = "pre_wvl_wab"
    wrap = (
        f"cd '{PRE}' && export HF_HOME='{COG}/.cache/huggingface' PYTHONUNBUFFERED=1 && "
        f"'{COG}/.venv/bin/python' -u run_preprocessing_wab.py --wav2vec2_layer_mode weighted"
    )
    jid = _sbatch(name, wrap, hours="12:00:00")
    if jid:
        jids.append(jid)
    return jids


def _sbatch(name: str, wrap: str, mem: str = "64GB", hours: str = "08:00:00") -> Optional[str]:
    COG_SLURM.mkdir(parents=True, exist_ok=True)
    cmd = [
        "sbatch", f"--job-name={name[:48]}", f"--output={COG_SLURM}/%x_%j.txt",
        "-A", "veu", "-p", "veu", "--cpus-per-task=16", f"--mem={mem}", "--gres=gpu:1",
        "--ntasks=1", "--exclude=veuc05,veuc01,veuc11,veuc10", f"--time={hours}", "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"sbatch {name}: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    return m.group(1) if m else None


def ensure_cog_config(dataset: str, fusion: str, technique: str) -> Path:
    CFG_OUT.mkdir(parents=True, exist_ok=True)
    stem = f"wv_{fusion}_{technique}_{dataset}"
    out = CFG_OUT / f"{stem}.yaml"
    if out.exists():
        return out
    if fusion == "mamba":
        src = COG / COG_MAMBA[dataset]
    elif dataset == "adresso":
        src = COG / "modules" / "configs" / COG_BASE_CFG[fusion]
    else:
        src = COG / "modules" / "configs" / dataset / COG_BASE_CFG[fusion]
    with open(src) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("model", {})
    cfg.setdefault("data", {})
    cfg["model"]["wav2vec2_layer_mode"] = "weighted" if technique == "weighted" else "last"
    cfg["model"]["use_weighted_audio_layers"] = technique == "weighted"
    cfg["model"]["audio_num_hidden_layers"] = 12
    if technique == "weighted":
        cfg["model"]["experiment_tag"] = "_wlayer"
        cfg["data"]["audio_layer_suffix"] = "_wlayer"
    else:
        cfg["model"].pop("experiment_tag", None)
        cfg["data"]["audio_layer_suffix"] = ""
    if dataset != "adresso":
        cfg["dataset_prefix"] = dataset
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return out


def submit_cog_train(dataset: str, fusion: str, technique: str) -> Optional[str]:
    cfg_path = ensure_cog_config(dataset, fusion, technique)
    main_py = next(m for d, _, m, _ in DATASETS if d == dataset)
    cfg_rel = cfg_path.relative_to(COG)
    name = f"wv_cog_{fusion}_{technique}_{dataset}"[:48]
    wrap = (
        f"cd '{COG}' && export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false "
        f"HF_HOME='{COG}/.cache/huggingface' WANDB_DIR='{COG}/.cache/wandb' "
        f"PYTHONPATH='{COG}/modules' && "
        f"'{COG}/.venv/bin/python' -u {main_py} --config {cfg_rel}"
    )
    return _sbatch(name, wrap)


def submit_ser_train(dataset: str, ser_method: str, ser_label: str, technique: str, fold: int, is_mamba: bool) -> Optional[str]:
    script = next(s for d, s, _, _ in DATASETS if d == dataset)
    exp_tag = f"thesis_{dataset}_{ser_exp_label(ser_label, technique)}" if dataset != "ppa" else f"thesis_ppa_{ser_exp_label(ser_label, technique)}"
    job_name = f"wv_ser_{dataset}_{ser_label}_{technique}_f{fold}"[:48]
    if technique == "final":
        precomp_dir = str(SER_CACHE_LAST)
        wv_mode = "last"
        use_pre = "1"
    else:
        precomp_dir = str(SER_CACHE_WEIGHTED)
        wv_mode = "weighted"
        use_pre = "1"
    export_kv = {
        "SEQ_METHOD": ser_method,
        "HEADS": "4",
        "LR": "0.00005",
        "MAX_EPOCHS": "25",
        "LR_PATIENCE": "10",
        "EXP_TAG": exp_tag,
        "WAV2VEC_LAYER_MODE": wv_mode,
        "USE_PRECOMPUTED_EMBEDDINGS": use_pre,
        "PRECOMPUTED_EMBEDDINGS_DIR": precomp_dir,
    }
    if is_mamba:
        export_kv.update({"MAMBA_N_BLOCKS": "4", "MAMBA_D_STATE": "16", "MAMBA_D_CONV": "4", "MAMBA_EXPAND": "2"})
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_kv.items())
    cmd = ["sbatch", f"--job-name={job_name}", f"--export={export_str}", script, str(fold)]
    p = subprocess.run(cmd, cwd=str(AD), capture_output=True, text=True)
    if p.returncode != 0:
        log.error(f"SER sbatch {job_name}: {p.stderr.strip()}")
        return None
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip())
    return m.group(1) if m else None


def submit_cog_test(dataset: str, fusion: str, technique: str) -> Optional[str]:
    if dataset != "adresso":
        return None
    cfg_path = ensure_cog_config("adresso", fusion, technique)
    cfg_rel = cfg_path.relative_to(COG)
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["dataset_prefix"] = "adresso"
    test_dir = COG / "modules" / "configs" / "ablation" / "wav2vec_layer_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    test_cfg = test_dir / cfg_path.name
    with open(test_cfg, "w") as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    rel = test_cfg.relative_to(COG)
    name = f"wv_test_cog_{fusion}_{technique}"[:48]
    wrap = (
        f"cd '{COG}' && export WANDB_MODE=disabled PYTHONPATH='{COG}/modules' "
        f"HF_HOME='{COG}/.cache/huggingface' && "
        f"'{COG}/.venv/bin/python' -u modules/test.py --config {rel}"
    )
    return _sbatch(name, wrap, mem="32GB", hours="02:00:00")


def submit_ser_test(fusion: str, ser_method: str, ser_label: str, technique: str, is_mamba: bool) -> Optional[str]:
    label = ser_exp_label(ser_label, technique)
    fold_ckpts = [AD / "models" / f"thesis_adresso_{label}" / f"best_adresso_model_fold{f}.pt" for f in range(5)]
    if not all(p.exists() for p in fold_ckpts):
        return None
    ckpts_arg = " ".join(f"'{p}'" for p in fold_ckpts)
    mamba_args = " --mamba_n_blocks 4 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2" if is_mamba else ""
    wv = "weighted" if technique == "weighted" else "last"
    name = f"wv_test_ser_{ser_label}_{technique}"[:48]
    wrap = (
        f"cd '{AD}' && . '{AD}/.venv/bin/activate' && export PYTHONUNBUFFERED=1 && "
        f"python -u adresso_classify/test_adresso.py --model_paths {ckpts_arg} "
        f"--exp_tag thesis_adresso_{label} --labels_csv '{AD}/task1.csv' "
        "--audio_dir '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/test-dist/audio' "
        "--transcription_dir '/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/test-dist' "
        "--batch_size 8 --sample_rate 16000 --padding_type repetition_pad --window_secs 40 --stride_secs 40 "
        "--speech_feature_extractor WAV2VEC2_XLSR_300M --speech_feature_extractor_output_vectors_dimension 1024 "
        "--text_feature_extractor MODERN_BERT_BASE --text_feature_extractor_output_vectors_dimension 768 "
        f"--wav2vec_layer_mode {wv} "
        "--speech_adapter LinearAdapter --speech_adapter_output_vectors_dimension 256 "
        "--text_adapter LinearAdapter --text_adapter_output_vectors_dimension 256 "
        f"--seq_to_seq_method {ser_method} --seq_to_seq_heads_number 4 "
        "--seq_to_seq_input_dropout 0.3 --skip_connections "
        "--seq_to_one_method AttentionPooling --seq_to_one_input_dropout 0.3 "
        "--classifier_hidden_layers 1 --classifier_hidden_layers_width 256 --classifier_layer_drop_out 0.3 "
        "--acoustic_features_path '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/acoustic_features.csv'"
        f"{mamba_args}"
    )
    cmd = [
        "sbatch", f"--job-name={name}", f"--output={AD_SLURM}/slurm_%x_%j.txt",
        "-A", "veu", "-p", "veu", "--cpus-per-task=8", "--mem=32GB", "--gres=gpu:1",
        "--ntasks=1", "--exclude=veuc05,veuc01", "--time=02:00:00", "--wrap", wrap,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"Submitted batch job (\d+)", p.stdout.strip()) if p.returncode == 0 else None
    return m.group(1) if m else None


def load_step_b_index() -> Dict[Tuple[str, str, str], dict]:
    idx = {}
    if not STEP_B_CSV.exists():
        return idx
    with open(STEP_B_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("arch") not in ("crossgated", "bicross", "mamba"):
                continue
            idx[(row["model"], row["dataset"], row["arch"])] = row
    return idx


def parse_cog_metrics(dataset: str, fusion: str, technique: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    p = cog_log_dir(dataset, fusion, technique) / "cross_fold_summary.txt"
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
    ms = RSB.parse_cog_val_inference_ms(dataset, fusion)
    if technique == "weighted":
        d = cog_log_dir(dataset, fusion, technique)
        times = []
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
        ms = sum(times) / len(times) if times else ms
    return sum(accs) / len(accs), max(accs), ms


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(s: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2, default=str)


def all_done(state: dict) -> bool:
    sb = load_step_b_index()
    for fusion, _, ser_label in FUSIONS:
        for technique, _, _ in TECHNIQUES:
            for dataset, _, _, _ in DATASETS:
                for model in ("CogniAligned", "SER"):
                    if model == "CogniAligned":
                        if not cog_cell_complete(dataset, fusion, technique):
                            return False
                    elif not ser_cell_complete(dataset, ser_label, technique):
                        return False
            for model in ("CogniAligned", "SER"):
                tkey = f"test_{model}_{fusion}_{technique}"
                tj = state.get("test_jobs", {}).get(tkey, {})
                if tj.get("test_acc") is not None:
                    continue
                if not needs_train(model, technique) and sb.get((model, "adresso", fusion), {}).get("test_acc"):
                    continue
                if tj.get("job_id"):
                    res = RSB.parse_test_result_from_slurm("", tj["job_id"], COG_SLURM if model == "CogniAligned" else AD_SLURM)
                    if res.get("acc") is not None:
                        continue
                return False
    return True


def submit_pending(state: dict) -> List[str]:
    active: List[str] = []
    state.setdefault("preprocess_jobs", {})
    state.setdefault("train_jobs", {})
    state.setdefault("test_jobs", {})

    adresso_text = Path("/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/text")
    wab_text = Path("/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/text")
    need_pre = not wlayer_embed_count(adresso_text) or not wlayer_embed_count(wab_text, 50)
    if need_pre:
        pre_jids = [k for k in state["preprocess_jobs"] if k.isdigit()]
        failed_pre = [j for j in pre_jids if is_terminal(job_state(j)) and not job_state(j).startswith("COMPLETED")]
        running_pre = [j for j in pre_jids if not is_terminal(job_state(j))]
        active.extend(running_pre)
        if failed_pre or not pre_jids:
            for j in failed_pre:
                state["preprocess_jobs"].pop(j, None)
            if failed_pre or not pre_jids:
                log.warning(f"resubmitting preprocess ({len(failed_pre)} failed)")
                for jid in submit_preprocess():
                    if jid:
                        state["preprocess_jobs"][jid] = "submitted"
                        active.append(jid)
                save_state(state)
        if need_pre and (running_pre or active):
            return active
        if need_pre and not wlayer_embed_count(adresso_text):
            return active
        state["preprocess_jobs"]["done"] = True
        need_pre = False
        save_state(state)

    if wlayer_embed_count(adresso_text) and wlayer_embed_count(wab_text, 50):
        state["preprocess_jobs"]["done"] = True

    if not ser_cache_ready("final"):
        skey = "ser_last_preprocess"
        meta = state.get("ser_preprocess_jobs", {})
        jid = meta.get("job_id")
        if jid and not is_terminal(job_state(jid)):
            active.append(jid)
        elif not (jid and job_state(jid).startswith("COMPLETED")):
            if jid and is_terminal(job_state(jid)):
                meta.pop("job_id", None)
            new = submit_ser_preprocess_last()
            state.setdefault("ser_preprocess_jobs", {})["job_id"] = new
            if new:
                active.append(new)
            save_state(state)
            return active
        state.setdefault("ser_preprocess_jobs", {})["ready"] = True
    else:
        state.setdefault("ser_preprocess_jobs", {})["ready"] = True

    for fusion, ser_method, ser_label in FUSIONS:
        is_mamba = fusion == "mamba"
        for technique, _, _ in TECHNIQUES:
            for dataset, _, _, _ in DATASETS:
                if needs_train("CogniAligned", technique):
                    ckey = f"cog_{dataset}_{fusion}_{technique}"
                    if cog_cell_complete(dataset, fusion, technique):
                        continue
                    meta = state["train_jobs"].get(ckey, {})
                    jid = meta.get("job_id")
                    if jid and not is_terminal(job_state(jid)):
                        active.append(jid)
                        continue
                    if jid and job_state(jid).startswith("COMPLETED") and cog_cell_complete(dataset, fusion, technique):
                        continue
                    new = submit_cog_train(dataset, fusion, technique)
                    state["train_jobs"][ckey] = {"job_id": new}
                    if new:
                        active.append(new)
                if needs_train("SER", technique):
                    for fold in range(5):
                        skey = f"ser_{dataset}_{fusion}_{technique}_f{fold}"
                        if ser_cell_complete(dataset, ser_label, technique):
                            continue
                        meta = state["train_jobs"].get(skey, {})
                        jid = meta.get("job_id")
                        if jid and not is_terminal(job_state(jid)):
                            active.append(jid)
                            continue
                        if jid and job_state(jid).startswith("COMPLETED") and ser_cell_complete(dataset, ser_label, technique):
                            continue
                        new = submit_ser_train(dataset, ser_method, ser_label, technique, fold, is_mamba)
                        state["train_jobs"][skey] = {"job_id": new}
                        if new:
                            active.append(new)

    sb = load_step_b_index()
    for fusion, ser_method, ser_label in FUSIONS:
        is_mamba = fusion == "mamba"
        for technique, _, _ in TECHNIQUES:
            for model in ("CogniAligned", "SER"):
                tkey = f"test_{model}_{fusion}_{technique}"
                if not needs_train(model, technique):
                    row = sb.get((model, "adresso", fusion))
                    if row and row.get("test_acc"):
                        try:
                            state["test_jobs"][tkey] = {
                                "test_acc": float(row["test_acc"]),
                                "test_ms": float(row["test_inference_ms_per_sample"]) if row.get("test_inference_ms_per_sample") else None,
                                "note": "reused_stepB",
                            }
                        except ValueError:
                            pass
                        continue
                if model == "CogniAligned":
                    if not cog_cell_complete("adresso", fusion, technique):
                        continue
                elif not ser_cell_complete("adresso", ser_label, technique):
                    continue
                tj = state["test_jobs"].get(tkey, {})
                if tj.get("test_acc") is not None:
                    continue
                jid = tj.get("job_id")
                if jid and not is_terminal(job_state(jid)):
                    active.append(jid)
                    continue
                if jid:
                    res = RSB.parse_test_result_from_slurm("", jid, COG_SLURM if model == "CogniAligned" else AD_SLURM)
                    if res.get("acc") is not None:
                        tj["test_acc"] = res["acc"]
                        tj["test_ms"] = res.get("ms_per_sample")
                        continue
                new = submit_cog_test("adresso", fusion, technique) if model == "CogniAligned" else submit_ser_test(fusion, ser_method, ser_label, technique, is_mamba)
                state["test_jobs"][tkey] = {"job_id": new}
                if new:
                    active.append(new)

    save_state(state)
    return active


def build_csv(state: dict) -> None:
    sb = load_step_b_index()
    cols = [
        "phase", "model", "dataset", "wav2vec_layers", "fusion", "lr",
        "val_acc_mean", "best_fold_acc", "val_inference_ms_per_sample",
        "test_acc", "test_inference_ms_per_sample", "note",
    ]
    rows = []
    for fusion, ser_method, ser_label in FUSIONS:
        for technique, _, _ in TECHNIQUES:
            for dataset, _, _, _ in DATASETS:
                for model in ("CogniAligned", "SER"):
                    note = ""
                    lr = "2e-5" if model == "CogniAligned" else "5e-5"
                    v = b = ms = test_acc = test_ms = None
                    if not needs_train(model, technique):
                        sb_row = sb.get((model, dataset, fusion))
                        if sb_row:
                            def fn(k):
                                try:
                                    return float(sb_row[k]) if sb_row.get(k) else None
                                except ValueError:
                                    return None
                            v, b, ms = fn("val_acc_mean"), fn("best_fold_acc"), fn("val_inference_ms_per_sample")
                            if dataset == "adresso":
                                test_acc = fn("test_acc")
                                test_ms = fn("test_inference_ms_per_sample")
                            note = "reused_stepB"
                    if model == "CogniAligned":
                        if note != "reused_stepB":
                            v, b, ms = parse_cog_metrics(dataset, fusion, technique)
                    else:
                        if note != "reused_stepB":
                            v, b, _, _, ms = RSB.parse_ser_per_fold(dataset, ser_exp_label(ser_label, technique))
                    tkey = f"test_{model}_{fusion}_{technique}"
                    tj = state.get("test_jobs", {}).get(tkey, {})
                    if test_acc is None and tj.get("test_acc") is not None:
                        test_acc, test_ms = tj.get("test_acc"), tj.get("test_ms")
                    if test_acc is None and tj.get("job_id"):
                        res = RSB.parse_test_result_from_slurm("", tj["job_id"], COG_SLURM if model == "CogniAligned" else AD_SLURM)
                        test_acc, test_ms = res.get("acc"), res.get("ms_per_sample")
                    if v is None:
                        note = (note + ";incomplete").strip(";")
                    rows.append({
                        "phase": "WvLayer",
                        "model": model,
                        "dataset": dataset,
                        "wav2vec_layers": technique,
                        "fusion": fusion,
                        "lr": lr,
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
    log.info("Wav2Vec layer ablation monitor started")
    while True:
        state = load_state()
        active = submit_pending(state)
        state = load_state()
        if not active:
            for tkey, tj in list(state.get("test_jobs", {}).items()):
                if tj.get("test_acc") is None and tj.get("job_id"):
                    model = "CogniAligned" if "CogniAligned" in tkey else "SER"
                    res = RSB.parse_test_result_from_slurm("", tj["job_id"], COG_SLURM if model == "CogniAligned" else AD_SLURM)
                    if res.get("acc") is not None:
                        tj["test_acc"], tj["test_ms"] = res["acc"], res.get("ms_per_sample")
            save_state(state)
            if all_done(state):
                build_csv(state)
                state["finished_at"] = datetime.now().isoformat(timespec="seconds")
                save_state(state)
                log.info("Wav2Vec layer ablation complete.")
                return
        else:
            for jid in list(active):
                if is_terminal(job_state(jid)):
                    st = job_state(jid)
                    log.info(f"job {jid} -> {st}")
                    if not st.startswith("COMPLETED"):
                        log.warning(f"retry after failure {jid} ({st})")
        save_state(state)
        time.sleep(POLL)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--monitor", action="store_true")
    ap.add_argument("--submit-only", action="store_true")
    args = ap.parse_args()
    state = load_state()
    active = submit_pending(state)
    log.info(f"tracking {len(active)} jobs")
    if args.submit_only:
        return
    if args.monitor:
        monitor_loop()
    else:
        log.info("Run with --monitor to wait for completion")


if __name__ == "__main__":
    main()
