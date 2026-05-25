#!/usr/bin/env python3
"""
Precompute CogniAligned embeddings for WAB_samples (Amyloid + PPA share this tree).

Uses WAB_samples/text/transcriptions.csv (built from labels + word-level CSVs if missing).

Usage (from CogniAligned/modules/preprocess):
  python run_preprocessing_wab.py --alignment_mode token_token_uniform
  python run_preprocessing_wab.py --alignment_mode token_token_char
"""
import argparse
import os
import sys

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--alignment_mode",
    type=str,
    default="word_token",
    choices=["word_token", "token_token_uniform", "token_token_char", "none"],
)
parser.add_argument(
    "--wav2vec2_layer_mode",
    type=str,
    default=None,
    choices=["last", "weighted"],
    help='Wav2Vec2: "last" (default) or "weighted" (stack layers for learnable mixing).',
)
args = parser.parse_args()

WAB_ROOT = os.environ.get(
    "WAB_ROOT", "/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples"
)
AUDIO_PATH = f"{WAB_ROOT}/"
TEXT_OUTPUT_PATH = f"{WAB_ROOT}/text/"
CSV_LABELS_PATH = f"{WAB_ROOT}/labels.csv"
TRANSCRIPTIONS_CSV = f"{TEXT_OUTPUT_PATH.rstrip('/')}/transcriptions.csv"
# preprocessembeddings.py expects this filename; we symlink/copy from transcriptions.csv
TEXT_TRANSCRIPTIONS_CSV = f"{TEXT_OUTPUT_PATH.rstrip('/')}/text_transcriptions.csv"

print(f"WAB root: {WAB_ROOT}")
print(f"Alignment mode: {args.alignment_mode}")

os.makedirs(TEXT_OUTPUT_PATH, exist_ok=True)


def ensure_text_transcriptions_csv() -> str:
    """Build text_transcriptions.csv (uid, diagno, transcription, transcription_pause) for WAB."""
    if os.path.exists(TEXT_TRANSCRIPTIONS_CSV):
        return TEXT_TRANSCRIPTIONS_CSV

    if os.path.exists(TRANSCRIPTIONS_CSV):
        df = pd.read_csv(TRANSCRIPTIONS_CSV, encoding="utf-8")
        if "diagno" not in df.columns:
            df["diagno"] = ""
        out = df[["uid", "diagno", "transcription", "transcription_pause"]]
        out.to_csv(TEXT_TRANSCRIPTIONS_CSV, index=False, encoding="utf-8")
        print(f"Created {TEXT_TRANSCRIPTIONS_CSV} from transcriptions.csv ({len(out)} rows)")
        return TEXT_TRANSCRIPTIONS_CSV

    # Fallback: assemble from labels.csv + per-file word-level CSV
    labels = pd.read_csv(CSV_LABELS_PATH)
    rows = []
    for _, row in labels.iterrows():
        fname = str(row.iloc[0])
        uid = os.path.splitext(fname)[0]
        if uid.endswith(".alac"):
            uid = uid[:-5]
        wl = os.path.join(TEXT_OUTPUT_PATH, uid + ".csv")
        if not os.path.exists(wl):
            print(f"WARNING: missing word-level CSV for {uid}")
            continue
        wdf = pd.read_csv(wl)
        words = " ".join(str(w) for w in wdf["word"].tolist())
        pause = " ".join(f"{w} ." for w in wdf["word"].tolist())
        rows.append({
            "uid": uid,
            "diagno": "",
            "transcription": words,
            "transcription_pause": pause,
        })
    if not rows:
        print("ERROR: Could not build text_transcriptions.csv — no source data found.")
        sys.exit(1)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(TEXT_TRANSCRIPTIONS_CSV, index=False, encoding="utf-8")
    out_df.to_csv(TRANSCRIPTIONS_CSV, index=False, encoding="utf-8")
    print(f"Built {TEXT_TRANSCRIPTIONS_CSV} from word-level CSVs ({len(out_df)} rows)")
    return TEXT_TRANSCRIPTIONS_CSV


ensure_text_transcriptions_csv()

print("Step 2/2: Generating embeddings")
print("-" * 60)

import preprocessembeddings
import yaml
from dotmap import DotMap

with open(os.path.join(os.path.dirname(__file__), "..", "configs", "default.yaml"), "r") as f:
    config_yaml = yaml.safe_load(f)
config = DotMap(config_yaml)

preprocessembeddings.root_path = AUDIO_PATH
preprocessembeddings.root_text_path = TEXT_OUTPUT_PATH
preprocessembeddings.textual_data = TEXT_TRANSCRIPTIONS_CSV
preprocessembeddings.textual_model = config.model.textual_model
preprocessembeddings.audio_model = config.model.audio_model
preprocessembeddings.pauses = config.model.pauses
preprocessembeddings.alignment_mode = args.alignment_mode
preprocessembeddings.wav2vec2_layer_mode = (
    args.wav2vec2_layer_mode
    if args.wav2vec2_layer_mode is not None
    else getattr(config.model, "wav2vec2_layer_mode", "last")
)
_n_layers = getattr(config.model, "audio_num_hidden_layers", 12)
preprocessembeddings.wav2vec2_num_hidden_layers = (
    int(_n_layers) if isinstance(_n_layers, (int, float, str)) else 12
)
_suffix = getattr(config.data, "audio_layer_suffix", "")
preprocessembeddings.wav2vec2_layer_suffix_override = _suffix if isinstance(_suffix, str) else ""
print(f"Wav2Vec2 layer mode: {preprocessembeddings.wav2vec2_layer_mode}")

_labels = pd.read_csv(CSV_LABELS_PATH)
_fname_by_uid = {}
for _, _row in _labels.iterrows():
    _fn = str(_row.iloc[0])
    _stem = os.path.splitext(_fn)[0]
    _fname_by_uid[_stem] = _fn


def _wab_audio_path(root_path, uid, diagno_str):
    """Resolve WAB audio (.wav or .alac) from labels.csv first column."""
    audio_dir = os.path.join(root_path, diagno_str) if diagno_str else root_path
    fn = _fname_by_uid.get(uid)
    if fn:
        p = os.path.join(audio_dir, fn)
        if os.path.exists(p):
            return p
    for ext in (".wav", ".alac"):
        p = os.path.join(audio_dir, uid + ext)
        if os.path.exists(p):
            return p
    return None


preprocessembeddings.audio_path_resolver = _wab_audio_path
preprocessembeddings.preprocess_text()
print("WAB preprocessing complete.")
