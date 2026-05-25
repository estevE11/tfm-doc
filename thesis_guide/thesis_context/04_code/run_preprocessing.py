#!/usr/bin/env python3
"""
Wrapper script to run preprocessing with configurable paths.
This avoids modifying the original preprocessing scripts.
"""

import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Run preprocessing on the test set instead of train set')
parser.add_argument('--alignment_mode', type=str, default='word_token',
                    choices=['word_token', 'token_token_uniform', 'token_token_char', 'none'],
                    help=('How to assign audio features to text tokens. '
                          '"word_token" (default) broadcasts one mean per word to every '
                          'subword. "token_token_uniform" / "token_token_char" split each '
                          'word\'s audio span across its subwords. "none" stores a '
                          'temporally-pooled raw audio sequence (no token alignment).'))
parser.add_argument('--wav2vec2_layer_mode', type=str, default=None,
                    choices=['last', 'weighted'],
                    help=('Wav2Vec2 feature mode. '
                          '"last" saves last hidden state; '
                          '"weighted" saves per-layer states to be learned/combined by the model.'))
args = parser.parse_args()

# Set paths before importing preprocessing modules
ADRESSO_ROOT = os.environ.get(
    'ADRESSO_ROOT', 
    '/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21'
)

if args.test:
    AUDIO_PATH = f"{ADRESSO_ROOT}/diagnosis/test-dist/audio/"
    TEXT_OUTPUT_PATH = f"{ADRESSO_ROOT}/diagnosis/test-dist/text/"
    # test_results_task1.csv only has ID, no dx. We'll use it to get the filenames
    CSV_LABELS_PATH = f"{ADRESSO_ROOT}/diagnosis/test-dist/test_results_task1.csv"
    SPLITS_PATH = f"{ADRESSO_ROOT}/diagnosis/test-dist/splits/"
else:
    AUDIO_PATH = f"{ADRESSO_ROOT}/diagnosis/train/audio/"
    TEXT_OUTPUT_PATH = f"{ADRESSO_ROOT}/diagnosis/train/text/"
    CSV_LABELS_PATH = f"{ADRESSO_ROOT}/diagnosis/train/adresso-train-mmse-scores.csv"
    SPLITS_PATH = f"{ADRESSO_ROOT}/diagnosis/train/splits/"

print(f"Using ADReSSo Root: {ADRESSO_ROOT}")
print(f"Audio Path: {AUDIO_PATH}")
print(f"Text Output Path: {TEXT_OUTPUT_PATH}")
print(f"CSV Labels Path: {CSV_LABELS_PATH}")
print(f"Splits Path: {SPLITS_PATH}")

# Create output directories
os.makedirs(f"{TEXT_OUTPUT_PATH}/ad", exist_ok=True)
os.makedirs(f"{TEXT_OUTPUT_PATH}/cn", exist_ok=True)
os.makedirs(SPLITS_PATH, exist_ok=True)

# Verify paths exist
if not os.path.exists(AUDIO_PATH):
    print(f"ERROR: Audio path does not exist: {AUDIO_PATH}")
    sys.exit(1)

if not os.path.exists(CSV_LABELS_PATH):
    print(f"ERROR: CSV labels file does not exist: {CSV_LABELS_PATH}")
    sys.exit(1)

print("\n" + "="*60)
print("Starting Preprocessing")
print("="*60 + "\n")

# Check if Step 1 (Whisper transcription) is already complete
transcription_csv = f"{TEXT_OUTPUT_PATH.rstrip('/')}/text_transcriptions.csv"
step1_complete = os.path.exists(transcription_csv)

if step1_complete:
    print("Step 1/3: Whisper transcription - ALREADY COMPLETED")
    print("-" * 60)
    print(f"Found existing transcription file: {transcription_csv}")
    # Count lines to verify
    with open(transcription_csv, 'r') as f:
        num_transcriptions = sum(1 for line in f) - 1  # Subtract header
    print(f"Total transcriptions found: {num_transcriptions}")
    print("Skipping to Step 2...\n")
else:
    # Import and patch the preprocessing modules
    print("Step 1/3: Running Whisper transcription...")
    print("-" * 60)
    
    import preprocesswhisper
    # Monkey patch the paths
    preprocesswhisper.root_path = AUDIO_PATH
    preprocesswhisper.textual_data = transcription_csv
    preprocesswhisper.test_mode = args.test
    
    # Run whisper preprocessing
    preprocesswhisper.preprocess_whisper()

print("\n" + "="*60)
print("Step 2/3: Generating embeddings...")
print("-" * 60)

import preprocessembeddings
# Monkey patch the paths
preprocessembeddings.root_path = AUDIO_PATH
preprocessembeddings.root_text_path = TEXT_OUTPUT_PATH
preprocessembeddings.textual_data = f"{TEXT_OUTPUT_PATH.rstrip('/')}/text_transcriptions.csv"

# Set default models if not configured (matching default.yaml)
from dotmap import DotMap
import yaml

with open(os.path.join(os.path.dirname(__file__), '..', 'configs', 'default.yaml'), 'r') as f:
    config_yaml = yaml.safe_load(f)
config = DotMap(config_yaml)

preprocessembeddings.textual_model = config.model.textual_model  # distilbert
preprocessembeddings.audio_model = config.model.audio_model   # wav2vec2
preprocessembeddings.pauses = config.model.pauses
preprocessembeddings.alignment_mode = args.alignment_mode
preprocessembeddings.wav2vec2_layer_mode = (
    args.wav2vec2_layer_mode
    if args.wav2vec2_layer_mode is not None
    else getattr(config.model, 'wav2vec2_layer_mode', 'last')
)
_n_layers = getattr(config.model, 'audio_num_hidden_layers', 12)
preprocessembeddings.wav2vec2_num_hidden_layers = (
    int(_n_layers) if isinstance(_n_layers, (int, float, str)) else 12
)
_suffix = getattr(config.data, 'audio_layer_suffix', '')
preprocessembeddings.wav2vec2_layer_suffix_override = _suffix if isinstance(_suffix, str) else ''
print(f"Alignment mode: {args.alignment_mode}")
print(f"Wav2Vec2 layer mode: {preprocessembeddings.wav2vec2_layer_mode}")

# Run embeddings preprocessing
preprocessembeddings.preprocess_text()

if not args.test:
    print("\n" + "="*60)
    print("Step 3/3: Creating cross-validation splits...")
    print("-" * 60)

    # Create splits using the dataset module
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from dataset import set_splits
    from dotmap import DotMap

    # Create minimal config for split creation
    config = DotMap({
        'data': {
            'csv_labels_path': CSV_LABELS_PATH,
            'splits_path': SPLITS_PATH
        }
    })

    set_splits(config)
    print(f"Splits saved to: {SPLITS_PATH}")

print("\n" + "="*60)
print("Preprocessing Complete!")
print("="*60)
print(f"\nOutput saved to: {TEXT_OUTPUT_PATH}")
