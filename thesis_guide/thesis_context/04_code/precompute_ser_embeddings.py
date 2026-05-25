import argparse
import os
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.join(REPO_ROOT, "adresso_classify"))
sys.path.append(os.path.join(REPO_ROOT, "amyloid"))
sys.path.append(os.path.join(REPO_ROOT, "ad_classification"))

from cached_embeddings import CachedEmbeddingDataset  # noqa: F401 - validates import path for trainers
from speech_feature_extractor import SpeechFeatureExtractor
from text_feature_extractor import TextFeatureExtractor
from utils import pad_collate
from data_adresso import AdressoDataset
from data_amyloid import AmyloidDataset
from data_ad import ADDataset


def parse_args():
    parser = argparse.ArgumentParser("Precompute SER speech/text embeddings.")
    parser.add_argument("--task", choices=["adresso", "amyloid", "ppa", "all"], default="all")
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(REPO_ROOT, "cache", "ser_embeddings"),
        help="Output root; use ser_embeddings_last when --wav2vec_layer_mode last.",
    )
    parser.add_argument("--folds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    parser.add_argument("--splits", nargs="*", default=["train", "val"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)

    # Shared feature settings.
    parser.add_argument("--sample_rate", type=int, default=16000)
    parser.add_argument("--padding_type", type=str, default="repetition_pad")
    parser.add_argument("--window_secs", type=float, default=40)
    parser.add_argument("--stride_secs", type=float, default=40)
    parser.add_argument("--speech_feature_extractor", type=str, default="WAV2VEC2_XLSR_300M")
    parser.add_argument(
        "--wav2vec_layer_mode",
        type=str,
        default="weighted",
        choices=["weighted", "last"],
        help="weighted: learnable layer mix (default, matches legacy cache); last: final transformer layer only.",
    )
    parser.add_argument("--speech_feature_extractor_output_vectors_dimension", type=int, default=1024)
    parser.add_argument("--text_feature_extractor", type=str, default="MODERN_BERT_BASE")
    parser.add_argument("--text_feature_extractor_output_vectors_dimension", type=int, default=768)

    # ADReSSo paths.
    parser.add_argument("--adresso_audio_base_dir", default="/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/audio")
    parser.add_argument("--adresso_transcription_dir", default="/home/usuaris/veussd/roger.esteve.sanchez/adresso/processed_data/diagnosis/train")
    parser.add_argument("--adresso_acoustic_features_path", default="/home/usuaris/veussd/roger.esteve.sanchez/adresso/ADReSSo21/diagnosis/train/acoustic_features.csv")

    # WAB/Amyloid/PPA paths.
    parser.add_argument("--wab_audio_dir", default="/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples")
    parser.add_argument("--wab_acoustic_features_path", default="/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/acoustic_features.csv")
    parser.add_argument("--amyloid_csv_path", default="/home/usuaris/veussd/roger.esteve.sanchez/ad-detection/amyloid/stpau_binary.csv")
    parser.add_argument("--ppa_csv_path", default="/home/usuaris/veussd/roger.esteve.sanchez/WAB_samples/labels.csv")
    parser.add_argument("--ppa_ignore_labels", nargs="*", default=["exclude", "bvFTD", "ADtyp", "control"])
    parser.add_argument("--amyloid_ignore_labels", nargs="*", default=["exclude"])
    return parser.parse_args()


def build_dataset(task, split, fold, args):
    if task == "adresso":
        return AdressoDataset(
            audio_base_dir=args.adresso_audio_base_dir,
            transcription_dir=args.adresso_transcription_dir,
            input_parameters=args,
            window_secs=args.window_secs,
            stride_secs=args.stride_secs,
            split=split,
            fold=fold,
            num_folds=5,
            use_precomputed_transcriptions=True,
            acoustic_features_path=args.adresso_acoustic_features_path,
        )
    if task == "amyloid":
        return AmyloidDataset(
            csv_path=args.amyloid_csv_path,
            audio_dir=args.wab_audio_dir,
            input_parameters=args,
            window_secs=args.window_secs,
            stride_secs=args.stride_secs,
            split=split,
            ignore_labels=args.amyloid_ignore_labels,
            fold=fold,
            num_folds=5,
            acoustic_features_path=args.wab_acoustic_features_path,
        )
    if task == "ppa":
        return ADDataset(
            csv_path=args.ppa_csv_path,
            audio_dir=args.wab_audio_dir,
            input_parameters=args,
            window_secs=args.window_secs,
            stride_secs=args.stride_secs,
            split=split,
            ignore_labels=args.ppa_ignore_labels,
            fold=fold,
            num_folds=5,
            acoustic_features_path=args.wab_acoustic_features_path,
        )
    raise ValueError(f"Unknown task: {task}")


def atomic_write_manifest(rows, cache_dir):
    manifest_path = os.path.join(cache_dir, "manifest.csv")
    new_df = pd.DataFrame(rows)
    if os.path.exists(manifest_path):
        old_df = pd.read_csv(manifest_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["task", "fold", "split", "sample_idx"], keep="last"
        )
    else:
        merged = new_df
    tmp_path = manifest_path + ".tmp"
    merged.to_csv(tmp_path, index=False)
    os.replace(tmp_path, manifest_path)


def precompute_task(task, args, speech_extractor, text_extractor, device):
    rows = []
    for fold in args.folds:
        for split in args.splits:
            dataset = build_dataset(task, split, fold, args)
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=pad_collate,
            )
            sample_offset = 0
            for batch in loader:
                acoustic = None
                if len(batch) == 5:
                    audio, labels, text_tokens, text_mask, acoustic = batch
                    acoustic = acoustic.to(device)
                else:
                    audio, labels, text_tokens, text_mask = batch

                audio = audio.to(device)
                text_tokens = text_tokens.to(device)
                text_mask = text_mask.to(device)

                with torch.no_grad():
                    speech_features = speech_extractor(audio).detach().cpu()
                    text_features = text_extractor(text_tokens, text_mask).detach().cpu()

                text_lengths = text_mask.sum(dim=1).long().cpu().tolist()
                labels = labels.cpu().long()
                acoustic_cpu = acoustic.detach().cpu() if acoustic is not None else None

                for i in range(labels.size(0)):
                    sample_idx = sample_offset + i
                    rel_path = os.path.join(task, f"fold{fold}", split, f"sample_{sample_idx:05d}.pt")
                    out_path = os.path.join(args.cache_dir, rel_path)
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    sample = {
                        "task": task,
                        "fold": fold,
                        "split": split,
                        "sample_idx": sample_idx,
                        "speech": speech_features[i],
                        "text": text_features[i, : text_lengths[i]],
                        "label": int(labels[i].item()),
                    }
                    if acoustic_cpu is not None:
                        sample["acoustic"] = acoustic_cpu[i]
                    torch.save(sample, out_path)
                    rows.append(
                        {
                            "task": task,
                            "fold": fold,
                            "split": split,
                            "sample_idx": sample_idx,
                            "label": int(labels[i].item()),
                            "cache_path": rel_path,
                        }
                    )
                sample_offset += labels.size(0)
            print(f"Precomputed {task} fold={fold} split={split}: {sample_offset} samples")

    atomic_write_manifest(rows, args.cache_dir)


def main():
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    speech_extractor = SpeechFeatureExtractor(args).to(device).eval()
    text_extractor = TextFeatureExtractor(args).to(device).eval()
    for module in (speech_extractor, text_extractor):
        for parameter in module.parameters():
            parameter.requires_grad = False

    tasks = ["adresso", "amyloid", "ppa"] if args.task == "all" else [args.task]
    for task in tasks:
        precompute_task(task, args, speech_extractor, text_extractor, device)


if __name__ == "__main__":
    main()
