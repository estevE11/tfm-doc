import os
import pandas as pd
from transformers import AutoTokenizer, RobertaModel, Wav2Vec2Processor, Wav2Vec2Model, BertTokenizer, BertModel, DistilBertModel, AutoModel
import torch
import torchaudio
import opensmile
import unicodedata
import librosa
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Avaiable: bert, roberta, distilbert, stella, mistral, qwen
textual_model = ''
audio_model = ''
pauses = False

# Audio<->text alignment mode for the precomputed audio tensor:
#   'word_token'           : (default, current behaviour) every subword token of a word
#                            receives the SAME mean audio vector for that word's span.
#   'token_token_uniform'  : split the word's audio span uniformly across its subword
#                            tokens; each token gets the mean of its own slice.
#   'token_token_char'     : split proportionally to each subword's character length.
#   'none'                 : ignore word timestamps; temporal-mean-pool the raw audio
#                            sequence to `max_length` frames (no token alignment).
alignment_mode = 'word_token'

# Suffix appended to the saved audio .pt filename so multiple alignment variants
# can coexist on disk. Empty string keeps backwards-compatible filenames.
ALIGNMENT_SUFFIX = {
    'word_token': '',
    'token_token_uniform': '_tokalign',
    'token_token_char': '_tokalignchar',
    'none': '_noalign',
}

# Optional hook: fn(root_path, uid, diagno_str) -> audio file path (WAB uses .alac/.wav).
audio_path_resolver = None
wav2vec2_layer_mode = 'last'  # 'last' | 'weighted'
wav2vec2_num_hidden_layers = 12
wav2vec2_layer_suffix_override = ''

WAV2VEC2_LAYER_SUFFIX = {
    'last': '',
    'weighted': '_wlayer',
}


def _default_audio_path(root_path, uid, diagno_str):
    audio_dir = os.path.join(root_path, diagno_str) if diagno_str else root_path
    return os.path.join(audio_dir, uid + '.wav')


def _get_wav2vec2_layer_suffix():
    if isinstance(wav2vec2_layer_suffix_override, str) and wav2vec2_layer_suffix_override:
        return wav2vec2_layer_suffix_override
    return WAV2VEC2_LAYER_SUFFIX.get(wav2vec2_layer_mode, '')


def _extract_wav2vec2_features(outputs_audio):
    if wav2vec2_layer_mode == 'weighted':
        hidden_states = outputs_audio.hidden_states
        if hidden_states is None:
            raise RuntimeError("wav2vec2_layer_mode='weighted' requires output_hidden_states=True")
        transformer_states = hidden_states[1:] if len(hidden_states) > 1 else hidden_states
        n_layers = int(wav2vec2_num_hidden_layers) if isinstance(wav2vec2_num_hidden_layers, int) else 12
        n_layers = max(1, min(n_layers, len(transformer_states)))
        selected_states = transformer_states[-n_layers:]
        stacked = torch.stack([h.squeeze(0).cpu() for h in selected_states], dim=0)  # (L, T, D)
        if torch.isnan(stacked).any():
            stacked = torch.nan_to_num(stacked, nan=0.0)
        return stacked

    last_hidden = outputs_audio.last_hidden_state.squeeze(0).cpu()
    if torch.isnan(last_hidden).any():
        last_hidden = torch.nan_to_num(last_hidden, nan=0.0)
    return last_hidden


def _save_wav2vec2_layer_metadata(save_dir, num_layers):
    suffix = _get_wav2vec2_layer_suffix()
    metadata_path = os.path.join(save_dir, f"wav2vec2_layer_metadata{suffix}.pt")
    if os.path.exists(metadata_path):
        return
    init_logits = torch.zeros(num_layers, dtype=torch.float32)
    init_weights = torch.softmax(init_logits, dim=0)
    torch.save({
        'layer_mode': wav2vec2_layer_mode,
        'num_hidden_layers': int(num_layers),
        'initial_layer_logits': init_logits,
        'initial_layer_weights': init_weights,
    }, metadata_path)

root_path = '/dataset/diagnosis/train/audio/'
root_text_path = '/dataset/diagnosis/train/text/'

textual_data = '/dataset/diagnosis/train/text_transcriptions.csv'
max_length = 200

# Global variables for models
tokenizer = None
model = None
processor = None
wav2vec_model = None
smile = None

def init_models():
    global tokenizer, model, processor, wav2vec_model, smile, segment_length
    
    if textual_model == 'bert':
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased").to(device)
    elif textual_model == 'roberta':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = RobertaModel.from_pretrained("roberta-base").to(device)
    elif textual_model == 'distil' or textual_model == 'distilbert':
        tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
        model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
    elif textual_model == 'stella':
        tokenizer = AutoTokenizer.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
        model = AutoModel.from_pretrained("NovaSearch/stella_en_1.5B_v5", trust_remote_code=True)
    elif textual_model == 'mistral':
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModel.from_pretrained("mistralai/Mistral-7B-v0.1", use_auth_token=True)
    elif textual_model == 'qwen':
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")
        model = AutoModel.from_pretrained("Qwen/Qwen2.5-7B")
    else:
        model = None

    if model is not None:
        model.eval()

    if audio_model == 'wav2vec2':
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(device)
        segment_length = 50
    elif audio_model == 'egemaps':
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        segment_length = 10
    else:
        segment_length = 50

def _fill_token_audio(processed, start_seg, end_seg, audio,
                      idx_lo, idx_hi, token_strings=None, mode='word_token'):
    """Assign audio features to processed[idx_lo+1 .. idx_hi] (token slots).

    Slot 0 of `processed` is reserved for the global utterance mean, hence the +1
    offset matches the original script's `processed_audio_tensor[idx + 1]`.

    Args:
        processed:     output tensor of shape (max_length, D_audio) being filled.
        start_seg:     start frame of the word's audio span (inclusive).
        end_seg:       end frame of the word's audio span (exclusive).
        audio:         (T_audio, D_audio) raw audio features for the full utterance.
        idx_lo, idx_hi: half-open flat-token index range to fill: [idx_lo, idx_hi).
        token_strings: list of length (idx_hi - idx_lo) with subword strings (only
                       used by 'token_token_char'); may be None for other modes.
        mode:          'word_token' | 'token_token_uniform' | 'token_token_char'.
    """
    n = idx_hi - idx_lo
    if n <= 0:
        return

    time_axis = 0 if audio.dim() == 2 else 1

    # Safety pad if the span is too thin (matches original behaviour).
    if end_seg - start_seg < 3:
        start_seg = max(0, start_seg - 2)
        end_seg = min(audio.shape[time_axis], end_seg + 2)

    if mode == 'word_token':
        if audio.dim() == 2:
            vec = torch.clamp(audio[start_seg:end_seg].mean(dim=0), min=-1e3, max=1e3)
        else:
            vec = torch.clamp(audio[:, start_seg:end_seg].mean(dim=1), min=-1e3, max=1e3)
        for k in range(n):
            processed[idx_lo + k + 1] = vec
        return

    span = max(end_seg - start_seg, 1)
    if mode == 'token_token_uniform':
        weights = [1.0] * n
    elif mode == 'token_token_char':
        if token_strings is None or len(token_strings) != n:
            weights = [1.0] * n
        else:
            weights = [max(1, len(_strip_subword(t))) for t in token_strings]
    else:
        raise ValueError(f"Unknown alignment_mode: {mode}")

    total_w = float(sum(weights))
    cum = 0.0
    for k in range(n):
        lo = start_seg + int(round(span * cum / total_w))
        cum += weights[k]
        hi = start_seg + int(round(span * cum / total_w))
        if hi <= lo:
            hi = min(audio.shape[time_axis], lo + 1)
        if hi - lo < 1:
            continue
        if audio.dim() == 2:
            vec = torch.clamp(audio[lo:hi].mean(dim=0), min=-1e3, max=1e3)
        else:
            vec = torch.clamp(audio[:, lo:hi].mean(dim=1), min=-1e3, max=1e3)
        processed[idx_lo + k + 1] = vec


def _strip_subword(token):
    """Remove BPE/WordPiece prefix markers so character counts reflect spoken length."""
    t = str(token)
    if t.startswith('##'):
        t = t[2:]
    t = t.lstrip('Ġ')
    return t


def _collect_token_strings(word_mapping, lo_map_idx, hi_map_idx):
    """Flatten subword strings across word_mapping[lo_map_idx : hi_map_idx+1]."""
    out = []
    for wm_idx in range(lo_map_idx, hi_map_idx + 1):
        if 0 <= wm_idx < len(word_mapping):
            out.extend(word_mapping[wm_idx][1])
    return out


def _save_noalign_audio(last_hidden_states_audio, max_length):
    """Temporal mean-pool the raw audio to `max_length` rows for the 'none' mode.

    The model expects a (max_length, D_audio) tensor parallel to text tokens. In
    this mode we drop the per-token alignment entirely and just compress the full
    audio sequence to that fixed shape, so the multimodal layers have to learn
    alignment themselves via cross-attention.
    """
    if last_hidden_states_audio.dim() == 2:
        T, D = last_hidden_states_audio.shape
        out = torch.zeros((max_length, D))
        if T == 0:
            return out
        out[0] = last_hidden_states_audio.mean(dim=0)
        if T <= max_length - 1:
            out[1:1 + T] = last_hidden_states_audio
            return out
        # Mean-pool with uniform bins so all frames contribute.
        edges = [int(round(i * T / (max_length - 1))) for i in range(max_length)]
        for k in range(max_length - 1):
            lo, hi = edges[k], max(edges[k] + 1, edges[k + 1])
            out[k + 1] = torch.clamp(last_hidden_states_audio[lo:hi].mean(dim=0),
                                     min=-1e3, max=1e3)
        return out

    # Layered case: (L, T, D) -> (max_length, L, D)
    L, T, D = last_hidden_states_audio.shape
    out = torch.zeros((max_length, L, D))
    if T == 0:
        return out
    out[0] = last_hidden_states_audio.mean(dim=1)
    if T <= max_length - 1:
        out[1:1 + T] = last_hidden_states_audio.permute(1, 0, 2)
        return out
    edges = [int(round(i * T / (max_length - 1))) for i in range(max_length)]
    for k in range(max_length - 1):
        lo, hi = edges[k], max(edges[k] + 1, edges[k + 1])
        out[k + 1] = torch.clamp(last_hidden_states_audio[:, lo:hi].mean(dim=1),
                                 min=-1e3, max=1e3)
    return out


def preprocess_text():
    init_models()
    
    pauses_data = '_pauses' if pauses else ''
    name_mapping_text = {
        'bert': '',
        'distil': 'distil',
        'distilbert': 'distil',
        'roberta': 'roberta',
        'mistral': 'mistral',
        'qwen': 'qwen',
        'stella': 'stella'
    }
    textual_model_data = name_mapping_text.get(textual_model, '')
    name_mapping_audio = {
        'wav2vec2': 'audio',
        'egemaps': 'egemaps',
        'mel': 'mel'
    }
    audio_model_data = '_' + name_mapping_audio.get(audio_model, '')
    audio_layer_suffix = _get_wav2vec2_layer_suffix() if audio_model == 'wav2vec2' else ''

    # Read textual data from CSV
    df = pd.read_csv(textual_data, encoding='utf-8')

    row_data = 'transcription_pause' if pauses else 'transcription'

    df[row_data] = df[row_data].apply(lambda x: unicodedata.normalize("NFC", str(x)))

    completed_audios = 0

    # Columns are     df = pd.DataFrame(columns=['uid', 'diagno', 'transcription', 'transcription_pause', 'probablities'])

    # Iteate over each row
    for index, row in df.iterrows():
        diagno_str = str(row['diagno']) if pd.notna(row['diagno']) else ''

        print(f"------------------------------------------")
        print(f"------------------------------------------")
        print(f"Processing {row['uid']}, {diagno_str}")


        # Get the transcription
        transcription = row[row_data]

        # Tokenize the transcription
        inputs_text = tokenizer(
            transcription,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Get the embeddings
        with torch.no_grad():
            outputs_text = model(**inputs_text)

        # Save the embeddings
        last_hidden_states_text = outputs_text.last_hidden_state.squeeze(0).cpu()
        save_dir = os.path.join(root_text_path, diagno_str) if diagno_str else root_text_path
        os.makedirs(save_dir, exist_ok=True)
        torch.save(last_hidden_states_text, os.path.join(save_dir, row['uid'] + textual_model_data + pauses_data + '.pt'))

        if audio_model != '':
            if audio_path_resolver is not None:
                audio_path = audio_path_resolver(root_path, row['uid'], diagno_str)
            else:
                audio_path = _default_audio_path(root_path, row['uid'], diagno_str)
            if not audio_path or not os.path.exists(audio_path):
                print(f"WARNING: audio not found for uid={row['uid']}, skipping.")
                continue

            if audio_model == 'wav2vec2':
                try:
                    wave_form, sample_rate = torchaudio.load(audio_path)
                except Exception as e:
                    print(f"WARNING: could not load audio {audio_path}: {e}")
                    continue
                        
                # Convert stereo to mono if necessary
                if wave_form.shape[0] > 1:
                    wave_form = wave_form.mean(dim=0, keepdim=True)

                wave_form = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(wave_form)
                sample_rate = 16000
                wave_form = wave_form.squeeze(0)

                inputs_audio = processor(wave_form, sampling_rate=sample_rate, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs_audio = wav2vec_model(
                        **inputs_audio,
                        output_hidden_states=(wav2vec2_layer_mode == 'weighted'),
                    )

                features_audio = _extract_wav2vec2_features(outputs_audio)
                if features_audio.dim() == 3:
                    processed_audio_tensor = torch.zeros((max_length, features_audio.shape[0], features_audio.shape[2]))
                    _save_wav2vec2_layer_metadata(save_dir, features_audio.shape[0])
                else:
                    processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))
                last_hidden_states_audio = features_audio
            elif audio_model == 'egemaps':
                y, sr = librosa.load(audio_path)
                frame_size = 0.1

                frame_samples = int(frame_size * sr)  # Samples per frame
                frames = librosa.util.frame(y, frame_length=frame_samples, hop_length=frame_samples).T

                features = []
                for frame in frames:
                    features.append(smile.process_signal(frame, sr))
                
                features = np.vstack(features)

                features_audio = torch.tensor(features).float().to(device)
                print(f"Features shape: {features_audio.shape}")

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    print(f"ERROR BEFORE in {diagno_str}, {row['uid']}: NaN values in features_audio")
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)
                
                last_hidden_states_audio = features_audio
            elif audio_model == 'mel':
                y, sr = librosa.load(audio_path)

                win_length = int(0.02 * sr)  # 20 ms en samples
                hop_length = int(0.02 * sr)  # 20 ms también para 50 segmentos por segundo
                n_mels = 80  # Número típico de filtros mel

                mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=win_length, hop_length=hop_length, n_mels=n_mels)

                features_audio = torch.tensor(mel).float().permute(1,0)

                processed_audio_tensor = torch.zeros((max_length, features_audio.shape[1]))

                if torch.isnan(features_audio).any():
                    features_audio = torch.nan_to_num(features_audio, nan=0.0)
                
                last_hidden_states_audio = features_audio
        
            if features_audio.dim() == 2:
                processed_audio_tensor[0] = features_audio.mean(dim=0)
            else:
                processed_audio_tensor[0] = features_audio.mean(dim=1)

            # 'none' alignment: skip per-word loop and store a temporally-pooled
            # raw audio sequence. Saves under a different suffix so it coexists
            # with the aligned variants on disk.
            if alignment_mode == 'none':
                processed_audio_tensor = _save_noalign_audio(
                    last_hidden_states_audio, max_length
                )
                noalign_suffix = ALIGNMENT_SUFFIX.get(alignment_mode, '')
                if torch.isnan(processed_audio_tensor).any():
                    print(f"ERROR in {diagno_str}, {row['uid']}: NaN values in processed_audio_tensor (noalign)")
                    print(f"Completed audios: {completed_audios}")
                    continue
                torch.save(
                    processed_audio_tensor,
                    os.path.join(
                        save_dir,
                        row['uid'] + textual_model_data + pauses_data
                        + audio_model_data + noalign_suffix + audio_layer_suffix + '.pt',
                    ),
                )
                completed_audios += 1
                continue

            # Tokenize and prepare inputs
            inputs_offset = tokenizer(
                transcription,
                return_tensors="pt",
                return_offsets_mapping=True,  # Get token-to-offset mappings
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)


            # Extract word-to-token mapping
                # print(text)
            input_ids = inputs_offset["input_ids"][-1]
            offset_mapping = inputs_offset["offset_mapping"][-1]

            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            word_mapping = []

            current_word = ""
            current_tokens = []
            current_token_ids = []

            for token, offset, token_id in zip(tokens, offset_mapping.tolist(), input_ids.tolist()):
                start, end = offset

                # Skip special tokens ([CLS], [SEP], [PAD])
                if start == 0 and end == 0:
                    continue

                # Check for subwords (##) and group tokens into words
                if token.startswith("##"):
                    current_word += token[2:]
                    current_tokens.append(token)
                    current_token_ids.append(token_id)
                else:
                    # Save previous word
                    if current_word:
                        word_mapping.append((current_word, current_tokens, current_token_ids))
                    # Start a new word
                    current_word = token
                    current_tokens = [token]
                    current_token_ids = [token_id]

            # Save the last word
            if current_word:
                word_mapping.append((current_word, current_tokens, current_token_ids))

            word_level_timestamp_path = os.path.join(save_dir, row['uid'] + '.csv')

            # Read the word level timestamps
            df_word_level = pd.read_csv(word_level_timestamp_path)
            # Columns pandas_word_level = pd.DataFrame(columns=['word', 'start', 'end', 'probability'])
            words = []
            for index, data in df_word_level.iterrows():
                words.append((data['word'], data['start'], data['end']))

            idx_probs = 0
            act_word = ''

            idx_att = 0
            idx_start_att = 0

            idx_start_map = 0
            idx_map = 0

            n_audio_segments = 0

            # Print results
            for word, tokens, token_ids in word_mapping:
                # print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                cleaned_word = word.replace('Ġ', '')
                act_word += cleaned_word.replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()

                print(f"Word: {word}, Tokens: {tokens}, Token IDs: {token_ids}")
                print(f"Act Word: {act_word}")
                if idx_probs < len(words):
                    # Check if words[idx_probs][0] is a string before printing
                    if isinstance(words[idx_probs][0], str):
                        print(f"Expected Word: {words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower()}")

                if word.strip() in ['.', ',', '?', '!', ';', 'Ġ','Ġ.', 'Ġ,', 'Ġ?', 'Ġ!', 'Ġ;', 'Ġ...', '...']:    # Ensure only real punctuation
                    if idx_probs > 0:  # Avoid index error
                        start = words[idx_probs-1][2]  # Get last word's end time
                    else:
                        start = 0  # Default to 0 if first word
                    end = words[idx_probs][1] if idx_probs < len(words) else None  # Safe check

                    start_segment = math.floor(start * segment_length)
                    audio_time_len = last_hidden_states_audio.shape[0] if last_hidden_states_audio.dim() == 2 else last_hidden_states_audio.shape[1]
                    end_segment = math.ceil(end * segment_length if end is not None else audio_time_len)
                    print(f"FOUND PUNCTUATION: {word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                    print("Token IDs:")
                    for idx in range(idx_start_map, idx_map + 1):
                        print(f"{idx}: {word_mapping[idx]}")
                        print('------------------------------------------')

                    idx_hi = idx_att + len(token_ids)
                    n_audio_segments += idx_hi - idx_start_att
                    tok_strings = _collect_token_strings(word_mapping, idx_start_map, idx_map)
                    _fill_token_audio(
                        processed_audio_tensor, start_segment, end_segment,
                        last_hidden_states_audio,
                        idx_start_att, idx_hi,
                        token_strings=tok_strings, mode=alignment_mode,
                    )

                    idx_start_att = idx_hi
                    idx_start_map = idx_map + 1



                if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word == words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                    
                        start = words[idx_probs][1]
                        end = words[idx_probs][2]

                        start_segment = math.floor(start * segment_length)
                        audio_time_len = last_hidden_states_audio.shape[0] if last_hidden_states_audio.dim() == 2 else last_hidden_states_audio.shape[1]
                        end_segment = math.ceil(end * segment_length if end is not None else audio_time_len)

                        print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                        print("Token IDs:")
                        for idx in range(idx_start_map, idx_map + 1):
                            print(f"{idx}: {word_mapping[idx]}")
                            print('------------------------------------------')

                        idx_hi = idx_att + len(token_ids)
                        n_audio_segments += idx_hi - idx_start_att
                        tok_strings = _collect_token_strings(word_mapping, idx_start_map, idx_map)
                        _fill_token_audio(
                            processed_audio_tensor, start_segment, end_segment,
                            last_hidden_states_audio,
                            idx_start_att, idx_hi,
                            token_strings=tok_strings, mode=alignment_mode,
                        )

                        idx_probs += 1
                        act_word = ''
                        idx_start_att = idx_hi
                        idx_start_map = idx_map + 1

                idx_att += len(token_ids)
                idx_map += 1

            if idx_probs < len(words) and isinstance(words[idx_probs][0], str) and act_word in words[idx_probs][0].replace('Ġ', '').replace('.', '').replace(',', '').replace(';', '').replace(' ', '').lower():
                start = words[idx_probs][1]
                end = words[idx_probs][2]

                start_segment = math.floor(start * segment_length)
                audio_time_len = last_hidden_states_audio.shape[0] if last_hidden_states_audio.dim() == 2 else last_hidden_states_audio.shape[1]
                end_segment = math.ceil(end * segment_length if end is not None else audio_time_len)

                print(f"FOUND WORD: {act_word}, Start: {start}, End: {end}, Start Segment: {start_segment}, End Segment: {end_segment}")
                print("Token IDs:")
                for idx in range(idx_start_map, idx_map):
                    print(f"{idx}: {word_mapping[idx]}")
                    print('------------------------------------------')

                idx_hi = idx_att
                n_audio_segments += max(0, idx_hi - idx_start_att)
                tok_strings = _collect_token_strings(word_mapping,
                                                    idx_start_map, idx_map - 1)
                _fill_token_audio(
                    processed_audio_tensor, start_segment, end_segment,
                    last_hidden_states_audio,
                    idx_start_att, idx_hi,
                    token_strings=tok_strings, mode=alignment_mode,
                )

                idx_probs += 1
                act_word = ''
                idx_start_att = idx_att + len(token_ids)
                idx_start_map = idx_map + 1


            print(f"Number of audio segments: {n_audio_segments}")        
            # See inputs numbers and compare with the number of audio segments, separate with PAD tokens, eclusding them
            #total_tokens = torch.sum(inputs_text['input_ids'][0] != 0).item()
            total_tokens = torch.sum(inputs_text['attention_mask'][0]).item()
            print(f"Total tokens: {total_tokens}")
            if n_audio_segments + 2 != total_tokens:
                print(f"ERROR in {diagno_str}, {row['uid']}: Number of audio segments ({n_audio_segments}) does not match the number of tokens ({total_tokens})")
                print(f"Completed audios: {completed_audios}")
                continue

            if torch.isnan(processed_audio_tensor).any():
                print(f"ERROR in {diagno_str}, {row['uid']}: NaN values in processed_audio_tensor")
                print(f"Completed audios: {completed_audios}")
                continue
            
            align_suffix = ALIGNMENT_SUFFIX.get(alignment_mode, '')
            torch.save(processed_audio_tensor, os.path.join(save_dir, row['uid'] + textual_model_data + pauses_data + audio_model_data + align_suffix + audio_layer_suffix + '.pt'))

            
            completed_audios += 1

        print(f"------------------------------------------")
        print(f"CORRECTLY PROCESSED ALL AUDIOS")
        print(f"Completed audios: {completed_audios}")

if __name__ == '__main__':
    preprocess_text()