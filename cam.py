import os
import json
import torchaudio
import torch
import numpy as np
import soundfile
from pathlib import Path
from tqdm import tqdm
from glob import glob
from collections import defaultdict
from modelscope.pipelines import pipeline
from sklearn.cluster import AgglomerativeClustering


# 初始化 PyAnnote 说话人嵌入模型
print("Loading model...")
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_campplus_sv_zh-cn_16k-common',
    model_revision='v1.0.0'
)

def extract_embedding(wav_path):
    waveform, sr = torchaudio.load(wav_path)

    # ✅ 如果采样率不是16k，进行重采样
    target_sr = 16000
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)

        # 可选：保存临时重采样后的音频，用于 pipeline 输入
        tmp_wav_path = "/tmp/tmp_resampled.wav"
        torchaudio.save(tmp_wav_path, waveform, target_sr)
    else:
        tmp_wav_path = wav_path

    try:
        result = sv_pipeline([tmp_wav_path, tmp_wav_path], output_emb=True)
    except soundfile.LibsndfileError:
        print(f"❌ Failed to load {wav_path}, skipping...")
        return None
    emb = result['embs'][0]
    return emb

    # result = sv_pipeline([wav_path, wav_path], output_emb=True)
    # emb = result['embs'][0]
    # return emb

def cluster_embeddings(embeddings, threshold=0.3):
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )
    return clustering.fit_predict(embeddings)

def process_subdir(subdir_path, output_dir):
    audio_dir = os.path.join(subdir_path, "audio")
    json_file = os.path.join(subdir_path, "json", "merged_audio_text.json")

    if not os.path.exists(audio_dir) or not os.path.exists(json_file):
        print(f"Missing audio or json in {subdir_path}, skipping...")
        return

    audio_files = sorted(glob(os.path.join(audio_dir, "*.wav")))
    if not audio_files:
        print(f"No .wav files in {audio_dir}")
        return

    print(f"\n📁 Processing: {subdir_path}  |  {len(audio_files)} files")

    with open(json_file, "r", encoding="utf-8") as f:
        try:
            meta = json.load(f)
            if not isinstance(meta, list): meta = [meta]
        except Exception as e:
            print(f"❌ Failed to parse {json_file}: {e}")
            return

    # 建立音频路径 -> text 映射
    path2text = {}
    for item in meta:
        fname = item.get("path")
        if fname:
            path2text[fname] = item.get("full_text", "unknown")

    embeddings = []
    utt_ids = []
    file_paths = []

    for wav_path in tqdm(audio_files, desc="Extracting embeddings"):
        emb = extract_embedding(wav_path)
        embeddings.append(emb)
        fname = os.path.basename(wav_path)
        utt_id = f"{Path(subdir_path).name}_{Path(fname).stem}"
        utt_ids.append(utt_id)
        file_paths.append(wav_path)

    embeddings = np.vstack(embeddings)
    labels = cluster_embeddings(embeddings)

    # 输出 kaldi 文件
    out_dir = os.path.join(output_dir, Path(subdir_path).name)
    os.makedirs(out_dir, exist_ok=True)

    utt2spk = {}
    spk2utt = defaultdict(list)

    with open(os.path.join(out_dir, "wav.scp"), "w") as f_wav, \
         open(os.path.join(out_dir, "utt2spk"), "w") as f_utt2spk, \
         open(os.path.join(out_dir, "text"), "w") as f_text:

        for utt_id, label, wav_path in zip(utt_ids, labels, file_paths):
            spk_id = f"spk_{label}"
            utt2spk[utt_id] = spk_id
            spk2utt[spk_id].append(utt_id)

            f_wav.write(f"{utt_id} {wav_path}\n")
            f_utt2spk.write(f"{utt_id} {spk_id}\n")

            fname = os.path.basename(wav_path)
            text = path2text.get(fname, "unknown")
            f_text.write(f"{utt_id} {text}\n")

    with open(os.path.join(out_dir, "spk2utt"), "w") as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")

    print(f"✅ Done: {subdir_path} -> {out_dir} (Detected speakers: {len(spk2utt)})")

def main(input_root, output_root):
    process_subdir(input_root, output_root)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="根目录，包含多个 subdir")
    parser.add_argument("--output_dir", default="kaldi_data", help="输出 Kaldi 格式目录")
    args = parser.parse_args()

    main(args.input_dir, args.output_dir)
