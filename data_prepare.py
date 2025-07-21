import os
import json
import glob
import hashlib
from tqdm import tqdm
import logging
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger()

class SpeakerDiarizer:
    def __init__(self, threshold=0.3, min_clusters=1, max_clusters=10):
        self.encoder = VoiceEncoder()
        self.threshold = threshold  # 余弦相似度阈值
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

    def cluster_speakers(self, audio_paths):
        # 预处理和特征提取
        wavs = [preprocess_wav(Path(fp)) for fp in tqdm(audio_paths, desc="Preprocessing")]
        embeddings = np.array([self.encoder.embed_utterance(wav) for wav in tqdm(wavs, desc="Extracting")])
        
        # 可视化嵌入分布（调试用）
        self._plot_embeddings(embeddings)
        print(len(embeddings),"embeddings extracted")
        # 自动确定聚类数量
        if len(embeddings) <= 1:  # 处理边界情况
            return {audio_paths[0]: "spk_0"} if audio_paths else {}
            
        # 使用层次聚类+自动阈值
        clustering = AgglomerativeClustering(
            n_clusters=None,  # 自动确定
            metric='cosine',
            linkage='average',
            distance_threshold=self.threshold
        ).fit(embeddings)
        
        # 确保聚类数量在合理范围内
        n_clusters = max(self.min_clusters, min(clustering.n_clusters_, self.max_clusters))
        
        # 如果自动聚类不合理，回退到固定数量
        if clustering.n_clusters_ == 1 and len(embeddings) > 3:  # 假设至少应该有多个说话人
            logger.warning("Auto-clustering resulted in 1 cluster, trying with n_clusters=2")
            clustering = AgglomerativeClustering(
                n_clusters=min(2, self.max_clusters),
                metric='cosine',
                linkage='average'
            ).fit(embeddings)
            n_clusters = clustering.n_clusters_
        
        # 生成标签
        labels = clustering.labels_
        return {audio_path: f"spk_{label}" for audio_path, label in zip(audio_paths, labels)}
    
    def _plot_embeddings(self, embeddings):
        """可视化嵌入分布（用于调试）"""
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            # 降维到2D
            pca = PCA(n_components=2)
            reduced = pca.fit_transform(embeddings)
            
            plt.scatter(reduced[:, 0], reduced[:, 1])
            plt.title("Speaker Embeddings (PCA)")
            plt.show()
        except Exception as e:
            logger.warning(f"Failed to plot embeddings: {str(e)}")

def process_custom_data(input_dir, output_dir, diarizer=None, target_subdir=None):
    # 初始化数据结构
    utt2wav = {}      # {utterance_id: wav_path}
    utt2text = {}     # {utterance_id: text_content}
    utt2spk = {}      # {utterance_id: speaker_id}
    utt2gender = {}   # {utterance_id: gender}
    
    # 第一阶段：收集所有音频和对应JSON
    audio_paths = []
    
    for subdir in tqdm(os.listdir(input_dir), desc="Scanning subdirectories"):
        if target_subdir and subdir != target_subdir:  # 如果指定了target_subdir，则跳过其他目录
            continue
        subdir_path = os.path.join(input_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        audio_dir = os.path.join(subdir_path, 'audio')
        json_dir = os.path.join(subdir_path, 'json')
        json_file = os.path.join(json_dir, f"merged_audio_text.json")  # json/subdir.json
        
        if not os.path.exists(audio_dir) or not os.path.exists(json_file):
            logger.warning(f"Missing audio dir or JSON in {subdir}")
            continue
        
        # 读取JSON文件（包含该子目录下所有音频的标注）
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                items = json.load(f)
                if not isinstance(items, list):
                    items = [items]  # 兼容单条数据的情况
            except Exception as e:
                logger.error(f"Failed to load {json_file}: {str(e)}")
                continue
        
        # 处理每个音频条目
        for item in items:
            if 'path' not in item:
                logger.warning(f"Missing 'path' field in {json_file}")
                continue
                
            audio_path = os.path.join(audio_dir, item['path'])  # 从JSON中提取相对路径
            if not os.path.exists(audio_path):
                logger.warning(f"Missing audio file: {audio_path}")
                continue
            
            # 生成唯一utterance_id（格式：子目录名_音频文件名）
            audio_filename = os.path.splitext(os.path.basename(item['path']))[0]
            utt_id = f"{subdir}_{audio_filename}"
            
            # 保存音频路径
            audio_paths.append(audio_path)
            
            # # 提取文本内容（保留副语言标签）
            # if 'vad_segments' not in item:
            #     logger.warning(f"Missing 'vad_segments' in {json_file} for {item['path']}")
            #     continue
                
            # clean_text = ' '.join(
            #     seg['text'] for seg in item['vad_segments']
            # )
            
            # 保存到字典
            utt2wav[utt_id] = audio_path
            utt2text[utt_id] = item.get('full_text','unknown')
            utt2gender[utt_id] = item.get('gender', 'unknown')
    
    # 第二阶段：说话人聚类
    speaker_mapping = diarizer.cluster_speakers(audio_paths) if diarizer else {
        path: f"spk_{hashlib.md5(path.encode()).hexdigest()[:8]}" for path in audio_paths
    }
    
    # 将说话人映射添加到utt2spk
    for utt_id, wav_path in utt2wav.items():
        utt2spk[utt_id] = speaker_mapping[wav_path]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入Kaldi格式文件
    def write_dict_to_file(data, filename):
        with open(os.path.join(output_dir, filename), 'w') as f:
            for key, value in data.items():
                f.write(f"{key} {value}\n")
    
    write_dict_to_file(utt2wav, 'wav.scp')
    write_dict_to_file(utt2text, 'text')
    write_dict_to_file(utt2spk, 'utt2spk')
    
    # 生成spk2utt
    spk2utt = {}
    for utt, spk in utt2spk.items():
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)
    
    with open(os.path.join(output_dir, 'spk2utt'), 'w') as f:
        for spk, utts in spk2utt.items():
            f.write(f"{spk} {' '.join(utts)}\n")
    
    # 保存额外元数据
    with open(os.path.join(output_dir, 'utt2gender'), 'w') as f:
        for utt, gender in utt2gender.items():
            f.write(f"{utt} {gender}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='根目录包含多个子文件夹')
    parser.add_argument('--output_dir', required=True, help='输出Kaldi格式目录')
    parser.add_argument('--target_subdir', default=None, help='指定处理的子目录（如subdir1），默认处理所有子目录')
    parser.add_argument('--no_diarization', action='store_true', help='禁用说话人聚类')
    args = parser.parse_args()
    
    diarizer = None if args.no_diarization else SpeakerDiarizer()
    process_custom_data(args.input_dir, args.output_dir, diarizer, args.target_subdir)