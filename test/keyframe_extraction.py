import torch
import clip
import os
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import transformers

# 加载CLIP模型
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
 
# 计算余弦相似度的函数
def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a, b)
 
# 关键帧选择策略
def select_keyframes(video_dataset, model, device):
    keyframes = []  # 存储关键帧索引
    video_features = []  # 存储帧的特征向量
    
    with torch.no_grad():
        # 提取第一帧的特征并将其设为第一个关键帧
        first_frame_data = video_dataset[0]
        first_frame = first_frame_data['img'].unsqueeze(0).to(device)
 
        # 检查图像是否为RGB三通道格式
        if first_frame.shape[1] == 1:  # 如果是单通道，扩展为三通道
            first_frame = first_frame.repeat(1, 3, 1, 1)
 
        # 对帧进行预处理并提取特征
        first_frame_preprocessed = preprocess(transforms.ToPILImage()(first_frame.squeeze(0).cpu())).unsqueeze(0).to(device)
        first_frame_features = model.encode_image(first_frame_preprocessed)
        
        # 将第一帧设为关键帧
        keyframes.append(0)
        video_features.append(first_frame_features)
 
        # 遍历剩下的帧
        for idx in tqdm(range(1, len(video_dataset))):
            current_frame_data = video_dataset[idx]
            current_frame = current_frame_data['img'].unsqueeze(0).to(device)
 
            # 检查图像格式是否为RGB三通道
            if current_frame.shape[1] == 1:
                current_frame = current_frame.repeat(1, 3, 1, 1)
 
            # 预处理当前帧并提取特征
            current_frame_preprocessed = preprocess(transforms.ToPILImage()(current_frame.squeeze(0).cpu())).unsqueeze(0).to(device)
            current_frame_features = model.encode_image(current_frame_preprocessed)
 
            # 计算当前帧与上一关键帧的相似度
            similarity = cosine_similarity(current_frame_features, video_features[-1])
 
            # 如果相似度低于设定的阈值，则将当前帧设为新的关键帧
            if similarity.item() < 0.85:  # 可以调整阈值
                keyframes.append(idx)
                video_features.append(current_frame_features)
 
    return keyframes


# 加载视频数据集
def load_video_dataset(frames_folder):
    video_dataset = []
    for file in os.listdir(frames_folder):
        if file.endswith('.jpg'):
            img = Image.open(f'{frames_folder}/{file}')
            img_tensor = transforms.ToTensor()(img)
            video_dataset.append({'img': img_tensor, 'file_name': file})
    return video_dataset


if __name__ == '__main__':
    # 加载视频数据集
    video_dataset = load_video_dataset('./output_frames')
    
    # 选择关键帧
    keyframes = select_keyframes(video_dataset, model, device)
    
    # 输出关键帧索引
    print(keyframes)
