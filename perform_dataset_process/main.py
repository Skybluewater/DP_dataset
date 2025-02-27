import base64
import requests
import json
import os

from openai import OpenAI
from utils import extract_text_from_audio, align_chunks_with_timestamps, split_video_frames_by_duration, extract_keyframes, extract_audio_from_video, image_to_base64
from PROMPT import PROMPTS
"""
    now 
"""

image_path = 'test/keyframe_144.60.jpg'
base64_image = image_to_base64(image_path)

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

background = "北京冬奥会开幕式节目《致敬人民 一起向未来》"
description = "现场大屏幕展现了冰雪运动员的运动瞬间，空中模拟跳台滑雪的发光人形从冰雪五环旁飞过场地内，四组滑冰运动员在雪地上滑出长长的轨迹。"
prompt = PROMPTS["frame_expansion"][1]
model = ["Qwen/Qwen2-VL-72B-Instruct", "Qwen/QVQ-72B-Preview"]


response = client.chat.completions.create(
        model=model[1],
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }
                },
                {
                    "type": "text",
                    "text": prompt.format(
                        background=background,
                        description=description
                    )
                }
            ]
        }],
        stream=True
)

for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)

