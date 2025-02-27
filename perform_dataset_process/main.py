import base64
import requests
import json
import os

from openai import OpenAI
from utils import extract_text_from_audio, align_chunks_with_timestamps, split_video_frames_by_duration, extract_keyframes, extract_audio_from_video
from PROMPT import PROMPTS
"""
    now 
"""


def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

image_path = 'test/keyframe_110.10.jpg'
base64_image = image_to_base64(image_path)

client = OpenAI(
    api_key=os.getenv("SILICON_API_KEY"),
    base_url="https://api.siliconflow.cn/v1"
)

response = client.chat.completions.create(
        model="Qwen/QVQ-72B-Preview",
        messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    }
                },
                {
                    "type": "text",
                    "text": PROMPTS["frame_expansion"].format(
                        background="2022年北京冬奥会开幕式",
                        description="现在大家听到的歌曲想象是奥运会开幕式的保留曲目，国脚会希望通过这首歌传递，打破隔阂，全世界团结一心的愿景。"
                    )
                }
            ]
        }],
        stream=True
)

for chunk in response:
    chunk_message = chunk.choices[0].delta.content
    print(chunk_message, end='', flush=True)

