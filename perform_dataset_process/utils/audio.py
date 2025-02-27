import moviepy.editor as mp
import os

def extract_audio_from_video(video_path, audio_path=None):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to save the extracted audio file.
    """
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    video_name = os.path.basename(video_path)
    audio_name = video_name.replace('.mp4', '.wav')
    if not os.path.exists(video_path.replace('.mp4', '')):
        os.makedirs(os.path.dirname(video_path))
    audio_path = audio_path or os.path.join(os.path.dirname(video_path), video_name)
    audio_path = os.path.join(audio_path, audio_name)
    audio.write_audiofile(audio_path)
    return audio
