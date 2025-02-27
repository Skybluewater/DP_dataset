import moviepy.editor as mp

def extract_audio_from_video(video_path, audio_path=None):
    """
    Extracts audio from a video file and saves it as an audio file.

    Args:
        video_path (str): Path to the video file.
        audio_path (str): Path to save the extracted audio file.
    """
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    if audio_path is None:
        audio_path = video_path.replace('.mp4', '.wav')
    audio.write_audiofile(audio_path)
    return audio
