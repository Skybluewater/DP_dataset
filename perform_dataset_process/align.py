import os
import logging

from utils import align_img_with_chunk

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

"""
    This file is used to extract video frames from the video files.
    input:
        video_file_dir: the directory of the video file
    output:
        each video file should has it's own directory, and the frames are stored in the directory
        the directory name is the same as the video file name
    Requirements:
        you should use methods from utils to extract frames from the video file
        like split_video_frames_by_duration, extract_keyframes
"""
def align_img(video_file_dir):
    # Iterate over video files in the directory
    for dir_name in os.listdir(video_file_dir):
        dir_path = os.path.join(video_file_dir, dir_name)
        if os.path.isdir(dir_path):
            align_img_with_chunk(dir_path)
            log.info(f"Aligning images in {dir_name} is done.")


if __name__ == "__main__":
    video_file_dir = "2008BeijingOlympicGames"
    align_img(video_file_dir)    
