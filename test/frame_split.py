import cv2
import os

def extract_frames(video_path, output_folder, interval=1):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    frame_count = 0
    extracted_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{extracted_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {extracted_count} frames.")

import moviepy.editor as mp

def extract_audio(video_path, output_audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)

if __name__ == "__main__":
    video_path = "video_test.mp4"
    output_folder = "./output_frames"
    extract_frames(video_path, output_folder)
    
    output_audio_path = "audio_test.wav"
    extract_audio(video_path, output_audio_path)