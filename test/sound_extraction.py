import moviepy.editor as mp

def split_audio(video_path, output_audio_path):
    video = mp.VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)


def sound_extraction(sound_file_path):
    from funasr import AutoModel
    # paraformer-zh is a multi-functional asr model
    # use vad, punc, spk or not as you need
    model = AutoModel(model="paraformer-zh", model_revision="v2.0.4",
                    vad_model="fsmn-vad", vad_model_revision="v2.0.4",
                    punc_model="ct-punc-c", punc_model_revision="v2.0.4",
                    # spk_model="cam++", spk_model_revision="v2.0.2",
                    )
    res = model.generate(input=sound_file_path, 
                        batch_size_s=300, 
                        hotword='魔搭')
    return res


if __name__ == "__main__":
    video_path = "video_1.mp4"
    output_folder = "./output_1"

    output_audio_path = "audio_test.wav"
    # split_audio(video_path, output_audio_path)
    sound_extraction(output_audio_path)