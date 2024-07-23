import os
import cv2
import json
import textwrap
import numpy as np
import scipy.io.wavfile as wav
import moviepy.editor as mpe


def crop(img, x, y, w, h):
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    x1, y1 = x0 + w, y0 + h
    return img[y0:y1, x0:x1]


def create_image_video(image_files, durations, transcripts, output_file="output.mp4"):
    video_dim = (1024, 1024)
    fps = 30
    start_center = (0.4, 0.6)
    end_center = (0.5, 0.5)
    start_scale = 0.7
    end_scale = 1.0
    frames = []

    for image_file, duration, transcript in zip(image_files, durations, transcripts):
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        orig_shape = img.shape[:2]

        num_frames = int(fps * duration)
        for alpha in np.linspace(0, 1, num_frames):
            rx = end_center[0] * alpha + start_center[0] * (1 - alpha)
            ry = end_center[1] * alpha + start_center[1] * (1 - alpha)
            x = int(orig_shape[1] * rx)
            y = int(orig_shape[0] * ry)
            scale = end_scale * alpha + start_scale * (1 - alpha)

            if orig_shape[1] / orig_shape[0] > video_dim[0] / video_dim[1]:
                h = int(orig_shape[0] * scale)
                w = int(h * video_dim[0] / video_dim[1])
            else:
                w = int(orig_shape[1] * scale)
                h = int(w * video_dim[1] / video_dim[0])

            cropped = crop(img, x, y, w, h)
            scaled = cv2.resize(
                cropped, dsize=video_dim, interpolation=cv2.INTER_LINEAR
            )

            transcript_wraped = textwrap.wrap(transcript, width=50)
            for i, line in enumerate(transcript_wraped):
                textsize = cv2.getTextSize(line, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2)[0]

                gap = textsize[1] + 10

                y = int((1.5*img.shape[0] + textsize[1]) / 2) + i * gap
                x = int((img.shape[1] - textsize[0]) / 2)
                scaled = cv2.putText(
                    scaled,
                    line,
                    org=(x, y),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(255, 255, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
            frames.append(scaled)

        # write to MP4 file
    vidwriter = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim
    )
    for frame in frames:
        vidwriter.write(frame)
    vidwriter.release()


def read_and_concatenate_wavs(filenames, output_filename):
    durations = []
    audio_data = []
    for filename in filenames:
        fs, data = wav.read(filename)
        duration = len(data) / fs
        durations.append(duration)

        data = data.astype(np.float32)

        audio_data.append(data)

    concatenated_data = np.concatenate(audio_data)

    wav.write(output_filename, fs, concatenated_data)

    return durations


def create_video(dir_path):
    image_files = sorted(os.listdir(dir_path + "/image"), key=lambda x:int(x.split(".")[0]))
    image_files = [os.path.join(dir_path, "image", fn) for fn in image_files]

    voice_files = sorted(os.listdir(dir_path + "/voice"), key=lambda x:int(x.split(".")[0]))
    voice_files = [os.path.join(dir_path, "voice", fn) for fn in voice_files]

    transcript_file = os.path.join(dir_path, "text_chunk.json")
    with open(transcript_file, "r") as f:
        transcripts = json.load(f)["text_chunks"]

    assert len(image_files) == len(voice_files), "Images and Voices do not have the same number of files"

    durations = read_and_concatenate_wavs(
        voice_files, output_filename=os.path.join(dir_path, "combined_audio.wav")
    )
    create_image_video(
        image_files,
        durations,
        transcripts,
        output_file=os.path.join(dir_path, "image_video.mp4"),
    )
    video = mpe.VideoFileClip(os.path.join(dir_path, "image_video.mp4"))
    voice = mpe.AudioFileClip(os.path.join(dir_path, "combined_audio.wav"))
    final_video = video.set_audio(voice)
    final_video.write_videofile(os.path.join(dir_path, "final_video.mp4"),codec= 'libx264' ,audio_codec='libvorbis')

if __name__ == "__main__":
    create_video("export/2024-07-23-08-09-40")
