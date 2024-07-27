import os
import cv2
import json
import textwrap
import numpy as np
import scipy.io.wavfile as wav
import librosa
import soundfile as sf
import moviepy.editor as mpe
from PIL import ImageFont, Image, ImageDraw

def crop(img, x, y, w, h):
    x0, y0 = max(0, x - w // 2), max(0, y - h // 2)
    x1, y1 = x0 + w, y0 + h
    return img[y0:y1, x0:x1]


def create_image_video(
        image_files,
        durations,
        transcripts,
        fps,
        font_type,
        sub_position_vertical,
        sub_position_horizontal,
        sub_alignment,
        sub_color,
        output_file="output.mp4"):
    video_dim = cv2.imread(image_files[0], cv2.IMREAD_COLOR).shape[0:2][::-1]
    font_size = 32
    vidwriter = cv2.VideoWriter(
        output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, video_dim
    )
    if font_type == "":
        font = ImageFont.truetype("font/FreeMono.ttf", font_size)
    else:
        font = ImageFont.truetype(font_type, font_size)

    start_center = (0.4, 0.6)
    end_center = (0.5, 0.5)
    start_scale = 0.7
    end_scale = 1.0
    frames = []

    for image_file, duration, transcript in zip(image_files, durations, transcripts):
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        upscale = cv2.resize(img, dsize=(int(img.shape[1]* 4),int(img.shape[0]*4)), interpolation=cv2.INTER_LANCZOS4)
        orig_shape = upscale.shape[:2]

        num_frames = int(fps * duration)
    # Write subtitle
        if sub_color == "yellow":
            color=(0, 255, 255)
        else:
            color=(255, 255, 255)

        transcript_wraped = textwrap.wrap(transcript, width=40)
        lines_coordinate = [] # contain (x,y) coordinate of each row

        # find x_max, y_max which is the max value of x and y where we can start to put text
        gap = font_size + 10 # height of each line
        y_max = img.shape[0] -  len(transcript_wraped)*gap

        longest_length = 0
        for i, line in enumerate(transcript_wraped):
            textsize = font.getbbox(line)[2:]

            if textsize[0] > longest_length:
                longest_length = textsize[0]
        x_max = img.shape[1] - longest_length

        for i, line in enumerate(transcript_wraped):
            textsize = font.getbbox(line)[2:]

            y = y_max * sub_position_vertical + (i+1) * gap
            if sub_alignment == "left":
                x = x_max * sub_position_horizontal
            elif sub_alignment == "mid":
                x = x_max * sub_position_horizontal + (longest_length - textsize[0])/2
            elif sub_alignment == "right":
                x = x_max * sub_position_horizontal + longest_length - textsize[0]
            lines_coordinate.append((int(x),int(y)))
            
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
            
            cropped = crop(upscale, x, y, w, h)
            scaled = cv2.resize(
                cropped, dsize=video_dim, interpolation=cv2.INTER_LANCZOS4
            )
            img_pil = Image.fromarray(scaled)
            draw = ImageDraw.Draw(img_pil)
            for coor, line in zip(lines_coordinate, transcript_wraped):
                draw.text(xy=coor,text=line,font=font,fill=color,stroke_width=1)
            scaled = np.array(img_pil)

            vidwriter.write(scaled)
        # write to MP4 file
    
       
    vidwriter.release()


def read_and_concatenate_wavs(filenames, speed, output_filename):
    durations = []
    audio_data = []
    for filename in filenames:
        fs, data = wav.read(filename)
        duration = len(data) / fs / speed
        durations.append(duration)

        data = data.astype(np.float32)

        audio_data.append(data)

    concatenated_data = np.concatenate(audio_data)
    y_shifted = librosa.effects.pitch_shift(y=concatenated_data, sr=24000, n_steps=int(round(-12 * np.log2(speed))))
    sf.write(output_filename, y_shifted, int(24000*speed), 'PCM_24')
    return durations


def create_video(dir_path,
                 fps=60,
                 speed=1.0,
                 font_type="",
                 sub_position_vertical=1.0,
                 sub_position_horizontal=1.0,
                 sub_alignment="mid",
                 sub_color="yellow"):
    image_files = sorted(os.listdir(dir_path + "/image"), key=lambda x:int(x.split(".")[0]))
    image_files = [os.path.join(dir_path, "image", fn) for fn in image_files]

    voice_files = sorted(os.listdir(dir_path + "/voice"), key=lambda x:int(x.split(".")[0]))
    voice_files = [os.path.join(dir_path, "voice", fn) for fn in voice_files]

    transcript_file = os.path.join(dir_path, "text_chunk.json")
    with open(transcript_file, "r") as f:
        transcripts = json.load(f)["text_chunks"]

    assert len(image_files) == len(voice_files), "Images and Voices do not have the same number of files"

    durations = read_and_concatenate_wavs(
        voice_files,
        speed=speed,
        output_filename=os.path.join(dir_path, "combined_audio.wav")
    )
    create_image_video(
        image_files,
        durations,
        transcripts,
        fps,
        font_type=font_type,
        sub_position_vertical=sub_position_vertical,
        sub_position_horizontal=sub_position_horizontal,
        sub_alignment=sub_alignment,
        sub_color=sub_color,
        output_file=os.path.join(dir_path, "image_video.mp4"),
    )
    video = mpe.VideoFileClip(os.path.join(dir_path, "image_video.mp4"))
    voice = mpe.AudioFileClip(os.path.join(dir_path, "combined_audio.wav"))
    final_video = video.set_audio(voice)
    output_name = f"final_video_{fps}_x{speed}_{sub_position_vertical}_{sub_position_horizontal}_{sub_alignment}_{sub_color}.mp4"
    final_video.write_videofile(os.path.join(dir_path, output_name), codec= 'libx264', audio_codec='libvorbis')

if __name__ == "__main__":
    create_video(
        "export/2024-07-23-08-09-40",
        speed=1.3,
        fps=30,
        sub_alignment="right",
        sub_position_horizontal=0,
        sub_position_vertical=0)
