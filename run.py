import os
import argparse
import time
import json

from utils.convert_prompt import convert_prompt
from utils.voice_gen import tts
from utils.text_chunk import split_into_chunks
from utils.image_gen import image_generate
from utils.video_create import create_video

NEG_PROMPT = "(worst quality)+, (low quality)++, (normal quality)+, lowres, bad anatomy, bad hands, normal quality, bad eyes"

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str,dest="input_file",required=True, help="Path to input file which contain the paragraph")
parser.add_argument("--height", type=int, default=1080, help="Height of generated image")
parser.add_argument("--width", type=int, default=1080, help="Height of generated image")
parser.add_argument("--voice-preset", type=str, default="en_speaker_6", help="Code of voice preset using to generate voice")
parser.add_argument("--create-video", action="store_true", default=True, help="Enable create video after generation")
parser.add_argument("--fps", type=float, default=60, help="Fps of video")
parser.add_argument("--speed", type=float, default=1.0, help="Speed of video")
parser.add_argument("--sub-position-vertical", type=float, default=1.0, help="Determine the specific vertical location of the subtitles")
parser.add_argument("--sub-position-horizontal", type=float, default=0.5, help="Determine the specific horizontal location of the subtitles")
parser.add_argument("--sub-alignment", choices=["left", "mid", "right"], default="mid", help="Determine the specific alignment of the subtitles")
parser.add_argument("--sub-color", choices=["white","yellow"], default="yellow", help="Determine the specific color of the subtitles")
args = parser.parse_args()

timestamp = time.strftime("%Y-%m-%d-%H:%M:%S", time.gmtime())
output_dir = os.path.join("export", timestamp)
if not os.path.exists("export"):
    os.mkdir("export")
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, "voice"))
os.mkdir(os.path.join(output_dir, "image"))

with open(args.input, "r") as f:
    content = f.read()

# Text chunk
text_chunks = split_into_chunks(content)
with open(os.path.join(output_dir, "text_chunk.json"), "w") as f:
    json_str = json.dump({"text_chunks": text_chunks}, f, indent=4)

# Transform to prompt
image_gen_prompt = convert_prompt(text_chunks)
with open(os.path.join(output_dir, "image_gen_prompt.json"), "w") as f:
    json_str = json.dump({"image_gen_prompts": text_chunks}, f, indent=4)

# Image gen
image_generate(
    image_gen_prompt,
    negative_prompt=NEG_PROMPT,
    height=args.height,
    width=args.width,
    output_dir=os.path.join(output_dir, "image"),
)

# Text to speech
tts(
    text_chunks,
    voice_preset=args.voice_preset,
    output_dir=os.path.join(output_dir, "voice")
)

# Create video
if args.create_video:
    create_video(
        output_dir,
        fps=args.fps,
        speed=args.speed,
        sub_position_vertical=args.sub_position_vertical,
        sub_position_horizontal=args.sub_position_horizontal,
        sub_alignment=args.sub_alignment,
        sub_color=args.sub_color
    )

print(f"Result is saved in {output_dir}")
