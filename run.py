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
parser.add_argument("--input", type=str,required=True, help="Path to input file which contain the paragraph")
parser.add_argument("--height", type=int, default=720, help="Height of generated image")
parser.add_argument("--width", type=int, default=1280, help="Width of generated image")
parser.add_argument("--voice-preset", type=str, default="en_speaker_6", help="Code of voice preset using to generate voice")
parser.add_argument("--create-video", action="store_true", default=False, help="Enable create video after generation")
parser.add_argument("--video-dim-h", type=int, default=1080, help="Height of output video")
parser.add_argument("--video-dim-w", type=int, default=1920, help="Width of output video")
parser.add_argument("--fps", type=float, default=60, help="Fps of video")
parser.add_argument("--speed", type=float, default=1.0, help="Speed of video")
parser.add_argument("--font-type", type=str, default="", help="Path to the font using to write subtitle")
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

print("*** Start text chunk ***")
# Text chunk
text_chunks = split_into_chunks(content)
with open(os.path.join(output_dir, "text_chunk.json"), "w") as f:
    json_str = json.dump({"text_chunks": text_chunks}, f, indent=4)
print("*** Finish text chunk ***")

print("*** Start text transform ***")
# Transform to prompt
image_gen_prompt = convert_prompt(text_chunks)
with open(os.path.join(output_dir, "image_gen_prompt.json"), "w") as f:
    json_str = json.dump({"image_gen_prompts": text_chunks}, f, indent=4)
print("*** Finish text transform ***")

print("*** Start image generate ***")
# Image gen
image_generate(
    image_gen_prompt,
    negative_prompt=NEG_PROMPT,
    height=args.height,
    width=args.width,
    output_dir=os.path.join(output_dir, "image"),
)
print("*** Finish image generate ***")

print("*** Start voice generate ***")
# Text to speech
tts(
    text_chunks,
    voice_preset=args.voice_preset,
    output_dir=os.path.join(output_dir, "voice")
)
print("*** Finish voice generate ***")

print("*** Start create video ***")
# Create video
if args.create_video:
    create_video(
        output_dir,
        video_dim_w=args.video_dim_w,
        video_dim_h=args.video_dim_h,
        fps=args.fps,
        speed=args.speed,
        font_type=args.font_type,
        sub_position_vertical=args.sub_position_vertical,
        sub_position_horizontal=args.sub_position_horizontal,
        sub_alignment=args.sub_alignment,
        sub_color=args.sub_color
    )
print("*** Finish create video ***")
print(f"Result is saved in {output_dir}")
