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
parser.add_argument("--input", type=str, help="Path to input.")

args = parser.parse_args()
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
output_dir = os.path.join("export", timestamp)
if not os.path.exists("export"):
    os.mkdir("export")
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, "voice"))
os.mkdir(os.path.join(output_dir, "image"))

with open(args.input, "r") as f:
    content = f.read()

text_chunks = split_into_chunks(content)
with open(os.path.join(output_dir, "text_chunk.json"), "w") as f:
    json_str = json.dump({"text_chunks": text_chunks}, f, indent=4)

image_gen_prompt = convert_prompt(text_chunks)
with open(os.path.join(output_dir, "image_gen_prompt.json"), "w") as f:
    json_str = json.dump({"image_gen_prompts": text_chunks}, f, indent=4)
    
image_generate(
    image_gen_prompt,
    negative_prompt=NEG_PROMPT,
    output_dir=os.path.join(output_dir, "image"),
)
tts(text_chunks, output_dir=os.path.join(output_dir, "voice"))

create_video(output_dir)

print(f"Result is saved in {output_dir}")
