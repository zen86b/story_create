import os
import argparse
import time

from utils.convert_prompt import convert_prompt
from utils.voice_gen import tts
from utils.text_chunk import split_into_chunks
from utils.image_gen import image_generate

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, help="Path to input.")

args = parser.parse_args()
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())
output_dir = os.path.join("export", timestamp)
os.mkdir(output_dir)
os.mkdir(os.path.join(output_dir, "voice"))
os.mkdir(os.path.join(output_dir, "image"))

with open(args.input, "r") as f:
    content = f.read()

    text_chunks = split_into_chunks(content)

image_gen_prompt = convert_prompt(text_chunks)

tts(text_chunks, output_dir=os.path.join(output_dir, "voice"))
image_generate(image_gen_prompt, output_dir=os.path.join(output_dir, "image"))
