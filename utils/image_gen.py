import random
import time
import os

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import FluxPipeline

model_path = "black-forest-labs/FLUX.1-dev"
lora_weight = ""


def image_generate(
    prompts,
    negative_prompt=None,
    height=1280,
    width=720,
    num_inference_steps=5,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    seed=-1,
    output_dir=""
):
    start_time = time.time()
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        random_seed = round(random.random() * pow(10, random.randint(1, 15)))
        generator = torch.Generator(device="cuda").manual_seed(random_seed)

    pipe = FluxPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16, token="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"
    ).to("cuda")

    names = []

    for idx, prompt in enumerate(prompts):
        results = pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        images = results.images

        name = os.path.join(output_dir, f"{idx}.png")
        names.append(name)
        images[0].save(name)
    pipe.to("cpu")
    return {"image": names, "execution_time": round(time.time() - start_time, 2)}

if __name__ == "__main__":
    image_generate(prompt=["A bustling cityscape in 2042: Robots of various sizes and shapes work alongside humans, constructing buildings, serving customers at cafes, and maintaining lush gardens. The sun sets over the horizon as the symbiotic duo continues their daily tasks, symbolizing human-robot coexistence and collaboration."])
