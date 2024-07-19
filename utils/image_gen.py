import random
import time

import torch
from compel import Compel, ReturnedEmbeddingsType
from diffusers import StableDiffusionXLPipeline

model_path = "model/SDXL_model_fp16/dreamshaperXL"
lora_weight = "model/add-detail-xl.safetensors"


def image_generate(
    prompt,
    negative_prompt=None,
    height=1024,
    width=1024,
    num_inference_steps=10,
    guidance_scale=2,
    num_images_per_prompt=1,
    seed=-1,
    output=""
):
    start_time = time.time()
    if seed != -1:
        generator = torch.Generator(device="cuda").manual_seed(seed)
    else:
        random_seed = round(random.random() * pow(10, random.randint(1, 15)))
        generator = torch.Generator(device="cuda").manual_seed(random_seed)

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16
    ).to("cuda")

    if lora_weight != "":
        pipe.load_lora_weights(lora_weight)
        lora_opt = {"cross_attention_kwargs": {"scale": 1}}

    with torch.no_grad():
        compel_proc = Compel(
            tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
            text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )
        prompt_embeds, pooled_prompt_embeds = compel_proc(prompt)
        if negative_prompt is not None:
            negative_prompt_embeds, negative_pooled_prompt_embeds = compel_proc(
                negative_prompt
            )
        else:
            negative_prompt_embeds, negative_pooled_prompt_embeds = None, None

    results = pipe(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_images_per_prompt,
        guidance_scale=guidance_scale,
        generator=generator,
        **lora_opt,
    )

    images = results.images
    nsfw_content_detected = None

    names = []

    name = f"export/text2img_XL{output}.png"
    names.append(name)
    images[0].save(name)
    return {"image": names, "execution_time": round(time.time() - start_time, 2)}

if __name__ == "__main__":
    image_generate(prompt="A bustling cityscape in 2042: Robots of various sizes and shapes work alongside humans, constructing buildings, serving customers at cafes, and maintaining lush gardens. The sun sets over the horizon as the symbiotic duo continues their daily tasks, symbolizing human-robot coexistence and collaboration.")