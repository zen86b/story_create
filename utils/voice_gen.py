import scipy
from transformers import AutoProcessor, BarkModel
import os
import torch
import numpy as np

def tts(texts, voice_preset, output_dir):
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark", torch_dtype=torch.float16).to("cuda")

    names = []
    for idx, text in enumerate(texts):
        text_prompt = f"""
            {text}
        """

        inputs = processor(text_prompt, voice_preset=f"v2/{voice_preset}")

        audio_array = model.generate(**inputs.to("cuda"))
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = model.generation_config.sample_rate
        name = os.path.join(output_dir, f"{idx}.wav")
        scipy.io.wavfile.write(name, rate=sample_rate, data=audio_array.astype(np.float32))
        names.append(name)
    
    return names

if __name__ == "__main__":
    tts(["this is a voice of a man"],"")