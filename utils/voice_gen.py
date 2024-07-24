import scipy
from transformers import AutoProcessor, BarkModel
import os


def tts(texts, output_dir):
    processor = AutoProcessor.from_pretrained("suno/bark")
    model = BarkModel.from_pretrained("suno/bark").to("cuda")

    names = []
    for idx, text in enumerate(texts):
        text_prompt = f"""
            {text}
        """

        inputs = processor(text_prompt, voice_preset="v2/en_speaker_6")

        audio_array = model.generate(**inputs.to("cuda"))
        audio_array = audio_array.cpu().numpy().squeeze()
        sample_rate = model.generation_config.sample_rate
        name = os.path.join(output_dir, f"{idx}.wav")
        scipy.io.wavfile.write(name, rate=sample_rate, data=audio_array)
        names.append(name)
    
    return names

if __name__ == "__main__":
    tts(["But for now, Lily was content to simply enjoy the company of her new friend."],"test_voice.wav")