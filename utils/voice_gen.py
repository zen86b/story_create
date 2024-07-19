import scipy
from transformers import AutoProcessor, BarkModel

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to("cuda")


def tts(text, output_name):
    text_prompt = f"""
        {text}
    """

    inputs = processor(text_prompt, voice_preset="v2/en_speaker_6")

    audio_array = model.generate(**inputs.to("cuda"))
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    scipy.io.wavfile.write(f"{output_name}.wav", rate=sample_rate, data=audio_array)

if __name__ == "__main__":
    tts("But for now, Lily was content to simply enjoy the company of her new friend.","output")