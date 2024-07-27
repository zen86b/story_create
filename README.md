# Installation

Install `llama-cpp-python`, change version of `llama-cpp-python` to fix with `CUDA` version on machine

```bash
pip install --no-cache-dir llama-cpp-python==0.2.82 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

Install other lib

```bash
pip install -r requirement.txt
apt-get install libgl1
```

Dowload model

- `Mistral-7b` for generate prompt
- `Dreamshaper-XL-turbo` + `add-detail-xl` (LoRA) for image generation
- `suno/bark` for TTS, [`HERE`](https://suno-ai.notion.site/8b8e8749ed514b0cbf3f699013548683?v=bc67cff786b04b50b3ceb756fd05f68c) is sample voice of `suno/bark`

```bash
mkdir model
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -P model/
gdown https://drive.google.com/uc\?export\=download\&confirm\=yTib\&id\=1HaYMgeQBRyEblJscALoE2lOCsQwaJ5Oz -O model/
```

# Usage

`tmp.txt` contains a long paragraph of text

Running full pipeline (chunk text, gen voices, gen images, create video)

```bash
python3 run.py --input tmp.txt --create-video
```

Running generate only

```bash
python3 run.py --input tmp.txt
```

More option

```bash
python3 run.py -h
```

Example:

```bash
python3 run.py \
--input tmp.txt --create-video \
--height 720 --width 1280 \
--voice-preset es_speaker_0 \
--video-dim-h 1080 \
--video-dim-w 1920 \
--fps 60 \
--speed 1.2 \
--sub-position-vertical 0.9 \
--sub-position-horizontal 0.5 \
--sub-alignment mid \
--sub-color yellow
```

After generation, you can re-create video with other option

```bash
python3 render_video.py \
--input export/2024-01-01-09:00:00 \
--video-dim-h 1080 \
--video-dim-w 1920 \
--fps 60 \
--speed 1.2 \
--sub-position-vertical 0.9 \
--sub-position-horizontal 0.5 \
--sub-alignment mid \
--sub-color yellow
```

# Docs

- First, `utils.text_chunk.split_into_chunks` split a long paragraph of text into small chunks.
- `utils.convert_prompt.convert_prompt` turns each chunk to a image decription which use to generate image in `utils.image_gen.image_generate`
- Each chunk also feed into `utils.voice_gen.tts` to generate voice
- All voice fragment and images are saved in `export/`, using `utils.video_create.create_video` to combine all of theme to a full video with voice and transcription
