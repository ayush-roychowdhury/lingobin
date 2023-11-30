import os
import whisper
from pydub import AudioSegment
import datasets
import pandas as pd

def translate_audio(audio_path, segments_dict, model_name="base"):
    model = whisper.load_model(model_name)
    full_audio = AudioSegment.from_file(audio_path, format="wav")

    translated_texts = []
    start_time = 0  # Start of the first segment

    for end_time, language in segments_dict.items():
        start_ms = start_time * 1000
        end_ms = end_time * 1000

        audio_segment = full_audio[start_ms:end_ms]
        temp_file = "temp_segment.wav"
        audio_segment.export(temp_file, format="wav")

        result = model.transcribe(temp_file, task="translate", language=language)
        translated_texts.append(result['text'])

        os.remove(temp_file)
        start_time = end_time

    full_translation = ' '.join(translated_texts)
    return full_translation

ds = datasets.load_dataset("CAiRE/ASCEND")
train = pd.DataFrame(ds["train"])
my_wav = train.at[9606, 'path']
audio = AudioSegment.from_wav(my_wav)
audio.export("test_audio.wav", format="wav")
audio_path = "test_audio.wav"
audio_path = "test_audio.wav"
segments_dict = {3: 'zh', 6: 'en', 8: 'zh', 9: 'en', 10: 'zh'}
translation = translate_audio(audio_path, segments_dict)
print(translation)