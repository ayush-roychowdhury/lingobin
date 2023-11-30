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
    current_language = list(segments_dict.values())[0]  # Language of the first segment

    for end_time, language in segments_dict.items():
        if language != current_language:
            # Process the accumulated segment in the current language
            start_ms = start_time
            end_ms = end_time - 1  # Adjusting to include the last moment of the previous language

            audio_segment = full_audio[start_ms:end_ms]
            temp_file = "temp_segment.wav"
            audio = whisper.load_audio(temp_file)
            audio = whisper.pad_or_trim(audio)

            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            options = whisper.DecodingOptions(task='transcribe', fp16=False)
            result = whisper.decode(model, mel, options)
            translated_texts.append(result['text'])

            os.remove(temp_file)

            # Update the start time and current language
            start_time = end_time
            current_language = language

    # Process the final segment
    start_ms = start_time
    end_ms = len(full_audio)  # Till the end of the audio
    audio_segment = full_audio[start_ms:end_ms]
    temp_file = "temp_segment.wav"
    audio = whisper.load_audio(temp_file)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    options = whisper.DecodingOptions(task='transcribe', fp16=False)
    result = whisper.decode(model, mel, options)
    translated_texts.append(result['text'])
    translated_texts.append(result['text'])
    os.remove(temp_file)

    full_translation = ' '.join(translated_texts)
    return full_translation