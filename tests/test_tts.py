
from pocket_tts import TTSModel
import scipy.io.wavfile
import numpy as np

print("Loading Indonesian Pocket TTS...")

model = TTSModel.load_model(
    config=(
        "hf://anak10thn/pocket-tts-indonesian/"
        "indonesian_6l.yaml@17257664e384561c957b02ac92edd1a24807f0e5"
    )
)

print("Model berhasil dimuat!")

# Gunakan rekaman suara sendiri sebagai voice prompt.
voice_state = model.get_state_for_audio_prompt(
    "voices/voice_1.wav"
)

text = "Halo, saya Hivi. Senang bisa berbicara dengan Anda."

print("Generating audio...")

audio = model.generate_audio(
    voice_state,
    text
)

audio_np = audio.detach().cpu().numpy()

scipy.io.wavfile.write(
    "test_hivi.wav",
    model.sample_rate,
    audio_np
)

print("Audio berhasil dibuat: test_hivi.wav")
print(f"Sample rate: {model.sample_rate} Hz")
print(f"Duration: {len(audio_np) / model.sample_rate:.2f} seconds")
