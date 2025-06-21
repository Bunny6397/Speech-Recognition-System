import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Load pre-trained model and processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

def transcribe_wav2vec(audio_path):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = transform(waveform)

    # Tokenize input
    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values

    # Get logits & decode
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    print("Transcription:", transcription)

# Example usage
transcribe_wav2vec("sample.wav")
