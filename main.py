from flask import Flask, request, jsonify
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Config
import torch
import torchaudio
from pydub import AudioSegment

from evaluate import load
from math import ceil


# Load the model and processor
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

# # Save the model state dictionary
# torch.save(model.state_dict(), './wav2vec2-base-960h.pt')

# # Save the processor if you also need to preprocess data similarly later
# processor.save_pretrained("wav2vec2_processor")

# Create a configuration object for Wav2Vec2
config = Wav2Vec2Config()

# Instantiate the model using the configuration
model = Wav2Vec2ForCTC(config)

# Load your custom model weights
model.load_state_dict(torch.load('./wav2vec2-base-960h.pt', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model.eval()

def resample_audio(audio_filepath, target_sampling_rate=16000):
    audio = AudioSegment.from_mp3(audio_filepath)
    audio = audio.set_frame_rate(target_sampling_rate)
    audio.export(audio_filepath, format="mp3")


# Load the processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

def speech_file_to_array_fn(audio_filepath):
    speech_array, sampling_rate = torchaudio.load(audio_filepath)
    return speech_array.squeeze(0).numpy(), sampling_rate


class Response:
    def __init__(self, compute_cer, accuracy,transcription):
        self.compute_cer = compute_cer
        self.accuracy = accuracy
        self.transcription =transcription 

app = Flask(__name__)

 
def predict(audio_data, sampling_rate, text_input):
    inputs = processor(audio_data, sampling_rate=sampling_rate, return_tensors="pt", padding=True)

    input_values = inputs.input_values

    # Check if attention_mask is provided
    if "attention_mask" in inputs:
        attention_mask = inputs.attention_mask
        logits = model(input_values, attention_mask=attention_mask).logits
    else:
        logits = model(input_values).logits

    with torch.no_grad():
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)
    
    reference = [text_input.upper()]

    # Load the CER metric
    cer_metric = load("cer")

    reference = str(reference)  # Replace with the actual ground truth text
    transc = str(transcription) # Replace with your model's predicted text

    # Compute CER
    cer = cer_metric.compute(references=[reference], predictions=[transc])
    prc = ceil(cer * 100)
    print("CER:", prc, "%")

    # Calculate Accuracy
    acc = 100 - prc
    print("Accuracy:", acc, "%")

    res = Response(prc, acc,transcription)
    
    return res


@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Check if the POST request has the file part
    if 'voice' not in request.files:
        return jsonify({'error': 'No voice file provided'}), 400

    voice_file = request.files['voice']
    text_input = request.form.get('text')


    # Check if the file is empty
    if voice_file.filename == '' and not text_input:
        return jsonify({'error': 'Empty voice file and no text input provided'}), 400

    # Check if the file is a valid audio file
    if voice_file:
        try:
            audio_data, sampling_rate = torchaudio.load(voice_file)
            res = predict(audio_data.squeeze().numpy(), sampling_rate, text_input)
            
            return jsonify({'status':'ok','message':'processing the audio file success','data': {'compute_cer':res.compute_cer ,'accuracy':res.accuracy,'transcription':res.transcription[0]} }),200
        except Exception as e:
            return jsonify({'error': f'Error processing the audio file: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
