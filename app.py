from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline
import torchaudio
import tempfile

# Load model + processor directly from Hugging Face
MODEL_ID = "Arsalan07/whisper-api"

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForSpeechSeq2Seq.from_pretrained(MODEL_ID)
pipe = pipeline("automatic-speech-recognition", model=model, tokenizer=processor)

app = FastAPI(title="Whisper API", description="Fine-tuned Whisper ASR API", version="1.0")

# Allow CORS (so HR can call API from anywhere)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    """Upload an audio file (.mp3/.wav) and get transcription"""
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Load audio
    speech_array, sampling_rate = torchaudio.load(tmp_path)

    # Run pipeline
    result = pipe(speech_array.squeeze().numpy(), sampling_rate=sampling_rate)

    return {"transcription": result["text"]}
