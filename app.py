from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import tempfile
import logging

# Logging for debugging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Whisper ASR API")

# Load Hugging Face Whisper model
try:
    logging.info("Loading model from Hugging Face...")
    pipe = pipeline("automatic-speech-recognition", model="Arsalan07/whisper-api")
    logging.info("Model loaded successfully!")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    pipe = None

@app.get("/")
def root():
    return {"message": "✅ Whisper API is running on Render"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if pipe is None:
        return {"error": "Model not loaded"}
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    
    try:
        result = pipe(tmp_path)
        return {"text": result["text"]}
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        return {"error": str(e)}
