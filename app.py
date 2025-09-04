from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import tempfile

app = FastAPI()

# Use Hugging Face hosted model
pipe = pipeline(
    "automatic-speech-recognition",
    model="Arsalan07/whisper-api",  # Hugging Face repo
    device=-1  # CPU, no GPU required
)

@app.get("/")
def root():
    return {"message": "Whisper API running"}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = pipe(tmp_path)
    return {"text": result["text"]}
