from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import os
import io
import json
import uuid
import whisper
import google.auth
import streamlit as st
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from moviepy import VideoFileClip, AudioFileClip
from pydub import AudioSegment
import requests
import torchaudio
import torch
from datetime import timedelta
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Google Sheets setup
sa_info = st.secrets["GOOGLE_SA_JSON"]
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
sheets_service = build('sheets', 'v4', credentials=credentials)

# TTS config
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
MODEL_ID = "eleven_multilingual_v2"

app = FastAPI()
INPUT_DIR = "./uploads"
os.makedirs(INPUT_DIR, exist_ok=True)

# --- Helper functions ---
def format_timestamp(sec: float) -> str:
    td = timedelta(seconds=round(sec))
    m, s = divmod(td.seconds, 60)
    return f"{m:02d}:{s:02d}"

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()

def safe_tts_request(text: str) -> bytes:
    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={"xi-api-key": ELEVEN_KEY, "Content-Type": "application/json"},
        json={"text": text, "model_id": MODEL_ID},
        stream=True
    )
    resp.raise_for_status()
    return resp.content

# --- Endpoint 1: process-video ---
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    path = os.path.join(INPUT_DIR, filename)
    with open(path, "wb") as f:
        f.write(await file.read())

    video = VideoFileClip(path)
    raw_audio = path.replace('.mp4', '.wav')
    video.audio.write_audiofile(raw_audio, logger=None)
    video_no_audio = path.replace('.mp4', '_noaudio.mp4')
    video.without_audio().write_videofile(video_no_audio, codec="libx264", audio_codec="aac", logger=None)

    model = whisper.load_model("base")
    audio = whisper.load_audio(raw_audio).astype("float32")
    result = model.transcribe(audio, word_timestamps=True)
    segments = result['segments']

    # refine with Gemini
    prompt = (
        "Refine the segments of the transcript below into a formal, coherent, "
        "professional-tone version without any errors. Remove filler words but "
        "preserve each segment’s original length.\n\n"
        "Output one line per segment as: mm:ss <refined text>\n\n"
    )
    for seg in segments:
        ts = format_timestamp(seg['start'])
        prompt += f"{ts} {seg['text'].strip()}\n"
    refined_lines = call_gemini(prompt).splitlines()

    # write to Google Sheet
    rows = [["Start", "End", "Original", "Refined"]]
    for seg, line in zip(segments, refined_lines):
        _, text = line.split(' ', 1)
        rows.append([
            format_timestamp(seg['start']),
            format_timestamp(seg['end']),
            seg['text'].strip(),
            text
        ])

    sheet = sheets_service.spreadsheets().create(
        body={'properties': {'title': os.path.splitext(file.filename)[0]}},
        fields='spreadsheetId'
    ).execute()
    sheet_id = sheet['spreadsheetId']
    # Set permissions to allow anyone with the link to edit
    try:
        drive_service = build('drive', 'v3', credentials=credentials)
        drive_service.permissions().create(
            fileId=sheet_id,
            body={
                'type': 'anyone',
                'role': 'writer'
            },
            fields='id'
        ).execute()
    except HttpError as error:
        print(f"An error occurred: {error}")
        raise HTTPException(status_code=500, detail="Failed to set spreadsheet permissions")
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheet_id, range='A1', valueInputOption='RAW', body={'values': rows}
    ).execute()

    # save metadata for refresh
    meta = {
        'uuid': uid,
        'noaudio': video_no_audio,
        'segment_count': len(segments)
    }
    with open(os.path.join(INPUT_DIR, f"{sheet_id}_meta.json"), 'w') as m:
        import json
        json.dump(meta, m)

    return JSONResponse({"spreadsheetId": sheet_id})

# --- Endpoint: fetch-segments ---
@app.get("/fetch-segments")
async def fetch_segments(sheetId: str):
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:D'
        ).execute()
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    values = resp.get('values', [])
    segments = [
        {'start': row[0], 'end': row[1], 'original': row[2], 'refined': row[3]}
        for row in values
    ]
    return JSONResponse({"segments": segments})

# --- Endpoint: refresh-voiceover ---
class RefreshRequest(BaseModel):
    spreadsheetId: str
    changedIndices: list[int]

@app.post("/refresh-voiceover")
async def refresh_voiceover(req: RefreshRequest):
    sheetId = req.spreadsheetId
    # load metadata
    meta_path = os.path.join(INPUT_DIR, f"{sheetId}_meta.json")
    if not os.path.exists(meta_path):
        raise HTTPException(404, "Metadata not found")
    import json
    with open(meta_path) as m:
        meta = json.load(m)
    uuid = meta['uuid']
    video_no_audio = meta['noaudio']
    count = meta['segment_count']

    # fetch updated sheet
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:D'
        ).execute()
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    values = resp.get('values', [])

    # parse segments
    segments = []
    for row in values:
        start = float(row[0].split(':')[0]) * 60 + float(row[0].split(':')[1])
        end = float(row[1].split(':')[0]) * 60 + float(row[1].split(':')[1])
        segments.append({'start': start, 'end': end, 'refined': row[3]})

    # Determine which segments need to be synthesized
    to_synthesize = []
    for idx in range(count):
        seg_path = os.path.join(INPUT_DIR, f"{uuid}_seg_{idx}.wav")
        if not os.path.exists(seg_path) or idx in req.changedIndices:
            to_synthesize.append(idx)

    # Synthesize necessary segments
    for idx in to_synthesize:
        seg = segments[idx]
        audio_bytes = safe_tts_request(seg['refined'])
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format='mp3')
        audio.export(os.path.join(INPUT_DIR, f"seg_{idx}.wav"), format='wav')

    # concatenate all segments in order
    parts = []
    for i in range(count):
        wav_path = os.path.join(INPUT_DIR, f"seg_{i}.wav")
        wav, sr = torchaudio.load(wav_path)
        if sr != 24000:
            wav = torchaudio.transforms.Resample(sr, 24000)(wav)
        parts.append(wav)
    full = torch.cat(parts, dim=1)
    final_audio = os.path.join(INPUT_DIR, f"{sheetId}_refreshed.wav")
    torchaudio.save(final_audio, full, 24000)

    # merge and return
    clip = VideoFileClip(video_no_audio)
    audio_clip = AudioFileClip(final_audio)
    clip.with_audio(audio_clip).write_videofile(
        os.path.join(INPUT_DIR, f"{sheetId}_final.mp4"),
        codec="libx264", audio_codec="aac", logger=None
    )
    return FileResponse(os.path.join(INPUT_DIR, f"{sheetId}_final.mp4"))
