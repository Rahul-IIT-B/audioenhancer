import logging
import subprocess

# Configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import io
import json
import uuid
import whisper
import warnings
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from moviepy import VideoFileClip
from pydub import AudioSegment
import requests
from datetime import timedelta
from dotenv import load_dotenv
import google.generativeai as genai
warnings.filterwarnings("ignore", module="whisper")
warnings.filterwarnings("ignore", message=".*bytes read.*")

load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_GEMINI_API_KEY"))

# Google Sheets setup
sa_info = json.loads(os.getenv("GOOGLE_SA_JSON"))
SCOPES = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
credentials = Credentials.from_service_account_info(sa_info, scopes=SCOPES)
sheets_service = build('sheets', 'v4', credentials=credentials, cache_discovery=False)

# TTS config
ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY")
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
MODEL_ID = "eleven_multilingual_v2"

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helper functions ---
def format_timestamp(sec: float) -> str:
    td = timedelta(seconds=sec)
    m, s = divmod(td.seconds, 60)
    ms = int((sec - int(sec)) * 1000)
    return f"{m:02d}:{s:02d}:{ms:03d}"

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
    try: 
        uid = str(uuid.uuid4())
        filename = file.filename.replace('.mp4', f"_{uid}.mp4")
        uploads_dir = "./Data/Original_videos"
        os.makedirs(uploads_dir, exist_ok=True)
        path = os.path.join(uploads_dir, filename)
        path = os.path.abspath(path).replace("\\", "/")
        with open(path, "wb") as f:
            f.write(await file.read())
        
        # Extract audio from video
        video = VideoFileClip(path)
        audio_name = os.path.splitext(os.path.basename(path))[0] + ".wav"
        raw_audio = os.path.join("Data/Extracted_Audio", audio_name)
        os.makedirs("Data/Extracted_Audio", exist_ok=True)
        video.audio.write_audiofile(raw_audio, logger=None)
        
        # Transcribe audio
        model = whisper.load_model("turbo")
        audio = whisper.load_audio(raw_audio).astype("float32")
        result = model.transcribe(audio, word_timestamps=True)
        segments = result['segments']
        
        prompt = (
            "Refine the segments of the transcript below into a formal, coherent, "
            "professional-tone version without any errors. Remove filler words but "
            "preserve each segment’s original length.\n\n"
            "Output one line per segment as: mm:ss <refined text>\n\n"
        )

        # Prepare the rows for the spreadsheet
        current_t = 0
        rows = [["Start", "End", "New Start", "New End", "Original", "Refined", "Audio Length", "Video Length", "Flag"]]
        for seg in segments:
            if seg["start"] - current_t > 1:
                start_ts = format_timestamp(current_t)
                end_ts = format_timestamp(seg['start'])
                duration_sec = seg['start'] - current_t
                video_len = format_timestamp(duration_sec)
                rows.append([start_ts, end_ts, start_ts, end_ts, "", "", video_len, video_len, ""])
            start_ts = format_timestamp(seg['start'])
            end_ts = format_timestamp(seg['end'])
            duration_sec = seg['end'] - seg['start']
            video_len = format_timestamp(duration_sec)
            prompt += f"{start_ts} {seg['text'].strip()}\n"
            rows.append([start_ts, end_ts, "", "", seg['text'].strip(), "", "", video_len, ""])
            current_t = seg['end']
        duration = VideoFileClip(path).duration
        if current_t - duration > 1:
            start_ts = format_timestamp(current_t)
            end_ts = format_timestamp(duration)
            duration_sec = duration - current_t
            video_len = format_timestamp(duration_sec)
            rows.append([start_ts, end_ts, start_ts, end_ts, "", "", video_len, video_len, ""])

        refined_lines = call_gemini(prompt).splitlines()

        # Update the rows with refined text
        i = 1
        for line in refined_lines:
            if rows[i][4] == "":
                i += 1
            _, text = line.split(' ', 1)
            rows[i][5] = text
            i += 1
        
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
            'original_video': path,
            'audio_path': raw_audio,
            'spreadsheet_id': sheet_id,
            'segment_count': len(rows) - 1
        }
        meta_path = os.path.join("./Data/metadata", f"{sheet_id}.json")
        os.makedirs("./Data/metadata", exist_ok=True)
        with open(meta_path, 'w') as m:
            json.dump(meta, m)
        
        return JSONResponse({"spreadsheetId": sheet_id})
    except Exception as e:
        # This logs the full traceback
        logger.exception("Error in process_video endpoint")
        # Optionally capture more context
        logger.error(f"Uploaded filename: {file.filename}, temp path: {path}")
        # Return a generic 500 to the client
        raise HTTPException(
            status_code=500,
            detail="Internal server error during video processing."
        )
        
# --- Endpoint: fetch-segments ---
@app.get("/fetch-segments")
async def fetch_segments(sheetId: str):
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:H'
        ).execute()
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    values = resp.get('values', [])
    segments = [
        {'start': row[0], 'end': row[1], 'original': row[4], 'refined': row[5]}
        for row in values
    ]
    return JSONResponse({"segments": segments})

def process_segments_with_ffmpeg(segments, input_path, output_path):
    uid = input_path.split('/')[-1].split('_')[-1].split('.')[0]
    tmp_base = os.path.join("Data/tmp", uid)
    tmp_raw = os.path.join(tmp_base, "raw")
    tmp_sped = os.path.join(tmp_base, "sped")
    tmp_final = os.path.join(tmp_base, "final")
    os.makedirs(tmp_raw, exist_ok=True)
    os.makedirs(tmp_sped, exist_ok=True)
    os.makedirs(tmp_final, exist_ok=True)
    segment_files = []

    for i, seg in enumerate(segments):
        raw_seg = os.path.join(tmp_raw, f"raw_{i}.mp4")
        sped_seg = os.path.join(tmp_sped, f"sped_{i}.mp4")
        final_seg = os.path.join(tmp_final, f"final_{i}.mp4")
        segment_files.append(final_seg)

        # 1. Extract segment (no audio)
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path,
            "-ss", str(seg["start"]), "-to", str(seg["end"]),
            "-an", "-preset", "ultrafast", "-crf", "23",
            "-c:v", "libx264", "-force_key_frames", "expr:gte(t,n_forced*2)",
            raw_seg
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 2. Speed up/down if needed
        if seg["factor"] != 1.0:
            setpts = f"{1 / seg['factor']}*PTS"
            subprocess.run([
                "ffmpeg", "-y", "-i", raw_seg,
                "-filter:v", f"setpts={setpts}",
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                sped_seg
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            sped_seg = raw_seg

        # 3. Add or replace audio
        if seg["audio_path"]:
            subprocess.run([
                "ffmpeg", "-y", "-i", sped_seg, "-i", seg["audio_path"],
                "-map", "0:v", "-map", "1:a",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
                "-shortest", final_seg
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # Add silent audio if missing
            duration_cmd = [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", sped_seg
            ]
            duration = subprocess.check_output(duration_cmd, text=True).strip()
            subprocess.run([
                "ffmpeg", "-y", "-i", sped_seg,
                "-f", "lavfi", "-t", duration,
                "-i", "anullsrc=r=48000:cl=stereo",
                "-shortest", "-c:v", "copy", "-c:a", "aac", final_seg
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Concatenate segments with guaranteed audio
    concat_command = ["ffmpeg", "-y"]
    filter_inputs = ""
    for i, seg in enumerate(segment_files):
        concat_command += ["-i", seg]
        filter_inputs += f"[{i}:v][{i}:a]"

    filter_str = f"{filter_inputs}concat=n={len(segment_files)}:v=1:a=1[outv][outa]"

    concat_command += [
        "-filter_complex", filter_str,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", output_path
    ]

    subprocess.run(concat_command)
    return output_path

# --- Endpoint: refresh-voiceover ---
class RefreshRequest(BaseModel):
    spreadsheetId: str
    changedIndices: list[int]

@app.post("/refresh-voiceover")
async def refresh_voiceover(req: RefreshRequest):
    sheetId = req.spreadsheetId
    # load metadata
    meta_path = os.path.join("./Data/metadata", f"{sheetId}.json")
    if not os.path.exists(meta_path):
        raise HTTPException(404, "Metadata not found")
    with open(meta_path) as m:
        meta = json.load(m)
    uid = meta['uuid']
    path = meta['original_video']
    count = meta['segment_count']

    tmp_base = os.path.join("Data/tmp", uid)
    cloned_path = os.path.join(tmp_base, "cloned")
    os.makedirs(tmp_base, exist_ok=True)
    os.makedirs(cloned_path, exist_ok=True)

    # fetch updated sheet
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:I'
        ).execute()
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    values = resp.get('values', [])
    
    # process segments
    updated_rows = values.copy()
    updated_rows[0][2] = format_timestamp(0)
    segments = []
    for idx, row in enumerate(updated_rows):
        start = sum(x * float(t) for x, t in zip([60, 1, 0.001], row[0].split(':')))
        end = sum(x * float(t) for x, t in zip([60, 1, 0.001], row[1].split(':')))
        new_start = sum(x * float(t) for x, t in zip([60, 1, 0.001], row[2].split(':')))
        video_len = end - start

        audio_path = os.path.join(cloned_path, f"seg_{idx}.wav")
        audio_path = os.path.abspath(audio_path).replace("\\", "/")
        refined = row[5]

        if refined != "":
            if not os.path.exists(audio_path) or idx in req.changedIndices:
                audio_bytes = safe_tts_request(refined)
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format='mp3')
                audio.export(audio_path, format='wav')

            audio_seg = AudioSegment.from_file(audio_path, format="wav")
            audio_len_sec = audio_seg.duration_seconds

            if audio_len_sec > video_len * 1.02:
                flag = "Slowed-down"
            elif audio_len_sec < video_len * 0.98:
                flag = "Fast-forwarded"
            else:
                flag = "No change"

            factor = video_len / audio_len_sec if flag == "Slowed-down" else (
                audio_len_sec / video_len if flag == "Fast-forwarded" else 1.0
            )

            new_end = new_start + audio_len_sec
            updated_rows[idx][3] = format_timestamp(new_end)
            if idx + 1 < len(updated_rows):
                updated_rows[idx + 1][2] = updated_rows[idx][3]
            updated_rows[idx][6] = format_timestamp(audio_len_sec)
            updated_rows[idx] = updated_rows[idx][:8] + [flag]
        else:
            factor = 1.0
            audio_path = None
            flag = "No change"
            new_end = end
            updated_rows[idx][3] = format_timestamp(new_end)
            if idx + 1 < len(updated_rows):
                updated_rows[idx + 1][2] = updated_rows[idx][3]
            updated_rows[idx] = updated_rows[idx][:8] + [flag]

        segments.append({
            "start": start,
            "end": end,
            "factor": factor,
            "audio_path": audio_path
        })

    # Update sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheetId, range='A2:I',
        valueInputOption='RAW', body={'values': updated_rows}
    ).execute()

    final_video_path = path.replace("Original_videos", "Final_videos")
    os.makedirs("./Data/Final_videos", exist_ok=True)
    processed_path = process_segments_with_ffmpeg(segments, path, final_video_path)

    return JSONResponse({"Final_video_path": processed_path})

@app.get("/stream_video")
async def stream_video(request: Request, path: str):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")

    def iterfile(start: int = 0, end: int = file_size - 1):
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk_size = 4096 if remaining >= 4096 else remaining
                data = f.read(chunk_size)
                if not data:
                    break
                yield data
                remaining -= len(data)

    if range_header:
        # Example: "bytes=0-1023"
        start_str, end_str = range_header.strip().split("=")[-1].split("-")
        start = int(start_str) if start_str else 0
        end = int(end_str) if end_str else file_size - 1
        content_length = end - start + 1
        return StreamingResponse(
            iterfile(start, end),
            media_type="video/mp4",
            status_code=206,
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
                "Content-Disposition": f"inline; filename={os.path.basename(path)}"
            }
        )

    # No range header – send whole file
    return StreamingResponse(
        iterfile(),
        media_type="video/mp4",
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f"inline; filename={os.path.basename(path)}"
        }
    )
