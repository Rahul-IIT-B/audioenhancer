import logging, shutil, subprocess, os, io, json, uuid, whisper, warnings, requests, boto3
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from moviepy import VideoFileClip
from pydub import AudioSegment
from datetime import timedelta
from dotenv import load_dotenv
import google.generativeai as genai
import boto3

warnings.filterwarnings("ignore", module="whisper")
warnings.filterwarnings("ignore", message=".*bytes read.*")
# Configure root logger once
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

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

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)
s3_bucket = os.getenv("S3_BUCKET")

def upload_file(local_path: str, unique_filename: str = None):
    try:
        s3.upload_file(
            Filename=local_path,
            Bucket=s3_bucket,
            Key=unique_filename,
        )
    except Exception as e:
        return {"file": unique_filename, "status": "error", "detail": str(e)}

    s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{unique_filename}"
    return {"file": unique_filename, "status": "success", "url": s3_url}

def download_file(file_name: str, path: str = None):
    try:
        s3.download_file(
            Bucket=s3_bucket,
            Key=file_name,
            Filename=path
        )
    except Exception as e:
        return {"file": file_name, "status": "error", "detail": str(e)}
    return {"file": file_name, "status": "success", "file_path": path}

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
        path = path.replace("\\", "/")
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
            "Refine the segments of the transcript below while retaining original authors words as much as possible."  
            "Change only if there are filler words, incomplete sentences or meaning, underdefined concepts.\n\n"
            "Output one line per segment as: mm:ss <refined text>\n\n"
        )

        # Prepare the rows for the spreadsheet
        current_t = 0
        rows = [["Start", "End", "New Start", "New End", "Original", "Refined", "Pause at end(sec)", "Audio Length", "Video Length", "Flag", "Video Speed flag", "Out of sync", "Additional comments"]]
        for seg in segments:
            if seg["start"] - current_t > 1:
                start_ts = format_timestamp(current_t)
                end_ts = format_timestamp(seg['start'])
                duration_sec = seg['start'] - current_t
                video_len = format_timestamp(duration_sec)
                rows.append([start_ts, end_ts, start_ts, end_ts, "", "", 0, video_len, video_len, "", "", "", ""])
            start_ts = format_timestamp(seg['start'])
            end_ts = format_timestamp(seg['end'])
            duration_sec = seg['end'] - seg['start']
            video_len = format_timestamp(duration_sec)
            prompt += f"{start_ts} {seg['text'].strip()}\n"
            rows.append([start_ts, end_ts, "", "", seg['text'].strip(), "", 0, "", video_len, "", "", "", ""])
            current_t = seg['end']
        duration = video.duration
        video.close()
        if current_t - duration > 1:
            start_ts = format_timestamp(current_t)
            end_ts = format_timestamp(duration)
            duration_sec = duration - current_t
            video_len = format_timestamp(duration_sec)
            rows.append([start_ts, end_ts, start_ts, end_ts, "", "", 0, video_len, video_len, "", "", "", ""])

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

        # Upload the video to S3
        print(upload_file(path, unique_filename=f"Original_videos/{filename}"))
        print(upload_file(meta_path, unique_filename=f"metadata/{sheet_id}.json"))

        os.remove(path)
        os.remove(raw_audio)
        os.remove(meta_path)
        
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
                final_seg
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

@app.post("/refresh-voiceover")
async def refresh_voiceover(sheetId: str):
    # load metadata
    meta_path = os.path.join("./Data/metadata", f"{sheetId}.json")
    os.makedirs("./Data/metadata", exist_ok=True)
    print(download_file(f"metadata/{sheetId}.json", path=meta_path))
    with open(meta_path) as m:
        meta = json.load(m)
    uid = meta['uuid']
    path = meta['original_video']

    os.makedirs("./Data/Original_videos", exist_ok=True)
    print(download_file(f"Original_videos/{path.split('/')[-1]}", path=path))

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
        audio_path = audio_path.replace("\\", "/")
        refined = row[5]

        if refined != "":
            audio_bytes = safe_tts_request(refined)
            audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format='mp3')
            audio.export(audio_path, format='wav')

            audio_seg = AudioSegment.from_file(audio_path, format="wav")
            audio_len_sec = audio_seg.duration_seconds
            audio_len_sec += float(row[6])

            if audio_len_sec > video_len * 1.02:
                flag = "Slowed-down"
            elif audio_len_sec < video_len * 0.98:
                flag = "Fast-forwarded"
            else:
                flag = "No change"

            factor = video_len / audio_len_sec if flag != "No change"  else 1.0

            new_end = new_start + audio_len_sec
            updated_rows[idx][3] = format_timestamp(new_end)
            if idx + 1 < len(updated_rows):
                updated_rows[idx + 1][2] = updated_rows[idx][3]
            updated_rows[idx][7] = format_timestamp(audio_len_sec)
            updated_rows[idx] = updated_rows[idx][:9] + [flag]
        else:
            factor = 1.0
            audio_path = None
            flag = "No change"
            new_end = new_start + video_len
            updated_rows[idx][3] = format_timestamp(new_end)
            if idx + 1 < len(updated_rows):
                updated_rows[idx + 1][2] = updated_rows[idx][3]
            updated_rows[idx] = updated_rows[idx][:9] + [flag]

        segments.append({
            "start": start,
            "end": end,
            "factor": factor,
            "audio_path": audio_path
        })

    # Update sheet
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheetId, range='A2:J',
        valueInputOption='RAW', body={'values': updated_rows}
    ).execute()

    final_video_path = path.replace("Original_videos", "Final_videos")
    os.makedirs("./Data/Final_videos", exist_ok=True)
    processed_path = process_segments_with_ffmpeg(segments, path, final_video_path)
    final_s3 = upload_file(processed_path, unique_filename=f"Final_videos/{processed_path.split('/')[-1]}")
    final_s3_url = final_s3['url']
    os.remove(path)
    os.remove(meta_path)
    os.remove(processed_path)
    shutil.rmtree(f"Data/tmp/{uid}", ignore_errors=True)

    return JSONResponse({"Final_s3_url": final_s3_url})
