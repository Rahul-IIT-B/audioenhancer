from itertools import chain
import time, logging, shutil, subprocess, os, json, uuid, warnings, boto3, pandas, requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, FileSource
import google.generativeai as genai
import torch
from voice_clone.src.chatterbox.tts import tts_generate_segment, get_tts_model
import concurrent.futures
import multiprocessing

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

def call_gemini(prompt: str) -> str:
    model = genai.GenerativeModel("gemini-2.0-flash")
    return model.generate_content(prompt).text.strip()

def get_audio_duration(audio_path):
    """
    Returns the duration (in seconds) of an audio file using ffprobe.
    """
    try:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of",
            "default=noprint_wrappers=1:nokey=1", audio_path
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip())
        return duration
    except Exception as e:
        logger.error(f"Failed to get audio duration for {audio_path}: {e}")
        return 0.0

def extract_audio_with_ffmpeg(video_path: str, audio_path: str):
    """
    Extracts the audio track from a video into a WAV file using ffmpeg.
    This is much faster than MoviePy’s Python-level extraction.
    """
    # -y: overwrite output
    # -i: input file
    # -vn: no video
    # -acodec pcm_s16le: uncompressed PCM 16-bit little-endian
    # -ar 44100: 44.1 kHz sample rate
    # -ac 1: mono
    subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "44100",
        "-ac", "1",
        audio_path
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def get_video_duration(path: str) -> float:
    """
    Returns the duration of the video at `path` in seconds, using ffprobe.
    """
    # -v error: only show fatal errors
    # -show_entries format=duration: print only the duration
    # -of default=noprint_wrappers=1:nokey=1: output the raw value
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

# --- Endpoint 1: process-video ---
@app.post("/process-video")
async def process_video(file: UploadFile = File(...)):
    try: 
        t0 = time.time()
        print(f"[1] Start upload @ {t0:.3f}")
        uid = str(uuid.uuid4())
        filename = file.filename.replace('.mp4', f"_{uid}.mp4")
        uploads_dir = "./Data/Original_videos"
        os.makedirs(uploads_dir, exist_ok=True)
        path = os.path.join(uploads_dir, filename)
        path = path.replace("\\", "/")
        with open(path, "wb") as f:
            f.write(await file.read())
        t1 = time.time()
        print(f"[2] Saved video Δ {t1-t0:.3f}s")

        # Extract audio from video
        audio_name = os.path.splitext(os.path.basename(path))[0] + ".wav"
        raw_audio = os.path.join("Data/Extracted_Audio", audio_name)
        os.makedirs("Data/Extracted_Audio", exist_ok=True)
        subprocess.run([
            "ffmpeg", "-y",
            "-i", path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            raw_audio
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        t2 = time.time()
        print(f"[3] Audio extracted Δ {t2-t1:.3f}s")
        
        # --- Deepgram Transcription ---
        with open(raw_audio, "rb") as audio_file:
            buffer_data = audio_file.read()
        payload: FileSource = {"buffer": buffer_data}
        options = PrerecordedOptions(model="nova-3", smart_format=True, diarize=True)
        deepgram = DeepgramClient()
        response = deepgram.listen.rest.v("1").transcribe_file(payload, options, timeout=300)
        dg_json = response.to_json()
        dg_data = json.loads(dg_json)
        # Save response for debugging
        with open("deepgram_response.json", "w", encoding="utf-8") as f:
            f.write(json.dumps(dg_data, indent=2))

        # --- Parse Deepgram segments ---
        paragraphs = dg_data["results"]["channels"][0]["alternatives"][0]["paragraphs"]["paragraphs"]
        segments = []
        for para in paragraphs:
            for sent in para["sentences"]:
                segments.append({
                    "start": sent["start"],
                    "end": sent["end"],
                    "text": sent["text"]
                })
        t3 = time.time()
        print(f"[5] Transcribed Δ {t3-t2:.3f}s, segments: {len(segments)}")

        # Parse words with speaker info
        words = dg_data["results"]["channels"][0]["alternatives"][0].get("words", [])
        # Build a list of (start, end, speaker) for each word
        word_speakers = []
        for w in words:
            word_speakers.append({
                "start": w["start"],
                "end": w["end"],
                "speaker": w.get("speaker")
            })
        # Count total speaking time per speaker
        speaker_durations = {}
        for w in word_speakers:
            spk = w["speaker"]
            if spk is not None:
                duration = w["end"] - w["start"]
                speaker_durations[spk] = speaker_durations.get(spk, 0) + duration
        majority_speaker = max(speaker_durations, key=speaker_durations.get) if speaker_durations else None
        print(f"[5.2] Diarization complete. Majority speaker: {majority_speaker}")
        # Assign speaker to each segment by majority overlap
        segment_speakers = []
        for seg in segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            # Find all words overlapping this segment
            overlap = [w for w in word_speakers if w["end"] > seg_start and w["start"] < seg_end and w["speaker"] is not None]
            speaker_times = {}
            for w in overlap:
                overlap_start = max(seg_start, w["start"])
                overlap_end = min(seg_end, w["end"])
                overlap_duration = overlap_end - overlap_start
                if overlap_duration > 0:
                    speaker_times[w["speaker"]] = speaker_times.get(w["speaker"], 0) + overlap_duration
            dominant_speaker = max(speaker_times, key=speaker_times.get) if speaker_times else None
            segment_speakers.append(dominant_speaker)
        
        prompt = """
            You are given a list of transcript segments transcribed by whisper model, each prefixed by its timestamp in MM:SS format.  
            Your task is to refine each segment individually, with these strict rules:

            1. Preserve the exact number of segments.
            – Do not delete, merge, or split any segments.
            – If a segment contains only a filler word (e.g. “Okay.”, “Um,”) and no other content, remove the filler word and return the empty segment.

            2. Minimal changes only:
            – Remove only filler words ("So", "okay", "ok", “um,” “uh,” “like,” “you know”).
            – Correct obvious spelling mistakes or minor grammar errors that would otherwise hinder readability.
            – If there are some terms or words which you dont recognize, search it on internet and then if its not a word then replace it with a meaningful term/word which relates to the context for eg ihave seen this enough that if there is a demo of lovable ai tool, but whisper transcribes it wrongly as Allowable so replace it with lovable .
            – Do not rephrase or rewrite content that already makes sense.

            3. Maintain timestamps:
            – Output exactly one line per input segment.
            – Each line must begin with the original timestamp (MM:SS), a space, then the refined text.

            4. If Given segment is empty, return it as is with the timestamp.

            5. Examples:
            – Input: 01:23 Okay.
                Output: 01:23 Okay.
            – Input: 02:45 I, uh, think we should go.
                Output: 02:45 I think we should go.
            – Input: 03:10 
                Output: 03:10

            ――――――  
            Now refine the following segments:\n
            """

        # Prepare the rows for the spreadsheet
        current_t = 0
        rows = [["Start", "End", "New Start", "New End", "Original", "Refined", "Pause at end(sec)", "Audio Length", "Video Length", "Flag", "Clone Voice", "Recommended number of words"]]
        for seg_index, seg in enumerate(segments):
            if seg["start"] - current_t > 0.2:
                start_ts = current_t
                end_ts = seg['start']
                duration_sec = seg['start'] - current_t
                video_len = duration_sec
                prompt += f"{start_ts} \n"
                rows.append([start_ts, end_ts, start_ts, end_ts, "", "", 0, video_len, video_len, "", "yes", video_len*2.8])
            start_ts = seg['start']
            end_ts = seg['end']
            duration_sec = seg['end'] - seg['start']
            video_len = duration_sec
            prompt += f"{start_ts} {seg['text'].strip()}\n"
            clone_voice = "yes"  # Default to yes
            # Use Deepgram diarization: set to "no" if minority speaker
            if seg_index < len(segment_speakers) and segment_speakers[seg_index] is not None:
                if segment_speakers[seg_index] != majority_speaker:
                    clone_voice = "no"
            rows.append([start_ts, end_ts, "", "", seg['text'].strip(), "", 0, "", video_len, "", clone_voice, video_len*2.8])
            current_t = seg['end']
        duration = get_audio_duration(raw_audio)
        if duration - current_t > 1:
            start_ts = current_t
            end_ts = duration
            duration_sec = duration - current_t
            video_len = duration_sec
            rows.append([start_ts, end_ts, start_ts, end_ts, "", "", 0, video_len, video_len, "", "yes", video_len*2.8])
        t4 = time.time()
        print(f"[6] Built prompt+rows Δ {t4-t3:.3f}s")
        for row in rows:
            print(row[4])

        refined_lines = call_gemini(prompt).splitlines()
        t5 = time.time()
        print(f"[7] Gemini returned Δ {t5-t4:.3f}s")
        print(refined_lines)
        
        # Update the rows with refined text
        i = 1
        for line in refined_lines:
            parts = line.split(' ', 1)
            if len(parts) >=2:
                if rows[i][4] == "": i+=1
                _, text = parts
                rows[i][5] = text
            i += 1
            # if len(parts) < 2:
            #     rows[i][5] = ""  
            # else:
            #     _, text = parts
            #     rows[i][5] = text
            # i += 1
        
        sheet = sheets_service.spreadsheets().create(
            body={'properties': {'title': filename}},
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
        t6 = time.time()
        print(f"[8] Sheet updated Δ {t6-t5:.3f}s")

        # Upload the video to S3
        print(upload_file(path, unique_filename=f"Original_videos/{filename}"))
        os.remove(path)
        os.remove(raw_audio)
        t7 = time.time()
        print(f"Uploading Time = {t7-t6:.3f}s")
        print(f"Total time taken = {t7-t0:.3f}s")
        
        return JSONResponse({"spreadsheetId": sheet_id, "SpreadsheetUrl": f"https://docs.google.com/spreadsheets/d/{sheet_id}/edit?gid=0#gid=0"})
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

def create_word_highlighted_ass(words, ass_path, words_per_caption=7):
    def ass_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 100)
        return f"{h:d}:{m:02d}:{s:02d}.{ms:02d}"

    # ASS header with styles
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
Collisions: Normal
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding, BlurEdges
Style: Default,Arial,75,&H00000000,&H00FFFF00,&H40FFFFFF,&H40FFFFFF,0,0,0,0,100,100,0,0,4,0,0,2,20,20,50,1,5
Style: Highlight,Arial,75,&HFF0000,&H0000FFFF,&H40FFFFFF,&H40FFFFFF,1,0,0,0,100,100,0,0,4,0,0,2,20,20,50,1,5

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events = []

    for i in range(0, len(words), words_per_caption):
        word_group = words[i:i+words_per_caption]
        for j, word in enumerate(word_group):
            start = ass_time(word["start"])
            end = ass_time(word["end"])

            # Build dialogue line with the j-th word highlighted, using Deepgram's punctuated_word
            line = []
            for k, w in enumerate(word_group):
                text = w["punctuated_word"]
                if k == j:
                    line.append(r"{\rHighlight}" + text + r"{\rDefault}")
                else:
                    line.append(text)

            text = " ".join(line).strip()
            dialogue = f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}"
            events.append(dialogue)

    with open(ass_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n".join(events))


def process_segments_with_ffmpeg(segments, input_path, output_path, ass_path):   
    t0 = time.time()
    uid = input_path.split('/')[-1].split('_')[-1].split('.')[-2]
    tmp_base = os.path.join("Data/tmp", uid)
    tmp_audio = os.path.join(tmp_base, "audio.wav")
    ass_path = ass_path.replace("\\", "/")
    os.makedirs(tmp_base, exist_ok=True)

    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def run_ffmpeg(cmd, desc):
        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                return
            except subprocess.CalledProcessError as e:
                attempt += 1
                err = e.stderr if e.stderr else e.stdout
                logger.error(f"[FFmpeg] {desc} failed (attempt {attempt}/{MAX_RETRIES}): {err}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"[FFmpeg] {desc} failed: {err}")

    # === STEP 1: Build audio track ===

    audio_inputs = []
    audio_filters = []
    concat_inputs = []

    for seg in segments:
        apath = seg.get("audio_path")
        if apath and os.path.exists(apath) and apath not in audio_inputs:
            audio_inputs.append(apath)

    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        factor = seg["factor"]
        duration = (end - start) / factor

        audio_path = seg.get("audio_path")
        if audio_path is not None and os.path.exists(audio_path):
            aud_idx = audio_inputs.index(audio_path)
            audio_filters.append(
                f"[{aud_idx}:a]atrim=0:{duration},"
                "aresample=48000:async=1,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo,"
                f"apad=whole_dur={duration},"
                "asetpts=PTS-STARTPTS"
                f"[a{i}]"
            )
        else:
            audio_filters.append(
                f"aevalsrc=0:d={duration}:s=48000,"
                "aformat=sample_fmts=fltp:channel_layouts=stereo"
                f"[a{i}]"
            )
        concat_inputs.append(f"[a{i}]")

    audio_filters = [f for f in audio_filters if f.strip()]
    concat_inputs = [c for c in concat_inputs if c.strip()]
    concat_filter = f"{''.join(concat_inputs)}concat=n={len(segments)}:v=0:a=1[outa]"
    filter_complex = ";".join(audio_filters + [concat_filter])

    audio_cmd = ["ffmpeg", "-y"]
    for path in audio_inputs:
        audio_cmd += ["-i", path]
    audio_cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outa]",
        "-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2",
        tmp_audio
    ]

    run_ffmpeg(audio_cmd, "Audio processing")
    t1 = time.time()
    logger.info(f"Audio processing completed: {t1-t0:.2f}s")

    # === STEP 2: Call Deepgram ===

    with open(tmp_audio, "rb") as audio_file:
        buffer_data = audio_file.read()

    payload: FileSource = {"buffer": buffer_data}
    options = PrerecordedOptions(model="nova-3", smart_format=True)
    deepgram = DeepgramClient()
    response = deepgram.listen.rest.v("1").transcribe_file(payload, options, timeout=120)
    dg_json = response.to_json()
    dg_data = json.loads(dg_json)

    with open("deepgram_response.json", "w", encoding="utf-8") as f:
        f.write(json.dumps(dg_data, indent=2))

    words = dg_data["results"]["channels"][0]["alternatives"][0]["words"]
    create_word_highlighted_ass(words, ass_path, words_per_caption=7)

    t2 = time.time()
    logger.info(f"Deepgram transcription completed: {t2-t1:.2f}s")

    # === STEP 3: Create final video with audio and subtitles ===
    logger.info("Starting final video processing with audio and subtitles")
    video_inputs = []
    video_filters = []
    video_concat_inputs = []

    # Collect all unique video inputs
    for seg in segments:
        if seg["path"] not in video_inputs:
            video_inputs.append(seg["path"])

    # Build filtergraph for video
    for i, seg in enumerate(segments):
        vid_idx = video_inputs.index(seg["path"])
        start = seg["start"]
        end = seg["end"]
        factor = seg["factor"]

        trim_filter = f"[{vid_idx}:v]trim=start={start}:end={end}"
        if factor != 1.0:
            speed_filter = f",setpts=(PTS-STARTPTS)/{factor}"
        else:
            speed_filter = ",setpts=PTS-STARTPTS"

        video_filters.append(f"{trim_filter}{speed_filter}[v{i}]")
        video_concat_inputs.append(f"[v{i}]")

    video_concat_filter = f"{''.join(video_concat_inputs)}concat=n={len(segments)}:v=1:a=0[outv]"
    video_filter_complex = ";".join(video_filters + [video_concat_filter])

    # ESCAPE THE ass PATH FOR WINDOWS
    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")  
    
    # Build final filter_complex with escaped path
    final_filter = (
        f"{video_filter_complex};"
        f"[outv]ass='{escaped_ass}'[vout]"
    )


    final_cmd = [
        "ffmpeg", "-y",
        *chain.from_iterable([["-i", p] for p in video_inputs]),
        "-i", tmp_audio,
        "-filter_complex", final_filter,
        "-map", "[vout]",
        "-map", f"{len(video_inputs)}:a",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-x264-params", "stitchable=1",
        "-vsync", "2",
        "-c:a", "aac", "-b:a", "128k",
        "-fflags", "+genpts",
        "-avoid_negative_ts", "make_zero",
        output_path
    ]

    logger.debug(f"FFmpeg video filter complex:\n{video_filter_complex}")
    logger.debug(f"Full command:\n{' '.join(final_cmd)}")

    run_ffmpeg(final_cmd, "Final video processing with audio and subtitles")
    
    total_time = time.time() - t0
    logger.info(f"Final video processing completed: {total_time-t2:.2f}s")
    logger.info(f"Processing complete! Total time: {total_time:.2f}s")
    return output_path

def tts_worker_init(device='cpu', audio_prompt_path=None):
    # Preload the model and conditionals in each worker process
    get_tts_model(device, audio_prompt_path)

@app.post("/refresh-voiceover")
async def refresh_voiceover(sheetId: str):
    t0 = time.time()
    print(f"[Refresh] start @ {t0:.3f}")
    try:
        resp = sheets_service.spreadsheets().values().get(
            spreadsheetId=sheetId, range='A2:K'  
        ).execute()
        spreadsheet = sheets_service.spreadsheets().get(spreadsheetId=sheetId).execute()
    except HttpError as e:
        raise HTTPException(status_code=400, detail=str(e))
    values = resp.get('values', [])
    filename = spreadsheet['properties']['title']
    uploads_dir = "./Data/Original_videos"
    
    # Extract UID correctly
    base_name = filename.split('.')[-2]  # Remove extension
    parts = base_name.split('_')
    uid = parts[-1] if len(parts) > 1 else str(uuid.uuid4())
    
    original_path = os.path.join(uploads_dir, filename)
    original_path = original_path.replace("\\", "/")
    final_local = f"./Data/Final_videos/{filename}"
    tmp_base = os.path.join("Data/tmp", uid)
    cloned_path = os.path.abspath(os.path.join(tmp_base, "cloned"))
    ass_path = os.path.abspath(os.path.join(tmp_base, "captions.ass"))
    
    # Create directories with exist_ok=True
    os.makedirs(uploads_dir, exist_ok=True)
    os.makedirs("./Data/Final_videos", exist_ok=True) 
    os.makedirs(tmp_base, exist_ok=True)
    os.makedirs(cloned_path, exist_ok=True)
    os.makedirs("Ref_voice", exist_ok=True)
    if os.path.exists("Ref_voice"):
        print("Ref_voice directory exists")

    ref_audio_path = "Ref_voice/Anshul_Ref_Voice_trimmed.wav"  # Local path for TTS
    ref_audio_url = "Ref_voice/Anshul_Ref_Voice_trimmed.wav"

    if not os.path.exists(ref_audio_path):
        # Download the reference audio file
        print(download_file(ref_audio_url, ref_audio_path))

    if not os.path.exists(original_path):
        # Download the original video file
        print(download_file(f"Original_videos/{filename}", original_path))

    # Load previous state from CSV (if exists)
    state_csv_path = os.path.join(tmp_base, "state.csv")
    prev_state = {}
    
    if os.path.exists(state_csv_path):
        print(f"Loading state from {state_csv_path}")
        try:
            df = pandas.read_csv(state_csv_path)
            for _, row in df.iterrows():
                idx = row['Index']
                prev_state[idx] = {
                    'refined': str(row['Refined']),
                    'pause': str(row['Pause']),
                    'clonevoice': str(row.get('CloneVoice', 'yes'))  
                }
        except Exception as e:
            print(f"Error loading state CSV: {e}")

    t1 = time.time()
    print(f"[Refresh] setup done Δ {t1-t0:.3f}s")

    # ADDED: FFmpeg helper function for audio extraction
    MAX_RETRIES = 3
    RETRY_DELAY = 1

    def run_ffmpeg(cmd, desc):
        attempt = 0
        while attempt <= MAX_RETRIES:
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd, result.stdout, result.stderr)
                return
            except subprocess.CalledProcessError as e:
                attempt += 1
                err = e.stderr if e.stderr else e.stdout
                print(f"[FFmpeg] {desc} failed (attempt {attempt}/{MAX_RETRIES}): {err}")
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                else:
                    raise RuntimeError(f"[FFmpeg] {desc} failed: {err}")

    def extract_audio_segment(video_path, start, end, output_path):
        duration = end - start
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-q:a", "0",
            "-map", "a",
            output_path
        ]
        run_ffmpeg(cmd, f"Audio extraction ({start}-{end}s)")

    # Process segments
    updated_rows = values.copy()
    segments = []
    current_new_start = 0.0  
    
    # Initialize timing arrays
    orig_start = [0.0] * len(updated_rows)
    orig_end = [0.0] * len(updated_rows)
    prev_start = [0.0] * len(updated_rows)
    prev_end = [0.0] * len(updated_rows)
    new_start = [0.0] * len(updated_rows)
    new_end = [0.0] * len(updated_rows)
    
    # First pass: convert all timing strings to seconds
    for idx, row in enumerate(updated_rows):
        orig_start[idx] = float(row[0])
        orig_end[idx] = float(row[1])
        prev_start[idx] = float(row[2]) if len(row) > 2 and row[2] else orig_start[idx]
        prev_end[idx] = float(row[3]) if len(row) > 3 and row[3] else orig_end[idx]

    # --- Parallel TTS Generation ---
    # Allow user to set TTS worker count via env or use all CPU cores
    num_workers = int('2', multiprocessing.cpu_count())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TTS] Using {num_workers} workers on device: {device}")
    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=tts_worker_init,
        initargs=(device, ref_audio_path)
    )
    tts_tasks = []
    for idx, row in enumerate(updated_rows):
        refined = row[5] if len(row) > 5 and row[5] else ''
        pause = row[6] if len(row) > 6 and row[6] else '0'
        clonevoice = row[10] if len(row) > 10 and row[10] else 'yes'  
        unchanged = False
        if idx in prev_state:
            prev = prev_state[idx]
            unchanged = (refined == prev['refined'] and 
                         pause == prev['pause'] and 
                         clonevoice == prev['clonevoice'])
        if not unchanged and refined and clonevoice.lower() == "yes":
            audio_path = os.path.abspath(os.path.join(cloned_path, f"seg_{idx}.wav"))
            tts_tasks.append((idx, refined, audio_path))
    futures = {}
    tts_start = time.time()
    for idx, text, audio_path in tts_tasks:
        seg_t0 = {}
        seg_t0["idx"] = time.time()
        futures[executor.submit(tts_generate_segment, text, audio_path, ref_audio_path, device)] = (idx, text, audio_path)
    for future in concurrent.futures.as_completed(futures, timeout=1800):
        idx, text, audio_path = futures[future]
        try:
            future.result()
            print(f"[TTS] Segment {idx} generated in {time.time()-seg_t0['idx']:.2f}s: {audio_path}")
        except Exception as e:
            logger.error(f"TTS generation failed for segment {idx}: {e}")
    executor.shutdown(wait=True)
    print(f"[TTS] All segments generated in {time.time() - tts_start:.2f}s")

    # Second pass: process segments
    for idx, row in enumerate(updated_rows):
        loop_i = time.time()
        refined = row[5] if len(row) > 5 and row[5] else ''
        pause = row[6] if len(row) > 6 and row[6] else '0'
        clonevoice = row[10] if len(row) > 10 and row[10] else 'yes'  

        # Check if segment is unchanged from previous state
        unchanged = False
        if idx in prev_state:
            prev = prev_state[idx]
            unchanged = (refined == prev['refined'] and 
                         pause == prev['pause'] and 
                         clonevoice == prev['clonevoice'])

        video_len = orig_end[idx] - orig_start[idx]
        audio_path = None
        factor = 1.0
        audio_len_sec = video_len
        if unchanged:
            # Use previous audio path if exists
            audio_path = os.path.join(cloned_path, f"seg_{idx}.wav")
            if clonevoice.lower() == "no":
                if not os.path.exists(audio_path):
                    extract_audio_segment(original_path, orig_start[idx], orig_end[idx], audio_path)
                factor = 1.0
            elif refined:
                if not os.path.exists(audio_path):
                    # Should not happen, but fallback to extract
                    extract_audio_segment(original_path, orig_start[idx], orig_end[idx], audio_path)
                audio_len_sec = get_audio_duration(audio_path)
                audio_len_sec += float(pause)
                factor = video_len / audio_len_sec if audio_len_sec > 0 else 1.0
                if factor > 2.0:
                    factor = 2.0
                    audio_len_sec = video_len / factor
            else:
                factor = 1.0
        else:
            if clonevoice.lower() == "no":
                audio_path = os.path.join(cloned_path, f"seg_{idx}.wav")
                extract_audio_segment(original_path, orig_start[idx], orig_end[idx], audio_path)
                factor = 1.0
            elif refined:
                audio_path = os.path.abspath(os.path.join(cloned_path, f"seg_{idx}.wav"))
                if not os.path.exists(audio_path):
                    logger.error(f"Expected TTS audio not found for segment {idx}")
                audio_len_sec = get_audio_duration(audio_path)
                audio_len_sec += float(pause)
                factor = video_len / audio_len_sec if audio_len_sec > 0 else 1.0
                if factor > 2.0:
                    factor = 2.0
                    audio_len_sec = video_len / factor
            else:
                factor = 1.0
        new_end_val = current_new_start + (video_len / factor)
        new_start[idx] = current_new_start
        new_end[idx] = new_end_val
        current_new_start = new_end_val
        row[2] = new_start[idx]
        row[3] = new_end[idx]
        row[7] = video_len / factor
        while len(row) < 11:  
            row.append("")
        row[9] = factor
        row[10] = clonevoice
        segments.append({
            "start": orig_start[idx],
            "end": orig_end[idx],
            "factor": factor,
            "audio_path": audio_path,
            "path": original_path
        })
        print(f"[Refresh] row {idx} processed Δ {time.time()-loop_i:.3f}s")

    t3 = time.time()
    print(f"[Refresh] TTS done Δ {t3-t1:.3f}s")

    # Update sheet with new timings
    sheets_service.spreadsheets().values().update(
        spreadsheetId=sheetId, range='A2:K',
        valueInputOption='RAW', body={'values': updated_rows}
    ).execute()
    t4 = time.time()
    print(f"[Refresh] sheet updated Δ {t4-t3:.3f}s")

    # Save current state to CSV
    state_data = []
    for idx, row in enumerate(updated_rows):
        state_data.append({
            "Index": idx,
            "Refined": row[5] if len(row) > 5 else "",
            "Pause": row[6] if len(row) > 6 else "0",
            "CloneVoice": row[10] if len(row) > 10 else "yes"  
        })
    state_df = pandas.DataFrame(state_data)
    print(f"Saving state to {state_csv_path}")
    state_df.to_csv(state_csv_path, index=False)

    # Process video segments
    print(f"Starting video processing with {len(segments)} segments")
    processed_path = process_segments_with_ffmpeg(segments, original_path, final_local, ass_path)
    t5 = time.time()

    final_s3 = upload_file(processed_path, unique_filename=f"Final_videos/{processed_path.split('/')[-1]}")
    final_s3_url = final_s3['url']

    tmp_audio = os.path.join(tmp_base, "audio.wav")
    os.remove(final_local)
    os.remove(original_path)
    os.remove(tmp_audio)
    os.remove(ass_path)
    # for f in os.listdir(cloned_path):
    #     try:
    #         os.remove(os.path.join(cloned_path, f))
    #     except:
    #         pass
    # os.remove(state_csv_path)
    
    t6 = time.time()
    print(f"[Refresh] Uploading Time = {t6-t5:.3f}s")
    print(f"[Refresh] Total time taken = {t6-t0:.3f}s")

    return JSONResponse({
        # "processed_path": processed_path,
        "Final_s3_url": final_s3_url,
        "message": "Refresh completed successfully"
    })