import asyncio
import os
import re
import subprocess
import tempfile
from pathlib import Path

import streamlit as st
import edge_tts
from deep_translator import GoogleTranslator
from faster_whisper import WhisperModel

st.set_page_config(page_title="Chinese → Khmer Dubber", page_icon="🎬", layout="wide")
st.title("🎬 Chinese → Khmer Video Dubber")
st.caption("Upload a Chinese video, translate the speech into Khmer, generate Khmer voices, and export a dubbed MP4 with Khmer subtitles.")

VOICE_OPTIONS = {
    "Male — Piseth": "km-KH-PisethNeural",
    "Female — Sreymom": "km-KH-SreymomNeural",
}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-4000:] or "Command failed")
    return p.stdout

def probe_duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())

def srt_time(seconds):
    ms = max(0, int(round(seconds * 1000)))
    h, rem = divmod(ms, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def clean_srt(text):
    return re.sub(r"\s+", " ", text).strip()

def fit_audio(src, dst, target_seconds):
    actual = probe_duration(src)
    if actual <= 0 or target_seconds <= 0:
        raise ValueError("Invalid audio segment duration")
    # atempo supports 0.5–2.0 per filter; chain filters for larger changes.
    speed = actual / target_seconds
    filters = []
    while speed > 2.0:
        filters.append("atempo=2.0")
        speed /= 2.0
    while speed < 0.5:
        filters.append("atempo=0.5")
        speed /= 0.5
    filters.append(f"atempo={speed:.5f}")
    run(["ffmpeg", "-y", "-i", str(src), "-filter:a", ",".join(filters),
         "-t", f"{target_seconds:.3f}", str(dst)])

async def make_tts(text, voice, output):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(str(output))

def translate(text):
    return GoogleTranslator(source="zh-CN", target="km").translate(text)

def build_srt(segments):
    rows = []
    for i, seg in enumerate(segments, 1):
        rows.extend([str(i), f"{srt_time(seg['start'])} --> {srt_time(seg['end'])}", seg["km"], ""])
    return "\n".join(rows)

def create_dub(video_path, segments, voice_map, output_path, keep_original):
    duration = probe_duration(video_path)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        clips = []
        for i, seg in enumerate(segments):
            text = seg["km"].strip()
            if not text:
                continue
            voice = voice_map[i]
            raw = td / f"voice_{i:04}.mp3"
            fitted = td / f"fit_{i:04}.wav"
            asyncio.run(make_tts(text, voice, raw))
            target = max(0.25, seg["end"] - seg["start"])
            fit_audio(raw, fitted, target)
            clips.append((seg["start"], fitted))

        # Place each speech segment on the original timeline with silence between lines.
        inputs = []
        filters = []
        for i, (start, clip) in enumerate(clips):
            inputs += ["-i", str(clip)]
            delay = max(0, int(start * 1000))
            filters.append(f"[{i}:a]adelay={delay}|{delay},apad=whole_dur={duration}[a{i}]")
        if not clips:
            raise ValueError("No translated speech was available to dub.")
        mix_inputs = "".join(f"[a{i}]" for i in range(len(clips)))
        filters.append(f"{mix_inputs}amix=inputs={len(clips)}:duration=longest:normalize=0[dub]")
        audio_graph = ";".join(filters)
        mixed = td / "khmer_dub.wav"
        run(["ffmpeg", "-y", *inputs, "-filter_complex", audio_graph,
             "-map", "[dub]", "-t", f"{duration:.3f}", str(mixed)])

        if keep_original:
            run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(mixed),
                 "-filter_complex", "[0:a]volume=0.18[bg];[bg][1:a]amix=inputs=2:duration=first:weights=0.25 1[d]",
                 "-map", "0:v", "-map", "[d]", "-c:v", "copy", "-c:a", "aac",
                 "-shortest", str(output_path)])
        else:
            run(["ffmpeg", "-y", "-i", str(video_path), "-i", str(mixed),
                 "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                 "-shortest", str(output_path)])

st.sidebar.header("Settings")
model_size = st.sidebar.selectbox("Speech recognition model", ["small", "medium", "large-v3"], index=0,
                                  help="Larger models are more accurate but need more RAM/VRAM and download time.")
device = st.sidebar.selectbox("Compute device", ["cpu", "cuda"], index=0)
compute_type = "int8" if device == "cpu" else "float16"
default_voice = st.sidebar.selectbox("Default Khmer voice", list(VOICE_OPTIONS), index=0)
keep_original = st.sidebar.checkbox("Keep quiet original audio underneath", value=False)

uploaded = st.file_uploader("Choose a Chinese video", type=["mp4", "mov", "mkv", "avi", "webm"])
if uploaded:
    suffix = Path(uploaded.name).suffix or ".mp4"
    workdir = Path(tempfile.mkdtemp(prefix="khmer_dubber_"))
    video_path = workdir / f"source{suffix}"
    video_path.write_bytes(uploaded.getbuffer())
    st.video(uploaded)
    if st.button("1 · Transcribe and translate", type="primary"):
        try:
            with st.spinner("Loading speech recognition model and transcribing Chinese…"):
                model = WhisperModel(model_size, device=device, compute_type=compute_type)
                segments_iter, info = model.transcribe(str(video_path), language="zh", vad_filter=True,
                                                       beam_size=5)
                segments = []
                for seg in segments_iter:
                    zh = clean_srt(seg.text)
                    if zh:
                        segments.append({"start": float(seg.start), "end": float(seg.end),
                                         "zh": zh, "km": ""})
            if not segments:
                st.error("No Chinese speech was detected.")
                st.stop()
            progress = st.progress(0)
            status = st.empty()
            for i, seg in enumerate(segments):
                status.write(f"Translating line {i+1}/{len(segments)}…")
                seg["km"] = translate(seg["zh"]) or seg["zh"]
                progress.progress((i + 1) / len(segments))
            st.session_state["segments"] = segments
            st.session_state["video_path"] = str(video_path)
            st.success(f"Found and translated {len(segments)} speech segments.")
        except Exception as e:
            st.error(f"Processing failed: {e}")

if "segments" in st.session_state:
    st.subheader("2 · Review Chinese and Khmer translation")
    segments = st.session_state["segments"]
    edited = []
    for i, seg in enumerate(segments):
        with st.expander(f"Line {i+1} · {seg['start']:.2f}s–{seg['end']:.2f}s", expanded=(i < 3)):
            st.text_area("Chinese", seg["zh"], key=f"zh_{i}", disabled=True)
            km = st.text_area("Khmer (editable)", seg["km"], key=f"km_{i}", height=90)
            voice_label = st.selectbox("Speaker voice", list(VOICE_OPTIONS), index=list(VOICE_OPTIONS).index(default_voice),
                                       key=f"voice_{i}")
            edited.append({"start": seg["start"], "end": seg["end"], "zh": seg["zh"],
                           "km": km, "voice": VOICE_OPTIONS[voice_label]})
    if st.button("3 · Generate Khmer dubbed video", type="primary"):
        try:
            output = Path(st.session_state["video_path"]).with_name("khmer_dubbed.mp4")
            with st.spinner("Generating Khmer speech and synchronizing audio…"):
                create_dub(st.session_state["video_path"], edited,
                           [x["voice"] for x in edited], output, keep_original)
            srt_path = output.with_suffix(".srt")
            srt_path.write_text(build_srt(edited), encoding="utf-8-sig")
            st.success("Your Khmer dub is ready.")
            st.video(str(output))
            st.download_button("Download dubbed MP4", output.read_bytes(), file_name="khmer_dubbed.mp4",
                               mime="video/mp4")
            st.download_button("Download Khmer subtitles (.srt)", srt_path.read_bytes(),
                               file_name="khmer_subtitles.srt", mime="application/x-subrip")
        except Exception as e:
            st.error(f"Dubbing failed: {e}")

st.divider()
st.caption("Privacy: processing runs on your computer after setup, but translation and text-to-speech use online services. Review translations before export. Very long dialogue may need timing or voice-speed adjustment.")
