# Chinese → Khmer Video Dubber

A local Streamlit starter app for:
- Chinese speech recognition (faster-whisper)
- Chinese → Khmer translation (Google Translate via deep-translator)
- Khmer neural voice generation (Edge TTS: Piseth male / Sreymom female)
- Timeline-aligned dubbed MP4 and Khmer SRT export

## Requirements
- Python 3.10+
- FFmpeg and ffprobe installed and available on PATH
- Internet connection for model download, translation, and Edge TTS
- For CUDA, a compatible NVIDIA setup and CUDA-enabled PyTorch/CTranslate2 runtime

## Run
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```
Open the local URL shown by Streamlit (usually http://localhost:8501).

## Notes / limitations
- This is a local starter implementation, not a hosted service.
- Translation and TTS require internet access and may be subject to provider limits.
- Review the Khmer translation before rendering.
- Dialogue is fitted to each detected speech segment; very long Khmer translations can sound fast.
- Speaker assignment is manual per segment. Automatic speaker diarization and voice cloning are not included.
- Original background/music is removed by default. The optional quiet-original setting mixes the source audio underneath.
- Test with a short clip first.
