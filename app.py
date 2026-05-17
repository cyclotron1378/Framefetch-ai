import streamlit as st
import cv2
import torch
import whisper
from whisper.model import disable_sdpa
import gc
import numpy as np
import tempfile
import os
import time
from sentence_transformers import SentenceTransformer, util
from PIL import Image

# --- CONFIG & SESSION STATE ---
st.set_page_config(page_title="FrameFetch AI", layout="wide", page_icon="🎬")

st.markdown(
    """
    <style>
    :root {
        color-scheme: light;
        font-family: Inter, system-ui, sans-serif;
    }
    .css-18e3th9 { padding-top: 1.5rem; }
    .block-container { padding: 1.5rem 2rem 2rem; max-width: 1440px; }
    body, .main { background: linear-gradient(135deg, #0f172a 0%, #1e293b 35%, #0f172a 100%); }
    .stSidebar { background: rgba(15, 23, 42, 0.96); border-right: 1px solid rgba(148, 163, 184, 0.12); }
    .stApp, .css-1v3fvcr { color: #e2e8f0; }
    .stButton>button { border-radius: 999px; padding: 0.9rem 1.3rem; font-weight: 600; }
    .stButton>button:hover { transform: translateY(-1px); }
    .stTextInput>div>div>input { background: rgba(15, 23, 42, 0.9); color: #f8fafc; border: 1px solid rgba(148, 163, 184, 0.2); }
    .stTextInput>div>label { color: #cbd5e1; }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #f8fafc; }
    .stProgress>div>div>div>div { background: linear-gradient(90deg, #7c3aed, #2563eb); }
    [data-testid="stMetric"] { background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(148, 163, 184, 0.16); box-shadow: 0 24px 60px rgba(15, 23, 42, 0.35); border-radius: 18px; color: #f8fafc; }
    div[data-testid="stExpander"] { border: 1px solid rgba(148, 163, 184, 0.16); border-radius: 24px; background: rgba(15, 23, 42, 0.88); }
    .hero-panel { padding: 2rem; border-radius: 28px; background: rgba(15, 23, 42, 0.92); border: 1px solid rgba(148, 163, 184, 0.14); box-shadow: 0 40px 90px rgba(15, 23, 42, 0.3); margin-bottom: 1.5rem; }
    .hero-title { font-size: clamp(2.5rem, 3.6vw, 4.4rem); line-height: 1.02; margin-bottom: 0.5rem; letter-spacing: -0.06em; }
    .hero-text { color: #cbd5e1; font-size: 1.05rem; max-width: 860px; margin-bottom: 1.5rem; }
    .glass-card { background: rgba(15, 23, 42, 0.88); border: 1px solid rgba(148, 163, 184, 0.12); border-radius: 20px; padding: 1rem; box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2); transition: transform 0.3s ease, box-shadow 0.3s ease; text-align: center; margin-bottom: 1rem; }
    .glass-card:hover { transform: translateY(-4px); box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3); border-color: rgba(59, 130, 246, 0.4); }
    .glass-card img { border-radius: 12px; }
    .result-pill { display: inline-flex; gap: 0.65rem; align-items: center; padding: 0.65rem 0.95rem; border-radius: 999px; background: rgba(59, 130, 246, 0.16); color: #dbeafe; font-size: 0.96rem; }
    div[data-testid="caption"] { color: #94a3b8 !important; font-size: 0.85rem !important; margin-bottom: 0.5rem; text-align: center; }
    .section-heading { font-size: 1.55rem; margin-bottom: 0.8rem; }
    .audio-snippet { color: #cbd5e1; margin-bottom: 0.9rem; padding: 0.85rem 1rem; border-radius: 18px; background: rgba(30, 41, 59, 0.82); border: 1px solid rgba(148, 163, 184, 0.1); }
    .pulse { animation: pulse 4s ease-in-out infinite; }
    @keyframes pulse { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
    .sparkle { position: absolute; width: 10px; height: 10px; background: rgba(59, 130, 246, 0.9); border-radius: 999px; filter: blur(1px); animation: sparkle 3s linear infinite; }
    @keyframes sparkle { 0% { opacity: 0; transform: translate(0, 0) scale(0.8); } 50% { opacity: 1; } 100% { opacity: 0; transform: translate(24px, -24px) scale(1.2); } }
    </style>
    """,
    unsafe_allow_html=True,
)

if "video_data" not in st.session_state:
    st.session_state.video_data = {
        "embeddings": None,
        "timestamps": None,
        "thumbnails": [],
        "path": None,
        "transcript": None,
        "text_embeddings": None,
    }

# --- MODEL LOADING (Optimized for CPU) ---
@st.cache_resource
def load_models():
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    with st.spinner(f"Loading AI models on {device.upper()}... Please wait."):
        # clip-ViT-B-32: fast and accurate enough for semantic search on CPU
        vision_model = SentenceTransformer('clip-ViT-B-32', device=device)
        text_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        # whisper base: ~3x faster than 'small' on CPU, still good accuracy
        audio_model = whisper.load_model("base", device=device)

    return vision_model, text_model, audio_model, device

vision_model, text_model, audio_model, device = load_models()

# --- LOGIC FUNCTIONS ---
def process_video(uploaded_file):
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    frames = []
    timestamps = []
    thumbnails = []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    duration_sec = total_frames / fps
    # Sample 1 frame every 2 seconds — halves processing time vs 1/sec
    sample_rate = max(1, int(fps * 2.0))

    st.info(f"Step 1: Extracting frames from {duration_sec:.0f}s video (1 frame every 2s)...")
    progress_bar = st.progress(0)
    t0 = time.time()

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % sample_rate == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            frames.append(pil_img.resize((224, 224)))
            timestamps.append(count / fps)
            thumbnails.append(pil_img.resize((240, 136)))

            progress_bar.progress(min(count / total_frames, 1.0))

        count += 1

    cap.release()
    st.success(f"Extracted {len(frames)} frames in {time.time() - t0:.1f}s")

    st.info(f"Step 2: Embedding {len(frames)} frames with CLIP...")
    t1 = time.time()
    embeddings = vision_model.encode(frames, convert_to_tensor=True, batch_size=32)
    st.success(f"CLIP encoding done in {time.time() - t1:.1f}s")

    st.info("Step 3: Transcribing and embedding speech...")
    t2 = time.time()
    with disable_sdpa():
        audio_result = audio_model.transcribe(video_path, fp16=False)
    transcript_segments = audio_result.get('segments', [])
    transcript_texts = [segment['text'].strip() for segment in transcript_segments]
    if transcript_texts:
        text_embeddings = text_model.encode(transcript_texts, convert_to_tensor=True, batch_size=16)
    else:
        text_embeddings = torch.zeros((0, 384))
    st.success(f"Transcription done in {time.time() - t2:.1f}s ({len(transcript_segments)} segments)")

    st.session_state.video_data.update({
        "embeddings": embeddings,
        "timestamps": timestamps,
        "thumbnails": thumbnails,
        "path": video_path,
        "transcript": transcript_segments,
        "text_embeddings": text_embeddings,
    })

    del frames
    gc.collect()
    st.success("Your video is now indexed and ready for powerful search.")

# --- APP HEADER ---
with st.container():
    st.markdown(
        """
        <div class="hero-panel">
            <div class="hero-title">FrameFetch AI</div>
            <p class="hero-text">
                Discover moments in video instantly with multi-modal semantic search, smooth playback,
                and a modern interactive experience built for fast exploration.
            </p>
            <div class="result-pill">Visual search + speech understanding + instant frame preview</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("## Upload & analyze")
    uploaded_video = st.file_uploader("Choose a video to index", type=["mp4", "mov", "avi"])
    analyze_button = st.button("Analyze video", type="primary", use_container_width=True)

    if analyze_button and uploaded_video:
        process_video(uploaded_video)

    if st.session_state.video_data["path"]:
        st.divider()
        st.markdown("### Session status")
        st.metric("Indexed frames", len(st.session_state.video_data["timestamps"] or []))
        st.caption(f"Running on **{device.upper()}**")
        if st.button("Clear indexed video", use_container_width=True):
            if os.path.exists(st.session_state.video_data["path"]):
                os.remove(st.session_state.video_data["path"])
            st.session_state.video_data = {
                "embeddings": None,
                "timestamps": None,
                "thumbnails": [],
                "path": None,
                "transcript": None,
                "text_embeddings": None,
            }
            st.experimental_rerun()

if st.session_state.video_data["embeddings"] is not None:
    query = st.text_input(
        "Search across scenes and speech",
        placeholder="e.g. crowded train station, closing deal, red jacket",
    )

    if query:
        query_visual_emb = vision_model.encode([query], convert_to_tensor=True)
        visual_hits = util.semantic_search(query_visual_emb, st.session_state.video_data["embeddings"], top_k=6)[0]

        query_text_emb = text_model.encode([query], convert_to_tensor=True)
        audio_hits = util.semantic_search(query_text_emb, st.session_state.video_data["text_embeddings"], top_k=4)[0]

        best_time = None
        if audio_hits and audio_hits[0]["score"] > 0.28:
            best_time = st.session_state.video_data["transcript"][audio_hits[0]["corpus_id"]]["start"]
        else:
            best_time = st.session_state.video_data["timestamps"][visual_hits[0]["corpus_id"]]

        summary_col, playback_col = st.columns([1, 1.4], gap="large")

        with summary_col:
            st.markdown("<div class='section-heading'>Search insights</div>", unsafe_allow_html=True)
            if audio_hits:
                st.markdown("<div class='audio-snippet'><strong>Top speech match</strong><br/>", unsafe_allow_html=True)
                for hit in audio_hits[:3]:
                    segment = st.session_state.video_data["transcript"][hit["corpus_id"]]
                    score = round(hit["score"] * 100)
                    st.markdown(
                        f"<div class='audio-snippet'><strong>{int(segment['start'])}s</strong> · {score}% match<br/>{segment['text'].strip()}</div>",
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No close speech matches found. Try another phrase or browse the visual previews.")

            st.markdown("<div class='section-heading'>Top visual matches</div>", unsafe_allow_html=True)
            thumb_cols = st.columns(2)
            for idx, hit in enumerate(visual_hits):
                frame_index = hit["corpus_id"]
                start_time = st.session_state.video_data["timestamps"][frame_index]
                score = round(hit["score"] * 100)
                with thumb_cols[idx % 2]:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.image(
                        st.session_state.video_data["thumbnails"][frame_index],
                        use_container_width=True,
                        caption=f"{int(start_time)}s · {score}%"
                    )
                    if st.button(f"Jump to {int(start_time)}s", key=f"jump_{idx}", use_container_width=True):
                        best_time = start_time
                    st.markdown("</div>", unsafe_allow_html=True)

        with playback_col:
            st.markdown("<div class='section-heading'>Playback</div>", unsafe_allow_html=True)
            st.video(st.session_state.video_data["path"], start_time=int(best_time))
            with st.expander("Expand transcript", expanded=False):
                for segment in st.session_state.video_data["transcript"]:
                    start = int(segment["start"])
                    st.markdown(f"**{start}s** — {segment['text'].strip()}")
else:
    st.info("Upload a video and click **Analyze video** to unlock frame search and speech exploration.")