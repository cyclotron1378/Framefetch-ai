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
    :root { color-scheme: dark; font-family: 'Inter', system-ui, sans-serif; }
    .block-container { padding: 2rem; max-width: 1200px; }
    body, .main { background-color: #0b0f19; }
    .stSidebar { background-color: #111827; border-right: none; }
    .stApp, .css-1v3fvcr { color: #f3f4f6; }
    
    /* Clean, soft UI elements */
    .stButton>button { 
        border-radius: 12px; padding: 0.75rem 1rem; font-weight: 500; 
        background: linear-gradient(180deg, #3b82f6 0%, #2563eb 100%);
        border: none; color: white; box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
    }
    .stButton>button:hover { box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3); transform: translateY(-1px); }
    
    .stTextInput>div>div>input { 
        background: #1f2937; color: #f9fafb; border: 1px solid #374151; border-radius: 12px; padding: 0.75rem; 
    }
    .stTextInput>div>div>input:focus { border-color: #3b82f6; box-shadow: none; }
    
    /* Logo gradient */
    .logo-text {
        font-size: 1.75rem; font-weight: 700; margin-bottom: 2rem;
        background: linear-gradient(90deg, #60a5fa, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    
    /* Menu items */
    .menu-item {
        padding: 0.75rem 1rem; color: #9ca3af; display: flex; align-items: center; gap: 0.75rem;
        border-radius: 8px; margin-bottom: 0.25rem; font-size: 0.95rem; cursor: pointer;
    }
    .menu-item:hover, .menu-item.active { background: #1f2937; color: #f3f4f6; }
    
    /* Cards */
    .clean-card {
        background: #1f2937; border-radius: 12px; padding: 0.5rem; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 0.5rem;
    }
    .clean-card img { border-radius: 8px; width: 100%; height: auto; object-fit: cover; }
    .card-caption { font-size: 0.8rem; color: #9ca3af; margin-top: 0.5rem; text-align: center; }
    
    .top-bar { display: flex; justify-content: flex-end; align-items: center; gap: 1rem; margin-bottom: 1.5rem; color: #9ca3af; }
    .user-profile { display: flex; align-items: center; gap: 0.5rem; background: #1f2937; padding: 0.4rem 0.8rem; border-radius: 999px; font-size: 0.9rem; border: 1px solid #374151;}
    </style>
    """,
    unsafe_allow_html=True,
)

if "video_data" not in st.session_state:
    st.session_state.video_data = {
        "name": None,
        "embeddings": None,
        "timestamps": None,
        "thumbnails": [],
        "path": None,
        "transcript": None,
        "text_embeddings": None,
    }
if "library" not in st.session_state:
    st.session_state.library = []

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

    if st.session_state.video_data.get("path") is not None:
        st.session_state.library.append(st.session_state.video_data.copy())

    st.session_state.video_data = {
        "name": uploaded_file.name,
        "embeddings": embeddings,
        "timestamps": timestamps,
        "thumbnails": thumbnails,
        "path": video_path,
        "transcript": transcript_segments,
        "text_embeddings": text_embeddings,
    }

    del frames
    gc.collect()
    st.success("Your video is now indexed and ready for powerful search.")

# --- LAYOUT & UI ---

with st.sidebar:
    st.markdown('<div class="logo-text">FrameFetch AI</div>', unsafe_allow_html=True)
    
    st.markdown("<div style='color: #9ca3af; font-size: 0.85rem; font-weight: 600; margin-bottom:0.5rem;'>NAVIGATION</div>", unsafe_allow_html=True)
    page = st.radio("Nav", ["Upload & Search", "Video Library"], label_visibility="collapsed")
    st.markdown("<br>", unsafe_allow_html=True)
    
    if page == "Upload & Search":
        uploaded_video = st.file_uploader("Drop Video File Here", type=["mp4", "mov", "avi"])
        analyze_button = st.button("UPLOAD VIDEO", type="primary", use_container_width=True)
    
        if analyze_button and uploaded_video:
            process_video(uploaded_video)
    
        if st.session_state.video_data["path"]:
            st.markdown("<br><div style='color:#9ca3af; font-size:0.85rem; margin-bottom:0.5rem; font-weight:600;'>Active Video</div>", unsafe_allow_html=True)
            st.progress(100, text="Ready - 100%")
            if st.button("Clear Active Video", use_container_width=True):
                st.session_state.video_data = {
                    "name": None, "embeddings": None, "timestamps": None, "thumbnails": [],
                    "path": None, "transcript": None, "text_embeddings": None
                }
                if hasattr(st, "rerun"): st.rerun()
                else: st.experimental_rerun()

# Main Interface
if page == "Upload & Search":
    if st.session_state.video_data["path"]:
        # 1. Video Player
        video_name = st.session_state.video_data.get("name") or "Unknown Video"
        video_name = os.path.splitext(video_name)[0].replace("_", " ").title()
            
        st.video(st.session_state.video_data["path"], start_time=int(st.session_state.get('jump_time', 0)))
        st.markdown(f"<h1 style='margin-top: 1rem;'>Current Video: <br><span style='color: #cbd5e1;'>{video_name}</span></h1>", unsafe_allow_html=True)
        
        st.write("---")
        
        # 2. Search Bar
        query = st.text_input("Search", placeholder="Search video library... (Type 'City Skyline' for results)", label_visibility="collapsed")
        
        # 3. Horizontal Results
        if query:
            query_visual_emb = vision_model.encode([query], convert_to_tensor=True)
            visual_hits = util.semantic_search(query_visual_emb, st.session_state.video_data["embeddings"], top_k=5)[0]
            
            query_text_emb = text_model.encode([query], convert_to_tensor=True)
            audio_hits = util.semantic_search(query_text_emb, st.session_state.video_data["text_embeddings"], top_k=5)[0]
            
            # Audio match text if any
            if audio_hits and audio_hits[0]["score"] > 0.28:
                best_speech = st.session_state.video_data["transcript"][audio_hits[0]["corpus_id"]]
                st.info(f"🎙️ **Speech Match ({int(best_speech['start'])}s):** {best_speech['text'].strip()}")
                
            cols = st.columns(5)
            for idx, hit in enumerate(visual_hits):
                frame_index = hit["corpus_id"]
                start_time = st.session_state.video_data["timestamps"][frame_index]
                score = round(hit["score"] * 100)
                
                with cols[idx]:
                    st.markdown("<div class='clean-card'>", unsafe_allow_html=True)
                    st.image(st.session_state.video_data["thumbnails"][frame_index], use_container_width=True)
                    st.markdown(f"<div class='card-caption'>{int(start_time)}s · {score}% Match</div>", unsafe_allow_html=True)
                    if st.button("Play", key=f"play_{idx}", use_container_width=True):
                        st.session_state['jump_time'] = start_time
                        if hasattr(st, "rerun"): st.rerun()
                        else: st.experimental_rerun()
                    st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Upload a video from the sidebar to start searching.")

elif page == "Video Library":
    st.markdown("<h1>📚 Video Library</h1>", unsafe_allow_html=True)
    st.write("Browse and manage all the videos you've analyzed during this session.")
    st.write("---")
    
    all_videos = []
    if st.session_state.video_data["path"]:
        all_videos.append(st.session_state.video_data)
    all_videos.extend(st.session_state.library)
    
    if len(all_videos) == 0:
        st.info("Your library is currently empty. Go to 'Upload & Search' to add a video!")
    else:
        # Show grid
        cols = st.columns(3)
        for idx, vid in enumerate(all_videos):
            with cols[idx % 3]:
                st.markdown("<div class='clean-card' style='margin-bottom: 1rem;'>", unsafe_allow_html=True)
                if vid.get("thumbnails"):
                    st.image(vid["thumbnails"][0], use_container_width=True)
                v_name = os.path.splitext(vid.get('name', 'Unknown'))[0].replace('_', ' ').title()
                st.markdown(f"<h4 style='margin: 0.5rem 0;'>{v_name}</h4>", unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Load", key=f"load_{idx}", use_container_width=True):
                        if idx > 0: # it's from library
                            curr = st.session_state.video_data.copy()
                            st.session_state.video_data = st.session_state.library[idx-1].copy()
                            st.session_state.library[idx-1] = curr
                        if hasattr(st, "rerun"): st.rerun()
                        else: st.experimental_rerun()
                with c2:
                    if st.button("Delete", key=f"del_{idx}", use_container_width=True, type="primary"):
                        if idx == 0: # deleting active video
                            st.session_state.video_data = {
                                "name": None, "embeddings": None, "timestamps": None, "thumbnails": [],
                                "path": None, "transcript": None, "text_embeddings": None
                            }
                        else:
                            st.session_state.library.pop(idx-1)
                        if hasattr(st, "rerun"): st.rerun()
                        else: st.experimental_rerun()
                st.markdown("</div>", unsafe_allow_html=True)