import os
import io
import streamlit as st
import google.generativeai as genai
from PIL import Image

# -----------------------------
# Config & Gemini initialization
# -----------------------------
st.set_page_config(page_title="Gemini + Streamlit", page_icon="✨", layout="centered")

def get_api_key():
    # Prefer Streamlit secrets, fallback to env
    if "GEMINI_API_KEY" in st.secrets:
        return st.secrets["GEMINI_API_KEY"]
    return os.getenv("GEMINI_API_KEY")

API_KEY = get_api_key()
if not API_KEY:
    st.error("❗ Gemini API key not found. Set GEMINI_API_KEY in Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=API_KEY)

# Default text model; vision used dynamically if images present
DEFAULT_TEXT_MODEL = "gemini-1.5-flash"
VISION_MODEL = "gemini-1.5-flash"  # supports text+image

# Safety: keep it simple and permissive for demos; adjust for prod
GEN_CONFIG = {
    "temperature": 0.6,
    "top_p": 0.9,
    "top_k": 40,
    "max_output_tokens": 1024,
}

# -----------------------------
# UI
# -----------------------------
st.title("✨ Gemini + Streamlit")
st.caption("Text chat with optional image understanding (Gemini 1.5).")

with st.sidebar:
    st.subheader("Settings")
    selected_model = st.selectbox(
        "Model",
        [DEFAULT_TEXT_MODEL, "gemini-1.5-pro"],
        help="1.5 Flash is fast & cost-effective. 1.5 Pro is more capable."
    )
    temperature = st.slider("Temperature", 0.0, 1.0, GEN_CONFIG["temperature"], 0.1)
    max_tokens = st.slider("Max output tokens", 64, 2048, GEN_CONFIG["max_output_tokens"], 64)
    st.markdown("---")
    st.write("**API Key Source**:", "secrets" if "GEMINI_API_KEY" in st.secrets else "env")
    st.caption("Keys are never shown or logged.")

# update gen config from UI
GEN_CONFIG["temperature"] = float(temperature)
GEN_CONFIG["max_output_tokens"] = int(max_tokens)

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat history display
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input row
col1, col2 = st.columns([4, 1])
with col1:
    prompt = st.chat_input("Ask something… (you can also attach an image from the sidebar)")
with col2:
    pass

# Optional file uploader for images
uploaded_images = st.file_uploader(
    "Attach image(s) (optional, JPG/PNG/WebP)",
    type=["png", "jpg", "jpeg", "webp"],
    accept_multiple_files=True,
    help="Including an image switches to a vision-capable model automatically."
)

def to_gemini_image_parts(files):
    """Convert uploaded files to the format expected by Gemini."""
    parts = []
    for f in files:
        img = Image.open(f).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        b = buf.getvalue()
        parts.append({
            "mime_type": "image/jpeg",
            "data": b
        })
    return parts

def generate_response(prompt_text, images=None):
    """
    If images are present, send a multimodal prompt to a vision-capable model.
    Otherwise, use text-only generation.
    """
    if images:
        model = genai.GenerativeModel(model_name=VISION_MODEL)
        # Build content: [text, image1, image2, ...]
        content = [prompt_text] + to_gemini_image_parts(images)
        resp = model.generate_content(content, generation_config=GEN_CONFIG)
        return resp.text
    else:
        model = genai.GenerativeModel(model_name=selected_model)
        resp = model.generate_content(prompt_text, generation_config=GEN_CONFIG)
        return resp.text

# Handle new user input
if prompt:
    # Show user turn
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Model response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer = generate_response(prompt, uploaded_images if uploaded_images else None)
            except Exception as e:
                st.error(f"Model error: {e}")
                st.stop()
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Footer
st.markdown("---")
st.caption(
    "Built with Streamlit + Google Gemini. "
    "Be mindful of costs and rate limits. Avoid sending sensitive data."
)
``
