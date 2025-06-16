import streamlit as st
import requests
import json
from PyPDF2 import PdfReader
from io import BytesIO
from newspaper import Article
import time
import speech_recognition as sr
import tempfile
import os
from audio_recorder_streamlit import audio_recorder

# --- Session State Initialization ---
def init_session_state():
    defaults = {
        "messages": [],
        "custom_profile": "",
        "url_contexts": [],
        "file_contexts": [],
        "enable_search": False,
        "deep_thinking": False,
        "last_urls": [],
        "last_file_ids": [],
        "voice_enabled": False,
        "recognizer": None
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

    # Initialize speech recognition
    if st.session_state.recognizer is None:
        st.session_state.recognizer = sr.Recognizer()

init_session_state()

# --- Speech to Text Function ---
def speech_to_text_from_audio(audio_bytes):
    """Convert audio bytes to text using speech_recognition"""
    try:
        # Save audio bytes to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_file_path = tmp_file.name
        
        # Convert audio to text
        with sr.AudioFile(tmp_file_path) as source:
            audio = st.session_state.recognizer.record(source)
            text = st.session_state.recognizer.recognize_google(audio)
        
        # Clean up temporary file
        os.unlink(tmp_file_path)
        return text
    
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Speech recognition error: {str(e)}"
    except Exception as e:
        return f"Audio processing error: {str(e)}"

# --- Optimized Utility Functions ---
@st.cache_data(ttl=3600, show_spinner=False)
def extract_text_from_url(url):
    """Cached URL content extraction with error handling"""
    try:
        article = Article(url, request_timeout=10)
        article.download()
        article.parse()
        return article.text[:5000]
    except Exception as e:
        return f"❌ URL Error: {str(e)}"

def extract_text_from_file(uploaded_file):
    """File extraction with memory optimization"""
    try:
        if uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode("utf-8")[:5000]
        elif uploaded_file.type == "application/pdf":
            with BytesIO(uploaded_file.getvalue()) as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                return "\n".join(page.extract_text() or "" for page in pdf_reader.pages)[:5000]
        return "⚠️ Unsupported file format"
    except Exception as e:
        return f"❌ File Error: {str(e)}"

# --- Streamlined AI Response with Caching ---
def get_ai_response_stream(prompt):
    """Streaming response for faster perceived performance"""
    openrouter_api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not openrouter_api_key:
        yield "❌ API key missing. Set OPENROUTER_API_KEY in secrets.toml"
        return

    # Build system message efficiently
    system_parts = ["You are an AI assistant. Respond helpfully and accurately."]
    
    if st.session_state.custom_profile:
        system_parts.append(f"\n\nUser profile: {st.session_state.custom_profile}")
    
    if st.session_state.url_contexts:
        url_texts = "\n\n".join(
            [f"URL Context {i+1}:\n{ctx}" for i, ctx in enumerate(st.session_state.url_contexts)]
        )
        system_parts.append(f"\n\nURL Contexts:\n{url_texts}")
    
    if st.session_state.file_contexts:
        file_texts = "\n\n".join(
            [f"File Context {i+1}:\n{ctx}" for i, ctx in enumerate(st.session_state.file_contexts)]
        )
        system_parts.append(f"\n\nFile Contexts:\n{file_texts}")
    
    if st.session_state.enable_search:
        system_parts.append("\n\nUse web search for real-time info when needed.")
    
    if st.session_state.deep_thinking:
        system_parts.append("\n\nReason step-by-step before answering.")
    
    if st.session_state.voice_enabled:
        system_parts.append("\n\nKeep responses conversational and suitable for voice interaction.")

    system_message = "".join(system_parts)

    messages = [
        {"role": "system", "content": system_message},
        *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[-6:]]
    ]

    payload = {
        "model": "deepseek/deepseek-r1-0528:free",
        "messages": messages,
        "temperature": 0.7 if st.session_state.deep_thinking else 0.3,
        "stream": True
    }

    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    try:
        start_time = time.time()
        with requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            stream=True,
            timeout=30
        ) as response:
            response.raise_for_status()

            full_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8').strip()
                    if decoded_line.startswith('data:'):
                        json_part = decoded_line[5:].strip()

                        # ✅ [DONE] marker ko ignore karo
                        if json_part == "[DONE]":
                            break

                        # ⚠️ Skip empty or malformed lines
                        if not json_part:
                            continue

                        try:
                            json_data = json.loads(json_part)
                        except json.JSONDecodeError as e:
                            yield f"❌ JSON Decode Error: {str(e)}"
                            continue

                        if 'choices' in json_data:
                            delta = json_data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                token = delta['content']
                                full_response += token
                                yield token

            st.toast(f"Response time: {time.time()-start_time:.2f}s", icon="⏱️")

    except requests.exceptions.RequestException as e:
        yield f"❌ Request Error: {str(e)}"
    except Exception as e:
        yield f"❌ API Error: {str(e)}"



# --- UI Components ---
with st.sidebar:
    st.header("⚙️ AI Configuration")

    # Voice Settings
    with st.expander("🎤 Voice Settings"):
        st.session_state.voice_enabled = st.checkbox(
            "Enable Voice Input",
            value=st.session_state.voice_enabled,
            help="Enable speech-to-text for voice commands"
        )

    with st.expander("🌐 URL Context"):
        url = st.text_input("Enter URL:", key="url_input", placeholder="https://example.com")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Add URL", key="add_url_btn") and url:
                if url not in st.session_state.last_urls:
                    with st.spinner("Extracting..."):
                        content = extract_text_from_url(url)
                        if not content.startswith("❌"):
                            st.session_state.url_contexts.append(content)
                            st.session_state.last_urls.append(url)
                            st.success("Content loaded!")
                        else:
                            st.error(content)
                else:
                    st.warning("URL already added")
        with col2:
            if st.button("Clear URLs", key="clear_url_btn"):
                st.session_state.url_contexts = []
                st.session_state.last_urls = []
                st.success("URLs cleared!")
        
        # Display current URLs
        if st.session_state.url_contexts:
            st.subheader("Current URLs")
            for i, url in enumerate(st.session_state.last_urls):
                st.caption(f"{i+1}. {url}")

    with st.expander("👤 User Profile"):
        st.session_state.custom_profile = st.text_area(
            "About you:", 
            value=st.session_state.custom_profile,
            height=100,
            help="This will persist across page refreshes"
        )

    with st.expander("🧠 AI Settings"):
        st.session_state.enable_search = st.checkbox(
            "Enable web search", 
            value=st.session_state.enable_search
        )
        st.session_state.deep_thinking = st.checkbox(
            "Deep thinking mode", 
            value=st.session_state.deep_thinking
        )

    with st.expander("📁 File Context"):
        uploaded_files = st.file_uploader(
            "Upload PDF/TXT", 
            type=["pdf", "txt"],
            key="file_uploader",
            accept_multiple_files=True
        )
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("Add Files", key="add_file_btn") and uploaded_files:
                for uploaded_file in uploaded_files:
                    current_file_id = f"{uploaded_file.name}-{uploaded_file.size}"
                    if current_file_id not in st.session_state.last_file_ids:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            content = extract_text_from_file(uploaded_file)
                            if not content.startswith("❌") and not content.startswith("⚠️"):
                                st.session_state.file_contexts.append(content)
                                st.session_state.last_file_ids.append(current_file_id)
                                st.success(f"{uploaded_file.name} processed!")
                            else:
                                st.error(content)
                    else:
                        st.warning(f"{uploaded_file.name} already added")
        with col2:
            if st.button("Clear Files", key="clear_file_btn"):
                st.session_state.file_contexts = []
                st.session_state.last_file_ids = []
                st.success("Files cleared!")
        
        # Display current files
        if st.session_state.last_file_ids:
            st.subheader("Current Files")
            for i, file_id in enumerate(st.session_state.last_file_ids):
                st.caption(f"{i+1}. {file_id.split('-')[0]}")

# --- Main Chat Interface ---
st.title("⚡ Turbo AI Assistant with Voice")
st.caption("Configure settings in the sidebar → | Voice input available!")

# Voice input section
if st.session_state.voice_enabled:
    st.markdown("### 🎤 Voice Input")
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#e8b62c",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x",
        pause_threshold=2.0
    )
    
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Converting speech to text..."):
            transcribed_text = speech_to_text_from_audio(audio_bytes)
        
        if transcribed_text and not any(e in transcribed_text.lower() for e in ["error", "could not"]):
            st.success(f"**You said:** {transcribed_text}")
            # Auto-submit and process the transcribed text
            st.session_state.messages.append({"role": "user", "content": transcribed_text})
            with st.chat_message("user"):
                st.markdown(transcribed_text)
            
            with st.chat_message("assistant"):
                response_placeholder = st.empty()
                full_response = ""
                for token in get_ai_response_stream(transcribed_text):
                    full_response += token
                    response_placeholder.markdown(full_response + "▌")
                response_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
            # Trim message history
            if len(st.session_state.messages) > 50:
                st.session_state.messages = st.session_state.messages[-30:]
            
            # st.rerun()
        else:
            st.error(f"Speech recognition failed: {transcribed_text}")

# Display chat history
for msg in st.session_state.messages[-10:]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Text input handling
if prompt := st.chat_input("Ask me anything... (or use voice input above)"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream tokens as they arrive
        for token in get_ai_response_stream(prompt):
            full_response += token
            response_placeholder.markdown(full_response + "▌")

        response_placeholder.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Trim message history
    if len(st.session_state.messages) > 50:
        st.session_state.messages = st.session_state.messages[-30:]
        
        
        