"""
Multi-Chat GPT Clone with Streamlit + OpenRouter

Features:
- Multiple chat sessions with sidebar navigation
- Model selection (GPT-4o, Claude, Gemini, Llama, etc.)
- LLM config parameters (temperature, max tokens, top_p, etc.)
- Full conversation history sent to the model
- Export, rename, delete chats
- System prompt customization
"""

import streamlit as st
import time
import uuid
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import os

# ─── Page Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multi-Chat GPT",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Load Environment ─────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPEN_ROUTER_KEY")

if not api_key:
    st.error("🔑 OPEN_ROUTER_KEY not found in .env file. Please set it and restart.")
    st.stop()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    default_headers={
        "HTTP-Referer": "https://localhost:8501",
        "X-Title": "Multi-Chat GPT Clone",
    }
)

# ─── Available Models ─────────────────────────────────────────────────
AVAILABLE_MODELS = {
    "GPT-4o": "openai/gpt-4o",
    "GPT-4o Mini": "openai/gpt-4o-mini",
    "Claude Sonnet 4": "anthropic/claude-sonnet-4",
    "Claude Haiku 3.5": "anthropic/claude-3.5-haiku",
    "Gemini 2.0 Flash": "google/gemini-2.0-flash-001",
    "Llama 3.1 8B": "meta-llama/llama-3.1-8b-instruct",
    "Llama 3.1 70B": "meta-llama/llama-3.1-70b-instruct",
    "Mistral 7B": "mistralai/mistral-7b-instruct",
    "DeepSeek V3": "deepseek/deepseek-chat",
}

DEFAULT_SYSTEM_PROMPT = "You are a helpful, friendly, and knowledgeable AI assistant."

# ─── Session State Initialization ─────────────────────────────────────
def initialize_session_state():
    """Initialize all session state variables."""
    if "chats" not in st.session_state:
        # chats is a dict of chat_id -> chat data
        first_chat_id = str(uuid.uuid4())
        st.session_state.chats = {
            first_chat_id: {
                "title": "New Chat",
                "messages": [],
                "created_at": datetime.now(),
                "model": "GPT-4o",
            }
        }
        st.session_state.active_chat_id = first_chat_id

    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = list(st.session_state.chats.keys())[0]

    # LLM configuration defaults
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = {
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "editing_chat_id" not in st.session_state:
        st.session_state.editing_chat_id = None

initialize_session_state()

# ─── Helper Functions ─────────────────────────────────────────────────
def get_active_chat():
    """Get the currently active chat data."""
    return st.session_state.chats.get(st.session_state.active_chat_id, None)

def create_new_chat():
    """Create a new chat session and switch to it."""
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now(),
        "model": list(AVAILABLE_MODELS.keys())[0],
    }
    st.session_state.active_chat_id = chat_id

def delete_chat(chat_id):
    """Delete a chat session."""
    if len(st.session_state.chats) <= 1:
        st.toast("⚠️ Cannot delete the last chat.", icon="⚠️")
        return

    del st.session_state.chats[chat_id]

    # If we deleted the active chat, switch to the most recent one
    if st.session_state.active_chat_id == chat_id:
        st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]

def auto_title_chat(chat_id, first_user_message):
    """Generate a short title from the first user message."""
    title = first_user_message[:40].strip()
    if len(first_user_message) > 40:
        title += "..."
    st.session_state.chats[chat_id]["title"] = title

def build_api_messages(chat):
    """Build the messages array for the API call, including system prompt and history."""
    messages = [{"role": "system", "content": st.session_state.system_prompt}]

    for msg in chat["messages"]:
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })

    return messages

def generate_response(chat):
    """Call OpenRouter API with full conversation history and LLM config."""
    model_key = chat.get("model", "GPT-4o")
    model_id = AVAILABLE_MODELS.get(model_key, "openai/gpt-4o")

    messages = build_api_messages(chat)
    config = st.session_state.llm_config

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            top_p=config["top_p"],
            frequency_penalty=config["frequency_penalty"],
            presence_penalty=config["presence_penalty"],
            stream=True,
        )
        return response
    except Exception as e:
        st.error(f"API Error: {str(e)}")
        return None

def export_chat(chat, assistant_name="Assistant"):
    """Export chat as text."""
    export = f"Chat: {chat['title']}\n"
    export += f"Model: {chat.get('model', 'Unknown')}\n"
    export += f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
    export += "=" * 50 + "\n\n"

    for msg in chat["messages"]:
        role = "You" if msg["role"] == "user" else assistant_name
        timestamp = msg.get("timestamp", "")
        if timestamp:
            timestamp = f" [{timestamp}]"
        export += f"{role}{timestamp}:\n{msg['content']}\n\n"

    return export

# ─── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:

    # New Chat button
    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        create_new_chat()
        st.rerun()

    st.divider()

    # ── Chat List ──
    st.subheader("💬 Chats")

    # Sort chats by creation time (newest first)
    sorted_chats = sorted(
        st.session_state.chats.items(),
        key=lambda x: x[1]["created_at"],
        reverse=True
    )

    for chat_id, chat_data in sorted_chats:
        is_active = chat_id == st.session_state.active_chat_id
        is_editing = st.session_state.editing_chat_id == chat_id

        col_chat, col_edit, col_del = st.columns([7, 1.5, 1.5])

        with col_chat:
            if is_editing:
                new_title = st.text_input(
                    "Rename",
                    value=chat_data["title"],
                    key=f"rename_{chat_id}",
                    label_visibility="collapsed",
                    on_change=lambda cid=chat_id: (
                        setattr(st.session_state, 'editing_chat_id', None),
                        st.session_state.chats[cid].update(
                            {"title": st.session_state.get(f"rename_{cid}", chat_data["title"])}
                        )
                    )
                )
            else:
                button_label = f"{'▶ ' if is_active else ''}{chat_data['title']}"
                if st.button(
                    button_label,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary"
                ):
                    st.session_state.active_chat_id = chat_id
                    st.rerun()

        with col_edit:
            if st.button("✏️", key=f"edit_{chat_id}", help="Rename"):
                st.session_state.editing_chat_id = chat_id
                st.rerun()

        with col_del:
            if st.button("🗑️", key=f"del_{chat_id}", help="Delete"):
                delete_chat(chat_id)
                st.rerun()

    st.divider()

    # ── Model Selection (per chat) ──
    st.subheader("🤖 Model")
    active_chat = get_active_chat()
    if active_chat:
        current_model = active_chat.get("model", "GPT-4o")
        model_names = list(AVAILABLE_MODELS.keys())
        current_index = model_names.index(current_model) if current_model in model_names else 0

        selected_model = st.selectbox(
            "Model:",
            model_names,
            index=current_index,
            label_visibility="collapsed"
        )
        active_chat["model"] = selected_model

        st.caption(f"`{AVAILABLE_MODELS[selected_model]}`")

    st.divider()

    # ── LLM Configuration ──
    st.subheader("⚙️ LLM Parameters")
    config = st.session_state.llm_config

    config["temperature"] = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=config["temperature"],
        step=0.1,
        help="Higher = more creative, Lower = more focused"
    )

    config["max_tokens"] = st.slider(
        "Max Tokens",
        min_value=256,
        max_value=8192,
        value=config["max_tokens"],
        step=256,
        help="Maximum length of the response"
    )

    with st.expander("Advanced Parameters"):
        config["top_p"] = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.0,
            max_value=1.0,
            value=config["top_p"],
            step=0.05,
            help="Controls diversity via nucleus sampling"
        )

        config["frequency_penalty"] = st.slider(
            "Frequency Penalty",
            min_value=0.0,
            max_value=2.0,
            value=config["frequency_penalty"],
            step=0.1,
            help="Reduces repetition of token sequences"
        )

        config["presence_penalty"] = st.slider(
            "Presence Penalty",
            min_value=0.0,
            max_value=2.0,
            value=config["presence_penalty"],
            step=0.1,
            help="Encourages the model to talk about new topics"
        )

    st.divider()

    # ── System Prompt ──
    st.subheader("📝 System Prompt")
    st.session_state.system_prompt = st.text_area(
        "System Prompt",
        value=st.session_state.system_prompt,
        height=100,
        label_visibility="collapsed",
        help="Instructions that define the assistant's behavior"
    )

    if st.button("Reset to Default", type="secondary"):
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
        st.rerun()

    st.divider()

    # ── Chat Actions ──
    st.subheader("🔧 Actions")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            if active_chat:
                active_chat["messages"] = []
                st.rerun()

    with col2:
        if active_chat and active_chat["messages"]:
            chat_export = export_chat(active_chat)
            st.download_button(
                "📤 Export",
                chat_export,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain",
                use_container_width=True
            )

# ─── Main Chat Area ───────────────────────────────────────────────────
active_chat = get_active_chat()

if not active_chat:
    st.error("No active chat found.")
    st.stop()

# Header
header_col1, header_col2 = st.columns([8, 2])
with header_col1:
    st.title(f"💬 {active_chat['title']}")
with header_col2:
    st.caption(f"🤖 {active_chat.get('model', 'GPT-4o')}")
    st.caption(f"🌡️ temp: {config['temperature']} | tokens: {config['max_tokens']}")

# Chat messages display
chat_container = st.container()
with chat_container:
    if not active_chat["messages"]:
        # Welcome message
        st.markdown(
            """
            <div style='text-align: center; padding: 60px 20px; color: #888;'>
                <h2>👋 Start a conversation</h2>
                <p>Type a message below to begin chatting with the selected model.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                if "timestamp" in message:
                    st.caption(message["timestamp"])
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your message..."):
    # Auto-title the chat from the first user message
    if not active_chat["messages"]:
        auto_title_chat(st.session_state.active_chat_id, prompt)

    # Add user message
    user_message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    }
    active_chat["messages"].append(user_message)

    # Display user message
    with st.chat_message("user"):
        st.caption(user_message["timestamp"])
        st.markdown(prompt)

    # Generate and display assistant response with streaming
    with st.chat_message("assistant"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.caption(timestamp)

        response_stream = generate_response(active_chat)

        if response_stream:
            # Stream the response
            full_response = st.write_stream(response_stream)

            # Save assistant message
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "timestamp": timestamp
            }
            active_chat["messages"].append(assistant_message)
        else:
            error_msg = "Sorry, I encountered an error generating a response. Please check the model and API settings."
            st.error(error_msg)
            active_chat["messages"].append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": timestamp
            })

    st.rerun()
