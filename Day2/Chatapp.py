from openai import OpenAI
from Secret import openrouter_key
import streamlit as st
import time
import random
from datetime import datetime

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=openrouter_key,
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables with defaults"""
    defaults = {
        "messages": [{"role": "assistant", "content": "Hello! I'm your chat assistant. How can I help you today?"}],
        "settings": {
            "assistant_name": "My Chat App",
            "response_style": "Friendly",
            "max_history": 50,
            "show_timestamps": True,
            "display_mode": "Dark"
        },
        "stats": {
            "total_messages": 0,
            "session_start": datetime.now()
        }
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize app
initialize_session_state()

# Helper functions
def add_message(role, content):
    """Add a message to chat history with timestamp"""
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    }
    st.session_state.messages.append(message)

    # Trim history if too long
    max_history = st.session_state.settings["max_history"]
    if len(st.session_state.messages) > max_history:
        # Keep first message (greeting) and trim from the middle
        st.session_state.messages = [st.session_state.messages[0]] + st.session_state.messages[-(max_history-1):]

def generate_response(user_input):
    """Fetch response from OpenRouter API model"""
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": user_input}
            ],
            extra_body={"reasoning": {"enabled": True}}
        )
        
        # Extract the response content
        assistant_message = response.choices[0].message.content
        return assistant_message
    
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def summarize_chat():
    """Summarize the entire chat conversation"""
    if len(st.session_state.messages) <= 1:
        return "No chat to summarize yet."
    
    # Create a conversation string for summarization
    conversation = "Here is the chat conversation:\n\n"
    for msg in st.session_state.messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        conversation += f"{role}: {msg['content']}\n\n"
    
    summarization_prompt = f"{conversation}\n\nPlease provide a concise summary of this conversation in 3-5 bullet points."
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {"role": "user", "content": summarization_prompt}
            ],
            extra_body={"reasoning": {"enabled": True}}
        )
        
        summary = response.choices[0].message.content
        return summary
    
    except Exception as e:
        return f"Error summarizing chat: {str(e)}"

# Main content area
assistant_name = st.session_state.settings["assistant_name"]
st.title(f"🚀 {assistant_name}")

# Chat display
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        role_display = "You" if message["role"] == "user" else assistant_name

        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
if prompt := st.chat_input(f"Message {assistant_name}..."):
    # Add user message
    add_message("user", prompt)
    st.session_state.stats["total_messages"] += 1

    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):

        # Show typing indicator and generate response
        with st.spinner(f"{assistant_name} is thinking..."):
            # Generate response from API
            response = generate_response(prompt)
        
        # Display the response
        st.write(response)

        # Add assistant response to history
        add_message("assistant", response)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Display Mode selector
    display_mode = st.selectbox(
        "Display Mode:",
        ["Dark", "Light"],
        index=["Dark", "Light"].index(st.session_state.settings["display_mode"])
    )
    
    # Update display mode in session state
    st.session_state.settings["display_mode"] = display_mode
    
    # Apply theme based on selection
    if display_mode == "Light":
        st.markdown("""
        <style>
        body {
            background-color: #FFFFFF;
            color: #000000;
        }
        .stApp {
            background-color: #FFFFFF;
        }
        .stChatMessage {
            background-color: #F0F0F0;
            border: 1px solid #E0E0E0;
        }
        .stTextInput > div > div {
            background-color: #F5F5F5;
        }
        </style>
        """, unsafe_allow_html=True)
    else:  # Dark mode
        st.markdown("""
        <style>
        body {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        .stChatMessage {
            background-color: #262730;
            border: 1px solid #30363D;
        }
        </style>
        """, unsafe_allow_html=True)
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I'm your chat assistant. How can I help you today?"}
        ]
        st.rerun()
    
    # Summarize chat button
    if st.button("📝 Summarize Chat"):
        with st.spinner("Summarizing chat..."):
            summary = summarize_chat()
            st.session_state.summary = summary
    
    # Display summary if it exists
    if "summary" in st.session_state and st.session_state.summary:
        st.divider()
        st.subheader("📋 Summary")
        st.write(st.session_state.summary)
    
    st.divider()
    
    # Info
    st.subheader("ℹ️ Info")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    st.write(f"**Mode:** {display_mode}")

    if st.button("Export Summary chat"):
        if "summary" in st.session_state and st.session_state.summary:
            st.download_button(
                label="Download Summary",
                data=st.session_state.summary,
                file_name="chat_summary.txt",
                mime="text/plain"
            )
        else:
            st.warning("No summary available to export.")
    
    if st.button("Export Chat History"):
        if st.session_state.messages:
            chat_history = "\n\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in st.session_state.messages])
            st.download_button(
                label="Download Chat History",
                data=chat_history,
                file_name="chat_history.txt",
                mime="text/plain"
            )
        else:
            st.warning("No chat history available to export.")