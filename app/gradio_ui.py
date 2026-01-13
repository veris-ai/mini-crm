"""Gradio UI for Mini CRM Agent with voice support.

Features:
- Model selection (OpenAI, Anthropic, Google, Mistral, DeepSeek)
- Ngrok MCP URL generation
- Voice mode with ElevenLabs TTS and Whisper STT
"""

import os
import tempfile
import uuid
from typing import Generator

import gradio as gr
from dotenv import load_dotenv
from pyngrok import ngrok
from elevenlabs import ElevenLabs
from openai import OpenAI

from agents import Agent, Runner, RunConfig, RunContextWrapper, TResponseInputItem
from agents.models.multi_provider import MultiProvider

from .tools import get_leads, lookup_lead, score_lead_industry, write_lead_update
from .schema import CRMRunContext

load_dotenv()

# Available frontier models
MODELS = {
    "GPT-4o (OpenAI)": "gpt-4o",
    "GPT-4.1 (OpenAI)": "gpt-4.1",
    "Claude Sonnet 4 (Anthropic)": "litellm/anthropic/claude-sonnet-4-20250514",
    "Claude Opus 4 (Anthropic)": "litellm/anthropic/claude-opus-4-0-20250514",
    "Gemini 2.0 Flash (Google)": "litellm/gemini/gemini-2.0-flash",
    "Gemini 2.5 Pro (Google)": "litellm/gemini/gemini-2.5-pro-preview-06-05",
    "Mistral Large (Mistral)": "litellm/mistral/mistral-large-latest",
    "DeepSeek Chat (DeepSeek)": "litellm/deepseek/deepseek-chat",
}

# ElevenLabs best voice
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - deep, warm, natural

# Global state
model_provider = MultiProvider()
ngrok_tunnel = None
current_mcp_url = None


def get_elevenlabs_client() -> ElevenLabs | None:
    """Get ElevenLabs client if API key is set."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if api_key:
        return ElevenLabs(api_key=api_key)
    return None


def get_openai_client() -> OpenAI | None:
    """Get OpenAI client for Whisper STT."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return OpenAI(api_key=api_key)
    return None


def create_agent(model_name: str) -> Agent:
    """Create agent with specified model."""
    instructions = (
        "You are a sales assistant. Use tools to find leads, apply a simple industry score, "
        "update status/notes, then confirm by listing relevant leads with `get_leads`. "
        "Keep responses short and businesslike."
    )
    return Agent(
        name="Mini CRM Lead Qualifier",
        instructions=instructions,
        tools=[lookup_lead, score_lead_industry, write_lead_update, get_leads],
        model=model_name,
    )


class ChatSession:
    """Manages a chat session with history."""

    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.input_items: list[TResponseInputItem] = []
        self.ctx = RunContextWrapper(CRMRunContext())

    async def chat(self, message: str, model_name: str) -> str:
        """Process a message and return response."""
        self.input_items.append({"content": message, "role": "user"})

        agent = create_agent(model_name)
        result = await Runner.run(
            starting_agent=agent,
            input=self.input_items,
            context=self.ctx.context,
            run_config=RunConfig(model_provider=model_provider),
        )

        self.input_items = result.to_input_list()
        return str(result.final_output) if result.final_output else ""


# Global session (reset on model change)
chat_session: ChatSession | None = None


def generate_ngrok_url() -> str:
    """Generate ngrok tunnel URL for MCP."""
    global ngrok_tunnel, current_mcp_url

    # Kill existing tunnel if any
    if ngrok_tunnel:
        ngrok.disconnect(ngrok_tunnel.public_url)

    # Get ngrok auth token from env
    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if auth_token:
        ngrok.set_auth_token(auth_token)

    # Create tunnel to FastAPI server (default port 8000)
    port = int(os.getenv("PORT", "8000"))
    ngrok_tunnel = ngrok.connect(port, "http")
    current_mcp_url = f"{ngrok_tunnel.public_url}/mcp"

    return f"MCP URL: {current_mcp_url}"


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    client = get_openai_client()
    if not client:
        return "[Error: OPENAI_API_KEY not set for transcription]"

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcription.text


def text_to_speech(text: str) -> str | None:
    """Convert text to speech using ElevenLabs."""
    client = get_elevenlabs_client()
    if not client:
        return None

    # Generate audio
    audio_generator = client.text_to_speech.convert(
        voice_id=ELEVENLABS_VOICE_ID,
        text=text,
        model_id="eleven_multilingual_v2",
    )

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        for chunk in audio_generator:
            f.write(chunk)
        return f.name


async def process_text_message(
    message: str,
    history: list,
    model_choice: str,
    voice_enabled: bool,
) -> Generator:
    """Process text message and optionally return audio."""
    global chat_session

    if not chat_session:
        chat_session = ChatSession()

    model_name = MODELS.get(model_choice, "gpt-4o")

    # Get response from agent
    response = await chat_session.chat(message, model_name)

    # Generate audio if voice mode enabled
    audio_path = None
    if voice_enabled:
        audio_path = text_to_speech(response)

    yield response, audio_path


async def process_voice_message(
    audio_path: str,
    history: list,
    model_choice: str,
) -> tuple[str, str, str | None]:
    """Process voice input and return text response + audio."""
    global chat_session

    if not audio_path:
        return "", "", None

    if not chat_session:
        chat_session = ChatSession()

    # Transcribe audio
    transcription = transcribe_audio(audio_path)

    model_name = MODELS.get(model_choice, "gpt-4o")

    # Get response from agent
    response = await chat_session.chat(transcription, model_name)

    # Generate audio response
    audio_response = text_to_speech(response)

    return transcription, response, audio_response


def reset_session(model_choice: str) -> list:
    """Reset chat session when model changes."""
    global chat_session
    chat_session = ChatSession()
    return []


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""

    with gr.Blocks(title="Mini CRM Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Mini CRM Lead Qualifier Agent")
        gr.Markdown("Chat with a sales assistant powered by your choice of frontier LLM.")

        with gr.Row():
            with gr.Column(scale=1):
                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="GPT-4o (OpenAI)",
                    label="Select Model",
                    interactive=True,
                )

                # Ngrok URL generation
                ngrok_btn = gr.Button("Generate MCP URL", variant="secondary")
                ngrok_output = gr.Textbox(
                    label="MCP Endpoint",
                    placeholder="Click button to generate ngrok URL",
                    interactive=False,
                )

                # Voice mode toggle
                voice_toggle = gr.Checkbox(
                    label="Enable Voice Mode",
                    value=False,
                    info="Use ElevenLabs TTS for responses",
                )

                gr.Markdown("---")
                gr.Markdown("### API Keys Required")
                gr.Markdown("""
                Set in `.env`:
                - `OPENAI_API_KEY` - For OpenAI models & Whisper
                - `ANTHROPIC_API_KEY` - For Claude models
                - `GEMINI_API_KEY` - For Gemini models
                - `ELEVENLABS_API_KEY` - For voice mode
                - `NGROK_AUTH_TOKEN` - For MCP URL generation
                """)

            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=400,
                    type="messages",
                )

                with gr.Row():
                    text_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=4,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)

                # Audio output for voice mode
                audio_output = gr.Audio(
                    label="Voice Response",
                    visible=False,
                    autoplay=True,
                )

                # Voice input
                with gr.Row(visible=False) as voice_input_row:
                    audio_input = gr.Audio(
                        label="Voice Input",
                        sources=["microphone"],
                        type="filepath",
                    )
                    voice_send_btn = gr.Button("Send Voice", variant="primary")

        # Event handlers
        ngrok_btn.click(
            fn=generate_ngrok_url,
            outputs=ngrok_output,
        )

        model_dropdown.change(
            fn=reset_session,
            inputs=[model_dropdown],
            outputs=[chatbot],
        )

        def toggle_voice_ui(enabled: bool):
            return (
                gr.update(visible=enabled),  # audio_output
                gr.update(visible=enabled),  # voice_input_row
            )

        voice_toggle.change(
            fn=toggle_voice_ui,
            inputs=[voice_toggle],
            outputs=[audio_output, voice_input_row],
        )

        async def handle_text_submit(message, history, model_choice, voice_enabled):
            if not message.strip():
                return history, "", None

            global chat_session
            if not chat_session:
                chat_session = ChatSession()

            model_name = MODELS.get(model_choice, "gpt-4o")
            response = await chat_session.chat(message, model_name)

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]

            audio_path = None
            if voice_enabled:
                audio_path = text_to_speech(response)

            return history, "", audio_path

        send_btn.click(
            fn=handle_text_submit,
            inputs=[text_input, chatbot, model_dropdown, voice_toggle],
            outputs=[chatbot, text_input, audio_output],
        )

        text_input.submit(
            fn=handle_text_submit,
            inputs=[text_input, chatbot, model_dropdown, voice_toggle],
            outputs=[chatbot, text_input, audio_output],
        )

        async def handle_voice_submit(audio_path, history, model_choice):
            if not audio_path:
                return history, None

            transcription, response, audio_response = await process_voice_message(
                audio_path, history, model_choice
            )

            history = history + [
                {"role": "user", "content": f"[Voice] {transcription}"},
                {"role": "assistant", "content": response},
            ]

            return history, audio_response

        voice_send_btn.click(
            fn=handle_voice_submit,
            inputs=[audio_input, chatbot, model_dropdown],
            outputs=[chatbot, audio_output],
        )

    return demo


def main():
    """Launch the Gradio UI."""
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
