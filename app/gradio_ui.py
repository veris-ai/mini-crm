"""Gradio UI for Mini CRM Agent with voice support.

Features:
- Model selection (OpenAI, Anthropic, Google, Mistral, DeepSeek)
- User-provided API keys via UI
- Ngrok MCP URL generation
- Voice mode with ElevenLabs TTS and Whisper STT
"""

import os
import tempfile
import uuid

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

# Available frontier models grouped by provider
MODELS = {
    "GPT-4o (OpenAI)": ("gpt-4o", "openai"),
    "GPT-4.1 (OpenAI)": ("gpt-4.1", "openai"),
    "Claude Sonnet 4 (Anthropic)": ("litellm/anthropic/claude-sonnet-4-20250514", "anthropic"),
    "Claude Opus 4 (Anthropic)": ("litellm/anthropic/claude-opus-4-0-20250514", "anthropic"),
    "Gemini 2.0 Flash (Google)": ("litellm/gemini/gemini-2.0-flash", "google"),
    "Gemini 2.5 Pro (Google)": ("litellm/gemini/gemini-2.5-pro-preview-06-05", "google"),
    "Mistral Large (Mistral)": ("litellm/mistral/mistral-large-latest", "mistral"),
    "DeepSeek Chat (DeepSeek)": ("litellm/deepseek/deepseek-chat", "deepseek"),
}

# Map provider to env var name
PROVIDER_KEY_MAP = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# ElevenLabs best voice
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"  # George - deep, warm, natural

# Global state
model_provider = MultiProvider()
ngrok_tunnel = None
current_mcp_url = None


def set_api_key(key_name: str, value: str) -> str:
    """Set API key in environment."""
    if value and value.strip():
        os.environ[key_name] = value.strip()
        return f"{key_name} set"
    return ""


def get_elevenlabs_client(api_key: str = None) -> ElevenLabs | None:
    """Get ElevenLabs client."""
    key = api_key or os.getenv("ELEVENLABS_API_KEY")
    if key:
        return ElevenLabs(api_key=key)
    return None


def get_openai_client(api_key: str = None) -> OpenAI | None:
    """Get OpenAI client for Whisper STT."""
    key = api_key or os.getenv("OPENAI_API_KEY")
    if key:
        return OpenAI(api_key=key)
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


def generate_ngrok_url(ngrok_token: str) -> str:
    """Generate ngrok tunnel URL for MCP."""
    global ngrok_tunnel, current_mcp_url

    # Set token if provided
    if ngrok_token and ngrok_token.strip():
        os.environ["NGROK_AUTH_TOKEN"] = ngrok_token.strip()

    auth_token = os.getenv("NGROK_AUTH_TOKEN")
    if not auth_token:
        return "Error: Please enter your ngrok auth token first"

    # Kill existing tunnel if any
    if ngrok_tunnel:
        try:
            ngrok.disconnect(ngrok_tunnel.public_url)
        except Exception:
            pass

    try:
        ngrok.set_auth_token(auth_token)
        # Create tunnel to FastAPI server (default port 8000)
        port = int(os.getenv("PORT", "8000"))
        ngrok_tunnel = ngrok.connect(port, "http")
        current_mcp_url = f"{ngrok_tunnel.public_url}/mcp"
        return current_mcp_url
    except Exception as e:
        return f"Error: {str(e)}"


def transcribe_audio(audio_path: str, openai_key: str = None) -> str:
    """Transcribe audio using Whisper."""
    client = get_openai_client(openai_key)
    if not client:
        return "[Error: OpenAI API key required for voice transcription]"

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
        )
    return transcription.text


def text_to_speech(text: str, elevenlabs_key: str = None) -> str | None:
    """Convert text to speech using ElevenLabs."""
    client = get_elevenlabs_client(elevenlabs_key)
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


def reset_session() -> list:
    """Reset chat session."""
    global chat_session
    chat_session = ChatSession()
    return []


def get_required_key_for_model(model_choice: str) -> str:
    """Get the required API key name for a model."""
    if model_choice in MODELS:
        _, provider = MODELS[model_choice]
        return PROVIDER_KEY_MAP.get(provider, "")
    return ""


def build_ui() -> gr.Blocks:
    """Build the Gradio UI."""

    with gr.Blocks(title="Mini CRM Agent", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Mini CRM Lead Qualifier Agent")
        gr.Markdown("Chat with a sales assistant powered by your choice of frontier LLM.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Configuration")

                # Model selection
                model_dropdown = gr.Dropdown(
                    choices=list(MODELS.keys()),
                    value="GPT-4o (OpenAI)",
                    label="Select Model",
                    interactive=True,
                )

                # Dynamic hint for required key
                required_key_hint = gr.Markdown("*Requires: `OPENAI_API_KEY`*")

                gr.Markdown("---")
                gr.Markdown("### API Keys")
                gr.Markdown("*Enter your keys below (stored in session only)*")

                with gr.Accordion("LLM Provider Keys", open=True):
                    openai_key = gr.Textbox(
                        label="OpenAI API Key",
                        type="password",
                        placeholder="sk-...",
                        value=os.getenv("OPENAI_API_KEY", ""),
                    )
                    anthropic_key = gr.Textbox(
                        label="Anthropic API Key",
                        type="password",
                        placeholder="sk-ant-...",
                        value=os.getenv("ANTHROPIC_API_KEY", ""),
                    )
                    google_key = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        placeholder="AI...",
                        value=os.getenv("GEMINI_API_KEY", ""),
                    )
                    mistral_key = gr.Textbox(
                        label="Mistral API Key",
                        type="password",
                        placeholder="...",
                        value=os.getenv("MISTRAL_API_KEY", ""),
                    )
                    deepseek_key = gr.Textbox(
                        label="DeepSeek API Key",
                        type="password",
                        placeholder="sk-...",
                        value=os.getenv("DEEPSEEK_API_KEY", ""),
                    )

                with gr.Accordion("Voice & MCP Keys", open=True):
                    elevenlabs_key = gr.Textbox(
                        label="ElevenLabs API Key",
                        type="password",
                        placeholder="sk_...",
                        value=os.getenv("ELEVENLABS_API_KEY", ""),
                        info="Required for voice mode TTS",
                    )
                    ngrok_token = gr.Textbox(
                        label="Ngrok Auth Token",
                        type="password",
                        placeholder="...",
                        value=os.getenv("NGROK_AUTH_TOKEN", ""),
                        info="Required for MCP URL generation",
                    )

                gr.Markdown("---")

                # Ngrok URL generation
                ngrok_btn = gr.Button("Generate MCP URL", variant="secondary")
                ngrok_output = gr.Textbox(
                    label="MCP Endpoint",
                    placeholder="Click button to generate",
                    interactive=False,
                )

                # Voice mode toggle
                voice_toggle = gr.Checkbox(
                    label="Enable Voice Mode",
                    value=False,
                    info="ElevenLabs TTS + Whisper STT",
                )

            with gr.Column(scale=2):
                # Chat interface
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
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
                        label="Voice Input (click to record)",
                        sources=["microphone"],
                        type="filepath",
                    )
                    voice_send_btn = gr.Button("Send Voice", variant="primary")

        # Event handlers

        def update_required_key_hint(model_choice):
            key_name = get_required_key_for_model(model_choice)
            return f"*Requires: `{key_name}`*"

        model_dropdown.change(
            fn=update_required_key_hint,
            inputs=[model_dropdown],
            outputs=[required_key_hint],
        )

        model_dropdown.change(
            fn=reset_session,
            outputs=[chatbot],
        )

        # Save keys to env when changed
        def save_key(key_name):
            def _save(value):
                if value and value.strip():
                    os.environ[key_name] = value.strip()
            return _save

        openai_key.change(fn=save_key("OPENAI_API_KEY"), inputs=[openai_key])
        anthropic_key.change(fn=save_key("ANTHROPIC_API_KEY"), inputs=[anthropic_key])
        google_key.change(fn=save_key("GEMINI_API_KEY"), inputs=[google_key])
        mistral_key.change(fn=save_key("MISTRAL_API_KEY"), inputs=[mistral_key])
        deepseek_key.change(fn=save_key("DEEPSEEK_API_KEY"), inputs=[deepseek_key])
        elevenlabs_key.change(fn=save_key("ELEVENLABS_API_KEY"), inputs=[elevenlabs_key])
        ngrok_token.change(fn=save_key("NGROK_AUTH_TOKEN"), inputs=[ngrok_token])

        ngrok_btn.click(
            fn=generate_ngrok_url,
            inputs=[ngrok_token],
            outputs=ngrok_output,
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

        async def handle_text_submit(message, history, model_choice, voice_enabled, el_key):
            if not message.strip():
                return history, "", None

            global chat_session
            if not chat_session:
                chat_session = ChatSession()

            model_name, _ = MODELS.get(model_choice, ("gpt-4o", "openai"))

            try:
                response = await chat_session.chat(message, model_name)
            except Exception as e:
                response = f"Error: {str(e)}"

            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": response},
            ]

            audio_path = None
            if voice_enabled and not response.startswith("Error:"):
                audio_path = text_to_speech(response, el_key)

            return history, "", audio_path

        send_btn.click(
            fn=handle_text_submit,
            inputs=[text_input, chatbot, model_dropdown, voice_toggle, elevenlabs_key],
            outputs=[chatbot, text_input, audio_output],
        )

        text_input.submit(
            fn=handle_text_submit,
            inputs=[text_input, chatbot, model_dropdown, voice_toggle, elevenlabs_key],
            outputs=[chatbot, text_input, audio_output],
        )

        async def handle_voice_submit(audio_path, history, model_choice, openai_key_val, el_key):
            if not audio_path:
                return history, None

            global chat_session
            if not chat_session:
                chat_session = ChatSession()

            # Transcribe
            transcription = transcribe_audio(audio_path, openai_key_val)
            if transcription.startswith("[Error"):
                history = history + [{"role": "assistant", "content": transcription}]
                return history, None

            model_name, _ = MODELS.get(model_choice, ("gpt-4o", "openai"))

            try:
                response = await chat_session.chat(transcription, model_name)
            except Exception as e:
                response = f"Error: {str(e)}"

            history = history + [
                {"role": "user", "content": f"[Voice] {transcription}"},
                {"role": "assistant", "content": response},
            ]

            audio_response = None
            if not response.startswith("Error:"):
                audio_response = text_to_speech(response, el_key)

            return history, audio_response

        voice_send_btn.click(
            fn=handle_voice_submit,
            inputs=[audio_input, chatbot, model_dropdown, openai_key, elevenlabs_key],
            outputs=[chatbot, audio_output],
        )

    return demo


def main():
    """Launch the Gradio UI."""
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860)


if __name__ == "__main__":
    main()
